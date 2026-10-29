"""GraphRAG inside the graph, on a laptop: scope by traversal first, then score.

No model download: `embed` is a hashed bag-of-words vector (unigrams + bigrams),
so results are deterministic and run in milliseconds. Swap in sentence-transformers
later; the comparison that matters (unscoped vs scoped vs hybrid) does not depend on it.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import ACTIONS, FLOW_BLUEPRINT, SYMPTOMS, AnswerKey
from .model import ContextGraph

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def embed(text: str, dim: int = 256) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    toks = tokens(text)
    grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
    for t in grams:
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


@dataclass
class Index:
    ids: list[str]
    M: np.ndarray            # rows = chunk vectors, L2-normalised
    pos: dict[str, int]


def build_index(g: ContextGraph) -> Index:
    chunks = g.by_label("Chunk")
    ids = [c.id for c in chunks]
    M = np.stack([embed(c["text"]) for c in chunks])
    return Index(ids=ids, M=M, pos={cid: i for i, cid in enumerate(ids)})


def _fulltext(g: ContextGraph, q: str, cid: str) -> float:
    qt = set(tokens(q))
    ct = set(tokens(g.nodes[cid]["text"]))
    return float(len(qt & ct))


def unscoped(g: ContextGraph, idx: Index, query: str, k: int = 5) -> list[tuple[str, float]]:
    """Plain RAG: top-k cosine over every chunk, any model, locale or revision."""
    s = idx.M @ embed(query)
    top = np.argsort(-s)[:k]
    return [(idx.ids[i], float(s[i])) for i in top]


def scope(g: ContextGraph, model_id: str, locale: str = "en-us", latest_only: bool = True) -> list[str]:
    """The traversal: Model <-ABOUT_PRODUCT- Document(is_latest, locale) -HAS_CHUNK-> Chunk."""
    out: list[str] = []
    for d in g.sources(model_id, "ABOUT_PRODUCT"):
        if (latest_only and not d["is_latest"]) or d["locale"] != locale:
            continue
        out.extend(c.id for c in g.targets(d.id, "HAS_CHUNK"))
    return out


def scoped(g: ContextGraph, idx: Index, query: str, model_id: str, k: int = 5, locale: str = "en-us") -> list[tuple[str, float]]:
    """GraphRAG: scope first (a few hundred chunks), then exact cosine. Never touches an ANN index."""
    cand = scope(g, model_id, locale)
    if not cand:
        return []
    rows = [idx.pos[c] for c in cand]
    s = idx.M[rows] @ embed(query)
    order = np.argsort(-s)[:k]
    return [(cand[i], float(s[i])) for i in order]


def hybrid(g: ContextGraph, idx: Index, query: str, model_id: str, k: int = 5, locale: str = "en-us", w_vec: float = 0.7) -> list[tuple[str, float]]:
    """0.7 * cosine + 0.3 * ft/(ft+3): bounded full-text rescue for codes that embeddings blur."""
    cand = scope(g, model_id, locale)
    rows = [idx.pos[c] for c in cand]
    s = idx.M[rows] @ embed(query)
    blended = []
    for i, c in enumerate(cand):
        ft = _fulltext(g, query, c)
        blended.append((c, w_vec * float(s[i]) + (1 - w_vec) * ft / (ft + 3)))
    return sorted(blended, key=lambda x: -x[1])[:k]


def expand(g: ContextGraph, chunk_ids: list[str]) -> dict[str, Any]:
    """After scoring: which canonical nodes the hits name (MENTIONS -> SAME_AS) and which steps they evidence."""
    entities, steps = set(), set()
    for c in chunk_ids:
        for ent in g.targets(c, "MENTIONS"):
            for canon in g.targets(ent.id, "SAME_AS"):
                entities.add(canon.id)
        for st in g.sources(c, "EVIDENCED_BY"):
            steps.add(st.id)
    return {"canonical": sorted(entities), "steps": sorted(steps)}


def evaluate(g: ContextGraph, key: AnswerKey, idx: Index, k: int = 5) -> dict[str, dict[str, float]]:
    """Score each strategy against the hidden key.

    A query is (model, symptom, step i): 'symptom text + action name'. A hit is correct when the
    returned chunk is a true evidence chunk for that step AND belongs to that model's latest
    en-us documents. We also count the failure modes: wrong model, wrong locale,
    stale revision.
    """
    stats: dict[str, dict[str, float]] = {}
    for name in ("unscoped", "scoped", "hybrid"):
        stats[name] = {"queries": 0, "hit@k": 0, "wrong_model": 0, "wrong_locale": 0, "stale_revision": 0, "mrr": 0.0}
    models = g.by_label("Model")
    for m in models:
        pl = g.nodes[g.out(m.id, "IN_LINE")[0].dst]["name"]
        for sym, blueprint in FLOW_BLUEPRINT.items():
            for i, st in enumerate(blueprint):
                sid = f"step:{pl.lower()}:{sym}:{i+1}"
                truth = {c for c in key.step_evidence.get(sid, []) if g.nodes[c]["model"] == m.id}
                # the device is session scope (service tag -> model), it is NOT in the question text
                query = f"{SYMPTOMS[sym]} {ACTIONS[st['action']][0]}"
                results = {
                    "unscoped": unscoped(g, idx, query, k),
                    "scoped": scoped(g, idx, query, m.id, k),
                    "hybrid": hybrid(g, idx, query, m.id, k),
                }
                for name, hits in results.items():
                    s = stats[name]
                    s["queries"] += 1
                    ids = [h for h, _ in hits]
                    if any(h in truth for h in ids):
                        s["hit@k"] += 1
                        rank = next(r for r, h in enumerate(ids, start=1) if h in truth)
                        s["mrr"] += 1.0 / rank
                    for h in ids:
                        c = g.nodes[h]
                        if c["model"] != m.id:
                            s["wrong_model"] += 1
                        if c["locale"] != "en-us":
                            s["wrong_locale"] += 1
                        if not c["is_latest"]:
                            s["stale_revision"] += 1
    for s in stats.values():
        q = s["queries"] or 1
        s["hit@k"] = round(s["hit@k"] / q, 3)
        s["mrr"] = round(s["mrr"] / q, 3)
        for f in ("wrong_model", "wrong_locale", "stale_revision"):
            s[f] = round(s[f] / (q * k), 3)
    return stats
