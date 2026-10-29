"""Retrieval, identical for every policy: BM25 over verbalised facts, top-k, then 1-hop expansion over
the admitted graph from the retrieved facts' non-hub entities, capped.

Superseded facts (valid_to set by consolidation) stay retrievable with a validity annotation, because
the store never deletes and temporal questions ask for past values; `include_superseded=False` gives
the live-only variant. Facts observed after the question time are excluded when `at` is given."""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime

import numpy as np

from .graph import MemoryGraph

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = set("a an the of to in on at for and or is are was were be been it its this that with as by from i you he she they we my your his her their our what when where who which how did do does".split())


def tokenize(s: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(s.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.tf = [Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - n_t + 0.5) / (n_t + 0.5)) for t, n_t in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n)
        if not self.n:
            return out
        for t in query:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return out


class Retriever:
    def __init__(self, G: MemoryGraph, k: int = 20, cap: int = 40, expand: bool = True, include_superseded: bool = True):
        self.G, self.k, self.cap, self.expand, self.include_superseded = G, k, cap, expand, include_superseded
        self._sig = None
        self.edges: list = []
        self.bm25: BM25 | None = None

    def _ensure(self, at: datetime | None):
        sig = (len(self.G.g.edges), self.G.invalidated, at, self.include_superseded)
        if sig == self._sig:
            return
        es = [e for e in self.G.g.edges if e.props.get("method") != "closure"]      # closure edges duplicate text
        if at is not None:
            es = [e for e in es if (e.props.get("observed_at") is None or e.props["observed_at"] <= at)]
        if not self.include_superseded:
            es = [e for e in es if e.props.get("valid_to") is None]
        self.edges = es
        self.bm25 = BM25([tokenize(self.G.verbalise(e)) for e in self.edges])
        self._sig = sig

    def text(self, e) -> str:
        """Verbalised fact for the answerer, with a validity note when consolidation superseded it."""
        t = self.G.verbalise(e)
        at = e.props.get("observed_at")
        if at is not None and hasattr(at, "strftime"):                      # the session date, as the official LoCoMo
            t = f"{t} (said on {at:%-d %B %Y})"                             # answer setup gives it; needed for temporal questions
        vt = e.props.get("valid_to")
        if vt is not None:
            return f"{t} [superseded on {vt:%Y-%m-%d}]" if hasattr(vt, "strftime") else f"{t} [superseded]"
        return t

    def retrieve(self, query: str, at: datetime | None = None) -> list:
        self._ensure(at)
        if not self.edges:
            return []
        s = self.bm25.scores(tokenize(query))
        order = [i for i in np.argsort(-s) if s[i] > 0][: self.k]
        picked = [self.edges[i] for i in order]
        if not self.expand or len(picked) >= self.cap:
            return picked[: self.cap]
        chosen = {id(e) for e in picked}
        ents = {v for e in picked for v in (e.src, e.dst) if v not in self.G.hubs}
        extra = []
        for i, e in enumerate(self.edges):
            if id(e) not in chosen and (e.src in ents or e.dst in ents):
                extra.append((s[i], -(e.props.get("observed_at") or datetime.min).timestamp() if e.props.get("observed_at") else 0.0, i))
        extra.sort(key=lambda x: (-x[0], x[1]))
        for _, _, i in extra[: self.cap - len(picked)]:
            picked.append(self.edges[i])
        return picked
