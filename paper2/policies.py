"""Admission policies and the gates that turn a score into an admit / reject decision.

Every policy sees the same inputs (memory graph, turn, candidate triples, structural features, and a
dict of extra per-turn features) and returns one score for the turn's candidate set; it never sees gold
evidence. The *gate* is shared: a fixed threshold, or an online quantile that admits roughly the top-q
fraction of scores without look-ahead; exact size matching is done by bisection on q in paper2.run.

Honest labels (critique section D) are in DESCRIPTIONS and must be used in every table:
  heat       length + revisit heuristic (MemoryOS-inspired; MemoryOS uses heat for promotion, not admission)
  amac_lite  A-MAC's feature definitions (ROUGE-L vs prior turns, lexical novelty vs all facts, predicate
             prior) with hand-set weights and no LLM utility call
  amac_fit   the same features with weights fitted by the same learner as the graph gate
  llm_gate   an LLM store/skip judgement with BM25-retrieved context; not Mem0's ADD/UPDATE/DELETE prompt
  siv_2010   the 2010 Trust x Impact product at turn granularity under this extraction; no cold-start rule
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import Conversation, Turn
from .extract import Triple
from .graph import MemoryGraph, norm
from .measures import FEATURE_NAMES, Features, jaccard, toks

PRIORITY_PREDICATES = {"lives_in", "moved_to", "works_at", "works_as", "born_in", "married_to", "dating", "has_child", "has_pet", "has_sibling",
                       "studies", "studied_at", "has_age", "plans_to", "diagnosed_with", "owns", "live_in", "work_at", "work_as", "study",
                       "plan", "move", "adopt", "graduate", "diagnose"}
AMAC_FEATURE_NAMES = ["amac_conf", "amac_nov", "amac_prior"]

DESCRIPTIONS = {
    "admit_all": "admit everything (ceiling)",
    "random": "uniform random score",
    "heat": "length + revisit heuristic (MemoryOS-inspired)",
    "siv_2010": "Trust x Impact, al-Saffar & Heileman 2010, at turn granularity",
    "impact_2010": "Impact alone (2010), ablation",
    "graph_gate": "logistic gate over graph features (prior weights unless fitted weights are given)",
    "amac_lite": "A-MAC feature definitions, hand-set weights, no utility call",
    "amac_fit": "A-MAC feature definitions, weights fitted by the same learner as graph_gate",
    "llm_gate": "LLM store/skip judgement with BM25 context (not Mem0's prompt)",
    "mem0_gate": "Mem0 v1.0.11 DEFAULT_UPDATE_MEMORY_PROMPT verbatim (ADD/UPDATE admit, NONE skip, DELETE counted not applied), BM25 top-5 per fact as old memory",
}


# ---------------------------------------------------------------------------
# A-MAC style features, computed once per turn by the run loop and shared by amac_lite / amac_fit
# ---------------------------------------------------------------------------
class FactTokens:
    """Token sets of live facts, cached by edge identity, for lexical novelty over the whole memory."""

    def __init__(self):
        self.cache: dict[int, set[str]] = {}

    def all(self, G: MemoryGraph) -> list[set[str]]:
        out = []
        for e in G.g.edges:
            if e.props.get("valid_to") is not None:
                continue
            k = id(e)
            if k not in self.cache:
                self.cache[k] = toks(G.verbalise(e))
            out.append(self.cache[k])
        return out


def amac_features(G: MemoryGraph, turn: Turn, triples: Sequence[Triple], prev_turn_texts: Sequence[str], fact_tokens: FactTokens) -> dict:
    """confidence = max ROUGE-L of the candidate sentence against the previous 20 dialogue turns (does the
    candidate recur in prior dialogue); novelty = 1 - max token Jaccard against every live fact; prior =
    share of candidate predicates in the durable-profile list. Recency is constant at write time and omitted."""
    if not triples:
        return {"amac_conf": 0.0, "amac_nov": 0.0, "amac_prior": 0.0}
    texts = [t.text or f"{t.s} {t.p} {t.o}" for t in triples]
    prev = [p.lower().split() for p in list(prev_turn_texts)[-20:] if p]
    conf = 0.0
    for c in texts:
        ct = c.lower().split()
        for pt in prev:
            if ct and pt:
                lcs = _lcs_len(ct, pt)
                if lcs:
                    pr_, rc_ = lcs / len(ct), lcs / len(pt)
                    conf = max(conf, 2 * pr_ * rc_ / (pr_ + rc_))
    facts = fact_tokens.all(G)
    nov = 1.0 - max((jaccard(toks(c), f) for c in texts for f in facts), default=0.0)
    prior = sum(1.0 if t.p in PRIORITY_PREDICATES else 0.0 for t in triples) / len(triples)
    return {"amac_conf": conf, "amac_nov": nov, "amac_prior": prior}


def _lcs_len(a: list[str], b: list[str]) -> int:
    try:
        from rapidfuzz.distance import LCSseq            # C implementation over hashable sequences
        return int(LCSseq.similarity(a, b))
    except ImportError:                                    # pure-Python fallback, same result
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b):
                cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
            prev = cur
        return prev[-1]


def rouge_l(cand: str, ref: str) -> float:
    """ROUGE-L F-measure on whitespace tokens (LCS-based), as in A-MAC's confidence feature."""
    a, b = cand.lower().split(), ref.lower().split()
    if not a or not b:
        return 0.0
    lcs = _lcs_len(a, b)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------
class Policy:
    name = "base"
    fit_convs: list[str] = []

    def reset(self, conv: Conversation) -> None:
        pass

    def score(self, G: MemoryGraph, turn: Turn, triples: Sequence[Triple], feats: Features, extra: dict | None = None) -> float:
        raise NotImplementedError

    def cost(self) -> dict:
        return {"llm_calls": 0, "llm_input_tokens": 0, "llm_output_tokens": 0}


class AdmitAll(Policy):
    name = "admit_all"

    def score(self, G, turn, triples, feats, extra=None) -> float:
        return 1.0


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self, conv):
        self.rng = random.Random(f"{self.seed}:{conv.id}")        # paired across policies and runs

    def score(self, G, turn, triples, feats, extra=None) -> float:
        return self.rng.random()


class HeatPolicy(Policy):
    """heat = a*log(1+visits of the candidate's non-hub entities) + b*(turn length / 50) + c*exp(-dt/mu),
    dt = seconds since those entities were last mentioned (MemoryOS uses mu = 1e7 s). With objects that
    rarely recur this is mostly a turn-length score; see DESCRIPTIONS."""
    name = "heat"

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, mu: float = 1e7):
        self.a, self.b, self.c, self.mu = a, b, c, mu
        self.visits: dict[str, int] = {}
        self.last: dict[str, datetime] = {}

    def reset(self, conv):
        self.visits, self.last = {}, {}

    def score(self, G, turn, triples, feats, extra=None) -> float:
        ents = set()
        for t in triples:
            for name, ty in ((t.s, t.s_type), (t.o, t.o_type)):
                rid = G.resolve(name, ty)
                if rid not in G.hubs:
                    ents.add(rid or norm(name))
        visits = sum(self.visits.get(e, 0) for e in ents)
        dts = [(turn.at - self.last[e]).total_seconds() for e in ents if e in self.last and turn.at]
        rec = math.exp(-max(0.0, min(dts)) / self.mu) if dts else 0.0
        s = self.a * math.log1p(visits) + self.b * (feats.text_len / 50.0) + self.c * rec
        for e in ents:
            self.visits[e] = self.visits.get(e, 0) + 1
            if turn.at:
                self.last[e] = turn.at
        return s


class SIV2010(Policy):
    name = "siv_2010"

    def score(self, G, turn, triples, feats, extra=None) -> float:
        return feats.siv_2010


class Impact2010(Policy):
    name = "impact_2010"

    def score(self, G, turn, triples, feats, extra=None) -> float:
        return feats.impact


class LogisticGate(Policy):
    """sigmoid(w . standardised(x) + b) over a named feature vector. Weights come from `fit_logistic` via
    paper2.run fit; the JSON records the feature names, the standardiser and the conversations it was
    fitted on so a run can refuse to evaluate on them."""
    names: list[str] = FEATURE_NAMES

    def __init__(self, weights_path: str | Path | None = None, prior: dict | None = None, bias: float = -0.5):
        self.w = np.array([(prior or {}).get(n, 0.0) for n in self.names])
        self.b = bias
        self.mean, self.std = np.zeros(len(self.names)), np.ones(len(self.names))
        self.source = "prior"
        self.fit_convs = []
        if weights_path:
            d = json.loads(Path(weights_path).read_text())
            assert d["names"] == self.names, f"weights were fitted on {d['names']}, policy expects {self.names}"
            self.w, self.b = np.array(d["w"]), float(d["b"])
            self.mean, self.std = np.array(d["mean"]), np.array(d["std"])
            self.fit_convs = list(d.get("fit_convs", []))
            self.source = str(weights_path)

    def vector(self, feats: Features, extra: dict | None) -> np.ndarray:
        raise NotImplementedError

    def score(self, G, turn, triples, feats, extra=None) -> float:
        x = np.clip((self.vector(feats, extra) - self.mean) / self.std, -5.0, 5.0)
        return 1.0 / (1.0 + math.exp(-float(x @ self.w + self.b)))


class GraphGate(LogisticGate):
    """Redesigned SIVp. Without a weights file it uses hand-set prior weights on *raw* features. These
    weights were set before any fitting but are NOT in the frozen pre-registration (critique A3.5); on raw
    scales the prior is effectively 0.3*novelty - 2*t_soft - 2*t_edge + 1.5*anchoring + impact_log
    + 0.5*confidence + 0.2*n_edges - 0.5, i.e. a novelty-minus-redundancy score."""
    name = "graph_gate"
    names = FEATURE_NAMES
    PRIOR = {"anchoring": 1.5, "t_soft": -2.0, "t_edge": -2.0, "impact_log": 1.0, "novelty": 0.3, "confidence": 0.5, "n_edges": 0.2}

    def __init__(self, weights_path: str | Path | None = None):
        super().__init__(weights_path, prior=self.PRIOR)

    def vector(self, feats, extra):
        return np.array(feats.vector())


class AMACFit(LogisticGate):
    name = "amac_fit"
    names = AMAC_FEATURE_NAMES

    def __init__(self, weights_path: str | Path | None = None):
        super().__init__(weights_path, prior={"amac_conf": 1.0, "amac_nov": 1.0, "amac_prior": 1.0}, bias=-1.0)

    def vector(self, feats, extra):
        extra = extra or {}
        return np.array([float(extra.get(n, 0.0)) for n in self.names])


class AMACLite(Policy):
    """A-MAC's four-signal score with its LLM utility signal removed and recency (constant at write time)
    dropped: 0.3*confidence + 0.3*novelty + 0.2*prior, weights hand-set by us, not A-MAC's."""
    name = "amac_lite"

    def score(self, G, turn, triples, feats, extra=None) -> float:
        extra = extra or {}
        return 0.3 * extra.get("amac_conf", 0.0) + 0.3 * extra.get("amac_nov", 0.0) + 0.2 * extra.get("amac_prior", 0.0)


class LLMGate(Policy):
    """Ask a chat model whether the candidate facts are worth storing, given the 10 facts most similar to
    the candidate (BM25). Cached on disk; calls and tokens are charged to the write path; parse failures
    are counted and scored 0.5."""
    name = "llm_gate"
    PROMPT = ("You maintain long-term memory for an assistant. Decide whether the following candidate facts, "
              "extracted from one conversational turn, should be stored for future conversations. Store durable, "
              "specific, personal facts; do not store small talk, generic statements or facts already implied by memory.\n\n"
              "Most similar stored facts (may be empty):\n{known}\n\nCandidate facts:\n{cands}\n\n"
              'Answer as JSON: {{"store": true|false, "confidence": 0-1}}')

    def __init__(self, client, max_known: int = 10):
        self.client, self.max_known = client, max_known
        self.conv_id = ""
        self.parse_failures = 0
        self._retriever = None

    def reset(self, conv):
        self.conv_id = conv.id
        self._retriever = None

    def score(self, G, turn, triples, feats, extra=None) -> float:
        from .retrieve import Retriever
        if self._retriever is None or self._retriever.G is not G:
            self._retriever = Retriever(G, k=self.max_known, cap=self.max_known, expand=False, include_superseded=False)
        cands_txt = [t.text or f"{t.s} {t.p} {t.o}" for t in triples]
        known = [G.verbalise(e) for e in self._retriever.retrieve(" ".join(cands_txt))]
        raw = self.client.complete_json(self.PROMPT.format(known="\n".join(f"- {k}" for k in known) or "(none)", cands="\n".join(f"- {c}" for c in cands_txt)),
                                        cache_key=f"gate:{self.conv_id}:{turn.id}", max_tokens=256)
        if not raw or "store" not in raw:
            self.parse_failures += 1
            return 0.5
        c = float(raw.get("confidence", 0.5))
        return c if raw.get("store") else 1.0 - c

    def cost(self) -> dict:
        u = self.client.usage
        return {"llm_calls": u["calls"], "llm_input_tokens": u["input_tokens"], "llm_output_tokens": u["output_tokens"],
                "llm_parse_failures": self.parse_failures}


class Mem0Gate(Policy):
    """Mem0's classic write gate (mem0ai 1.0.11, the last release with it): one call per candidate turn with the
    verbatim DEFAULT_UPDATE_MEMORY_PROMPT (paper2/mem0_prompt.py), old memory = the union of the top-5 BM25
    matches of each candidate fact over the live facts (Mem0: vector search limit=5 per fact, union, dedupe;
    ids "0","1",...), new facts = the candidate texts. Admit when any returned event is ADD or UPDATE
    (score 1.0), skip on NONE (0.0). DELETE events are counted but not applied (consolidation is the shared
    pipeline step, pre-registration 6.4); parse failures count and score 0.0, Mem0's own no-op on a bad
    response. Deviations from Mem0: BM25 instead of embeddings for the old-memory lookup; the Anthropic
    adapter cannot set Mem0's temperature 0.1 (llm.py). Binary score: no threshold to sweep, so the other
    policies are size-matched to this gate's realised memory, not the reverse (pre-registration section 8)."""
    name = "mem0_gate"

    def __init__(self, client, per_fact: int = 5):
        self.client, self.per_fact = client, per_fact
        self.conv_id = ""
        self.parse_failures = 0
        self.events: dict[str, int] = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0, "OTHER": 0}
        self._retriever = None

    def reset(self, conv):
        self.conv_id = conv.id
        self._retriever = None

    def score(self, G, turn, triples, feats, extra=None) -> float:
        from .mem0_prompt import get_update_memory_messages
        from .retrieve import Retriever
        if self._retriever is None or self._retriever.G is not G:
            self._retriever = Retriever(G, k=self.per_fact, cap=self.per_fact, expand=False, include_superseded=False)
        facts = [t.text or f"{t.s} {t.p} {t.o}" for t in triples]
        old_texts: list[str] = []
        for f in facts:
            for e in self._retriever.retrieve(f):
                v = G.verbalise(e)
                if v not in old_texts:
                    old_texts.append(v)
        old = [{"id": str(i), "text": v} for i, v in enumerate(old_texts)]
        raw = self.client.complete_json(get_update_memory_messages(old, facts), cache_key=f"mem0:{self.conv_id}:{turn.id}", max_tokens=1024)
        mem = raw.get("memory") if isinstance(raw, dict) else None
        if not isinstance(mem, list):
            self.parse_failures += 1
            return 0.0
        evs = [str(m.get("event", "")).upper() for m in mem if isinstance(m, dict)]
        for ev in evs:
            self.events[ev if ev in self.events else "OTHER"] += 1
        return 1.0 if any(ev in ("ADD", "UPDATE") for ev in evs) else 0.0

    def cost(self) -> dict:
        u = self.client.usage
        return {"llm_calls": u["calls"], "llm_input_tokens": u["input_tokens"], "llm_output_tokens": u["output_tokens"],
                "llm_parse_failures": self.parse_failures, **{f"mem0_{k.lower()}": v for k, v in self.events.items()}}


REGISTRY = {c.name: c for c in (AdmitAll, RandomPolicy, HeatPolicy, SIV2010, Impact2010, GraphGate, AMACLite, AMACFit, LLMGate, Mem0Gate)}


def make_policy(name: str, seed: int = 0, weights: str | None = None, client=None) -> Policy:
    if name == "random":
        return RandomPolicy(seed)
    if name == "graph_gate":
        return GraphGate(weights)
    if name == "amac_fit":
        return AMACFit(weights)
    if name == "llm_gate":
        if client is None:
            raise ValueError("llm_gate needs --gate-model")
        return LLMGate(client)
    if name == "mem0_gate":
        if client is None:
            raise ValueError("mem0_gate needs --gate-model")
        return Mem0Gate(client)
    return REGISTRY[name]()


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
class FixedThreshold:
    def __init__(self, tau: float):
        self.tau = tau

    def reset(self):
        pass

    def admit(self, score: float) -> bool:
        return score >= self.tau

    def describe(self) -> str:
        return f"threshold>={self.tau}"


class OnlineQuantile:
    """Admit when the score is above the running quantile of scores seen so far in this conversation, so
    that every policy admits roughly a fraction q of candidate turns without look-ahead.

    Corrections that keep realised admit rates comparable across policies (critique A5, B(iii)):
      warm-up  the first `warmup` candidates are decided by alternation (admit while the realised rate is
               below q), identically for every policy, instead of being admitted unconditionally;
      ties     a score exactly at the cut is admitted only while the realised rate is below q, so a
               degenerate scorer (the 2010 product is 0 on ~89% of candidates) admits a first-come
               fraction q instead of everything;
      control  the quantile level is nudged by the gap between q and the realised rate;
      band     a score that trends monotonically with graph size defeats any quantile rule, so when the
               realised rate leaves [q - band, q + band] the gate admits (or rejects) unconditionally
               until it is back inside.
    Exact size matching (live facts within a tolerance) is done by bisection on q in paper2.run."""

    def __init__(self, q: float, warmup: int = 20, control: float = 1.0, window: int | None = None, band: float = 0.05):
        self.q, self.warmup, self.control, self.window, self.band = q, warmup, control, window, band
        self.hist: list[float] = []
        self.admitted = 0

    def reset(self):
        self.hist, self.admitted = [], 0

    def realised(self) -> float:
        return self.admitted / len(self.hist) if self.hist else 0.0

    def admit(self, score: float) -> bool:
        prev = self.hist[:]
        self.hist.append(score)
        rate = self.admitted / len(prev) if prev else 0.0
        if len(prev) < self.warmup:
            ok = rate < self.q
        elif rate < self.q - self.band:
            ok = True
        elif rate > self.q + self.band:
            ok = False
        else:
            q_eff = min(0.98, max(0.02, self.q + self.control * (self.q - rate)))
            ref = prev[-self.window:] if self.window else prev
            cut = float(np.quantile(ref, 1.0 - q_eff))
            ok = score > cut or (score == cut and rate < self.q)
        self.admitted += int(ok)
        return ok

    def describe(self) -> str:
        return f"online-quantile q={self.q:.3f}" + (f" w={self.window}" if self.window else "")


# ---------------------------------------------------------------------------
# fitting (numpy, no sklearn)
# ---------------------------------------------------------------------------
def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, epochs: int = 2000, lr: float = 0.05) -> tuple[np.ndarray, float]:
    """Full-batch gradient descent, inverse-frequency class weights, L2 = 1e-2, 2000 steps, lr 0.05
    (declared hyper-parameters, critique A5.5). Stops early when the gradient norm is below 1e-6."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    pos = max(1, int(y.sum()))
    cw = np.where(y == 1, (n - pos) / pos, 1.0)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = (p - y) * cw
        gw, gb = X.T @ g / n + l2 * w, g.mean()
        w -= lr * gw
        b -= lr * gb
        if np.sqrt((gw ** 2).sum() + gb ** 2) < 1e-6:
            break
    return w, b


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _standardiser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = X.mean(0), X.std(0)
    keep = std > 1e-6
    return mean, np.where(keep, std, 1.0), keep


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5, seed: int = 0, exclude: Sequence[str] = (),
                   names: Sequence[str] = FEATURE_NAMES) -> dict:
    """Grouped k-fold (groups = conversation ids) as a diagnostic AUC, then one fit on all rows. `exclude`
    names features forced to weight 0. The result records the conversations it was fitted on."""
    names = list(names)
    rng = np.random.default_rng(seed)
    ug = np.array(sorted(set(groups)))
    rng.shuffle(ug)
    folds = [set(ug[i::k]) for i in range(k)]
    drop = np.array([n in set(exclude) for n in names])
    aucs = []
    for f in folds:
        te = np.array([g in f for g in groups])
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        mean, std, keep = _standardiser(X[~te])
        w, b = fit_logistic(((X[~te] - mean) / std) * (keep & ~drop), y[~te])
        w = w * (keep & ~drop)
        aucs.append(auc(y[te], ((X[te] - mean) / std) @ w + b))
    mean, std, keep = _standardiser(X)
    w, b = fit_logistic(((X - mean) / std) * (keep & ~drop), y)
    w = w * (keep & ~drop)
    return {"names": names, "w": w.tolist(), "b": float(b), "mean": mean.tolist(), "std": std.tolist(), "excluded": list(exclude),
            "constant": [n for n, kp in zip(names, keep) if not kp], "fit_convs": sorted(set(map(str, groups))),
            "cv_auc": [float(a) for a in aucs], "cv_auc_mean": float(np.nanmean(aucs)) if aucs else float("nan"),
            "n": int(len(y)), "positives": int(y.sum()),
            "hyperparameters": {"l2": 1e-2, "epochs": 2000, "lr": 0.05, "class_weight": "inverse frequency"}}
