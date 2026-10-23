"""Structural measures on a memory graph Gc and a candidate Gt, all LLM-free, all linear time.

Trust / Impact / semantic information value follow al-Saffar & Heileman (2010), "Semantic
information value for ontology-based knowledge management". Their exact definitions (read from the
PDF, see docs/reads/measure-lineage.md): Trust = |Et ∩ Ec| / |Et|; Impact = Σ deg(overlap vertices) /
Σ deg(all Gc vertices); SIVp = Trust × Impact, with edge weights w = t · degree(v) for edges touching
overlap vertices and 1 otherwise; SIVa = (|Ec'| − |Ec|) / |Ec'| with Gc' = Gt ∪ Gc, i.e. an UNWEIGHTED
new-edge ratio under plain union. In this module `siva(..., weights="plain", consolidate=False)` is the
2010 SIVa; `siva(..., weights="degree", consolidate=True)` is this paper's consolidation-aware,
degree-weighted redesign (recorded as `siva_cons`), and `siva_scaled` is its size-invariant companion.
The anchoring, soft-agreement, log-degree impact and novelty variants are also this paper's. Nothing
here mutates the graph; SIVa is computed analytically as the change that admission (plus consolidation
for the redesign) *would* realise.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Sequence

from .data import Turn
from .extract import Triple
from .graph import FUNCTIONAL, SYMMETRIC, MemoryGraph, norm

TOKEN_RE = re.compile(r"[a-z0-9]+")
FEATURE_NAMES = ["t_edge", "anchoring", "t_soft", "impact", "impact_log", "novelty", "novelty_pred",
                 "n_edges", "n_new_vertices", "n_nonhub", "confidence", "recency", "text_len"]
# The graph-structural subset (critique A3.4): nothing about how many observations were written for the
# turn (n_edges, n_new_vertices, n_nonhub), nothing about the extractor (confidence), nothing about the
# dialogue surface (text_len) or the clock (recency).
STRUCTURAL = ["t_edge", "anchoring", "t_soft", "impact", "impact_log", "novelty", "novelty_pred"]
NON_STRUCTURAL = [n for n in FEATURE_NAMES if n not in STRUCTURAL]


def toks(s: str) -> set[str]:
    return set(TOKEN_RE.findall(s.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _live(e) -> bool:
    return e.props.get("valid_to") is None


class GraphView:
    """One pass over the live edges: degrees, extant-ontology pattern counts, predicate counts."""

    def __init__(self, G: MemoryGraph):
        self.G = G
        self.deg: dict[str, int] = defaultdict(int)
        self.patterns: Counter = Counter()
        self.preds: Counter = Counter()
        self.n_live = 0
        self.last_seen: dict[str, datetime] = {}
        nodes = G.g.nodes
        for e in G.g.edges:
            if not _live(e):
                continue
            self.n_live += 1
            self.deg[e.src] += 1
            self.deg[e.dst] += 1
            self.patterns[(nodes[e.src].props.get("type"), e.type, nodes[e.dst].props.get("type"))] += 1
            self.preds[e.type] += 1
            at = e.props.get("observed_at")
            if at is not None and (e.src not in self.last_seen or self.last_seen[e.src] < at):
                self.last_seen[e.src] = at
        self.log_denom = sum(math.log1p(d) for v, d in self.deg.items() if v not in G.hubs)

    def weight(self, e, scheme: str = "degree") -> float:
        """Edge weight t x deg(v) (the 2010 SIVp weighting): confidence times the larger endpoint degree;
        scheme "plain" gives unit weights."""
        if scheme == "plain":
            return 1.0
        return float(e.props.get("confidence", 1.0)) * max(self.deg.get(e.src, 0), self.deg.get(e.dst, 0), 1)


@dataclass
class Features:
    t_edge: float
    anchoring: float
    t_soft: float
    impact: float
    impact_log: float
    novelty: float
    novelty_pred: float
    n_edges: int
    n_new_vertices: int
    n_nonhub: int                 # non-hub candidate vertices; anchoring is undefined (recorded 0) when this is 0
    confidence: float
    recency: float
    text_len: int

    @property
    def siv_2010(self) -> float:
        return self.t_edge * self.impact

    def vector(self) -> list[float]:
        return [float(getattr(self, n)) for n in FEATURE_NAMES]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["siv_2010"] = self.siv_2010
        return d


# ---------------------------------------------------------------------------
# individual measures
# ---------------------------------------------------------------------------
def _resolved(G: MemoryGraph, triples: Sequence[Triple]) -> list[tuple[str | None, str | None]]:
    return [(G.resolve(t.s, t.s_type), G.resolve(t.o, t.o_type)) for t in triples]


def candidate_vertices(triples: Sequence[Triple]) -> set[tuple[str, str]]:
    return {(t.s, t.s_type) for t in triples} | {(t.o, t.o_type) for t in triples}


def trust_edge(G: MemoryGraph, triples: Sequence[Triple], res=None) -> float:
    """2010 Trust: share of candidate edges already present (live) in memory. |Et ∩ Ec| / |Et|."""
    if not triples:
        return 0.0
    res = res or _resolved(G, triples)
    hit = sum(1 for t, (s, o) in zip(triples, res) if s and o and G.has_fact(s, t.p, o))
    return hit / len(triples)


def anchoring(G: MemoryGraph, triples: Sequence[Triple]) -> tuple[float, int, int]:
    """Share of non-hub candidate vertices already known (NaN when the candidate has no non-hub vertex,
    critique A3.2), the count of brand-new vertices, and the number of non-hub vertices."""
    verts = candidate_vertices(triples)
    ids = {(n, ty): G.resolve(n, ty) for n, ty in verts}
    non_hub = [k for k, v in ids.items() if v not in G.hubs]
    new = sum(1 for k in non_hub if ids[k] is None)
    if not non_hub:
        return float("nan"), 0, 0
    return (len(non_hub) - new) / len(non_hub), new, len(non_hub)


def soft_agreement(G: MemoryGraph, triples: Sequence[Triple], res=None) -> float:
    """Mean over candidate edges of the best token-set Jaccard against live facts with the same subject."""
    if not triples:
        return 0.0
    res = res or _resolved(G, triples)
    nodes = G.g.nodes
    tot = 0.0
    for t, (s, _) in zip(triples, res):
        if s is None:
            continue
        cand = toks(t.p.replace("_", " ") + " " + t.o)
        best = 0.0
        for e in G.g.out(s):
            if not _live(e):
                continue
            best = max(best, jaccard(cand, toks(e.type.replace("_", " ") + " " + nodes[e.dst].props["name"])))
            if best >= 1.0:
                break
        tot += best
    return tot / len(triples)


def impact_2010(view: GraphView, triples: Sequence[Triple], res=None) -> float:
    """2010 Impact: degree mass of the shared vertices over the degree mass of memory."""
    if view.n_live == 0:
        return 0.0
    res = res or _resolved(view.G, triples)
    shared = {x for pair in res for x in pair if x is not None}
    return sum(view.deg.get(v, 0) for v in shared) / (2 * view.n_live)


def impact_log(view: GraphView, triples: Sequence[Triple], res=None) -> float:
    """Redesign: log-degree mass of shared non-hub vertices over the same for memory."""
    if view.log_denom <= 0:
        return 0.0
    res = res or _resolved(view.G, triples)
    shared = {x for pair in res for x in pair if x is not None and x not in view.G.hubs}
    return sum(math.log1p(view.deg.get(v, 0)) for v in shared) / view.log_denom


def novelty(view: GraphView, triples: Sequence[Triple]) -> tuple[float, float]:
    """Mean surprisal of the candidate's (type(s), p, type(o)) patterns and of its predicates under
    the current extant ontology, add-one smoothed."""
    if not triples:
        return 0.0, 0.0
    vp, vq = len(view.patterns) + 1, len(view.preds) + 1
    n = sum(-math.log((view.patterns[(t.s_type, t.p, t.o_type)] + 1) / (view.n_live + vp)) for t in triples) / len(triples)
    npred = sum(-math.log((view.preds[t.p] + 1) / (view.n_live + vq)) for t in triples) / len(triples)
    return n, npred


def recency_hours(view: GraphView, triples: Sequence[Triple], at: datetime | None, res=None) -> float:
    """log(1 + hours since the subject was last mentioned); 0 when unknown."""
    if at is None:
        return 0.0
    res = res or _resolved(view.G, triples)
    hs = []
    for (s, _) in res:
        if s and s in view.last_seen:
            hs.append(max(0.0, (at - view.last_seen[s]).total_seconds() / 3600))
    return math.log1p(sum(hs) / len(hs)) if hs else 0.0


def features(G: MemoryGraph, triples: Sequence[Triple], turn: Turn | None = None, view: GraphView | None = None) -> Features:
    view = view or GraphView(G)
    res = _resolved(G, triples)
    anc, new_v, n_nonhub = anchoring(G, triples)
    nov, novp = novelty(view, triples)
    text = turn.text if turn is not None else " ".join(t.text for t in triples)
    return Features(
        t_edge=trust_edge(G, triples, res), anchoring=0.0 if anc != anc else anc, t_soft=soft_agreement(G, triples, res),
        impact=impact_2010(view, triples, res), impact_log=impact_log(view, triples, res),
        novelty=nov, novelty_pred=novp, n_edges=len(triples), n_new_vertices=new_v, n_nonhub=n_nonhub,
        confidence=(sum(t.confidence for t in triples) / len(triples)) if triples else 0.0,
        recency=recency_hours(view, triples, turn.at if turn else None, res), text_len=len(TOKEN_RE.findall(text.lower())),
    )


# ---------------------------------------------------------------------------
# SIVa: the structural change that admission + consolidation would realise
# ---------------------------------------------------------------------------
def siva(G: MemoryGraph, triples: Sequence[Triple], view: GraphView | None = None, weights: str = "degree", consolidate: bool = True) -> float:
    """SIVa = (w(Ec' \\ Ec) + w(invalidated)) / w(Ec') with Gc' = c(Gc ∪ Gt), computed without mutating G.
    weights="plain", consolidate=False: the 2010 definition (unweighted new-edge ratio, plain union).
    weights="degree", consolidate=True: this paper's redesign; w(e) = t x deg(v) for edges touching an
    existing vertex, 1 for a new component, and invalidated edges count as change."""
    view = view or GraphView(G)
    live = [e for e in G.g.edges if _live(e)]
    mass_old = sum(view.weight(e, weights) for e in live)
    new_mass = inval_mass = 0.0
    seen: set = set()
    invalidated: set[int] = set()
    sim_new: set[tuple] = set()
    for t in triples:
        s, o = G.resolve(t.s, t.s_type), G.resolve(t.o, t.o_type)
        key = (s or ("new", norm(t.s)), t.p, o or ("new", norm(t.o)))
        if key in seen or (s and o and G.has_fact(s, t.p, o)):
            continue
        seen.add(key)
        if weights == "plain" or (s is None and o is None):
            w = 1.0
        else:
            w = t.confidence * max(view.deg.get(s, 0) if s else 0, view.deg.get(o, 0) if o else 0, 1)
        new_mass += w
        if not consolidate:
            continue
        if t.p in FUNCTIONAL and s is not None:
            for e in G.g.out(s, t.p):
                if _live(e) and e.dst != o and id(e) not in invalidated:
                    invalidated.add(id(e))
                    inval_mass += view.weight(e, weights)
        if t.p in SYMMETRIC and s and o and not G.has_fact(o, t.p, s) and (o, t.p, s) not in sim_new:
            sim_new.add((o, t.p, s))
            new_mass += w
    denom = mass_old - inval_mass + new_mass
    return (new_mass + inval_mass) / denom if denom > 0 else 0.0


def siva_scaled(G: MemoryGraph, triples: Sequence[Triple], view: GraphView | None = None, weights: str = "degree", consolidate: bool = True) -> float:
    """Size-invariant companion to `siva` (critique A4.2, E2): the realised change in units of the mean
    live-edge weight, i.e. siva x |Ec'|. For an empty memory it equals the number of new edges."""
    view = view or GraphView(G)
    n_after = view.n_live + len({(t.s, t.p, t.o) for t in triples})
    return siva(G, triples, view, weights, consolidate) * max(n_after, 1)


def snapshot(G: MemoryGraph, weights: str = "degree") -> dict:
    """Live-edge weights and degrees before a session, for the realised (not simulated) session-level SIVa."""
    view = GraphView(G)
    return {"w": {id(e): view.weight(e, weights) for e in G.g.edges if _live(e)}, "scheme": weights, "deg": dict(view.deg)}


def realised_change(snap: dict, G: MemoryGraph) -> dict:
    """Session-level SIVa from a snapshot taken before the session's admissions and consolidation. All
    weights use the before-session degree basis (critique A4.5): new edges attached to a vertex that
    already existed weigh confidence x its before-degree; edges between new vertices weigh 1."""
    before, deg0, scheme = snap["w"], snap["deg"], snap["scheme"]
    conf = lambda e: 1.0 if scheme == "plain" else float(e.props.get("confidence", 1.0))

    def w_new(e):
        if scheme == "plain":
            return 1.0
        d = max(deg0.get(e.src, 0), deg0.get(e.dst, 0))
        return conf(e) * d if d > 0 else 1.0
    new = inval = 0.0
    n_new = n_inval = 0
    live_ids = set()
    mass_after = 0.0
    for e in G.g.edges:
        if _live(e):
            live_ids.add(id(e))
            if id(e) in before:
                mass_after += before[id(e)]
            else:
                w = w_new(e)
                new += w
                mass_after += w
                n_new += 1
    for eid, w in before.items():
        if eid not in live_ids:
            inval += w
            n_inval += 1
    return {"siva": (new + inval) / mass_after if mass_after > 0 else 0.0, "siva_scaled": (new + inval) / (mass_after / len(live_ids)) if live_ids else 0.0,
            "new_edges": n_new, "invalidated": n_inval, "live_edges": len(live_ids)}
