"""The memory graph Gc: entities as nodes, facts as typed edges with provenance, on cgraph.ContextGraph.

Entity resolution: exact normalised match, then RapidFuzz token-sort ratio >= 90 within the same
entity type. Consolidation (per session): invalidate contradicted functional facts (valid_to, never
delete), close a small symmetric predicate set. Hubs (speakers, user, assistant) are tracked so the
measures can exclude them.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cgraph.model import ContextGraph, Edge, Node, edge_live_at  # noqa: E402

from .extract import Triple  # noqa: E402

# Predicate sets come from the extractor's closed vocabulary when it provides one (paper2.extract after
# the critique fixes); the fallback keeps the earlier hand list so the module imports either way.
try:  # pragma: no cover
    from .extract import FUNCTIONAL_PREDICATES as FUNCTIONAL, INVERSE_PREDICATES as INVERSE, SYMMETRIC_PREDICATES as SYMMETRIC  # type: ignore
except ImportError:
    FUNCTIONAL = {"live_in", "lives_in", "moved_to", "work_at", "works_at", "work_as", "works_as", "age", "has_age", "married_to", "dating",
                  "study", "studies", "born_in", "name", "has_hometown", "has_spouse", "has_partner"}
    SYMMETRIC = {"married_to", "friend_of", "sibling_of", "has_sibling", "dating"}
    INVERSE = {"has_child": "child_of", "child_of": "has_child", "parent_of": "child_of"}

# Entity-resolution rules (declared, see docs/00-preregistration.md section 14):
#  R1 exact match on the normalised surface form within a type;
#  R2 speaker names (hubs) resolve regardless of type, so "Melanie" as an object is the speaker node;
#  R3 Person <-> NamedEntity exact matches merge (a named person mentioned as an object);
#  R4 RapidFuzz token-sort ratio >= FUZZY within a type, never for Date, never when either side is a
#     single token shorter than 5 characters (blocks "max"/"man" type merges), length difference <= 12.
FUZZY = 90
NO_FUZZY_TYPES = {"Date"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


class MemoryGraph:
    def __init__(self, hubs: Iterable[str] = ()):
        self.g = ContextGraph()
        self.hubs: set[str] = set()
        self.by_norm: dict[tuple[str, str], str] = {}     # (type, normalised name) -> node id
        self.counter = 0
        self.invalidated = 0
        self.hub_by_norm: dict[str, str] = {}
        for h in hubs:
            nid = self.entity(h, "Person")
            self.hubs.add(nid)
            self.hub_by_norm[norm(h)] = nid

    # ---------------- entities ----------------
    def resolve(self, name: str, etype: str, fuzzy: int = FUZZY) -> str | None:
        n0 = norm(name)
        key = (etype, n0)
        if key in self.by_norm:                                   # R1
            return self.by_norm[key]
        hub = self.hub_by_norm.get(n0)
        if hub is not None:                                       # R2
            return hub
        if etype in ("Person", "NamedEntity"):                    # R3
            other = ("NamedEntity", n0) if etype == "Person" else ("Person", n0)
            if other in self.by_norm:
                return self.by_norm[other]
        if etype in NO_FUZZY_TYPES or (len(n0) < 5 and " " not in n0):
            return None
        best, best_score = None, 0
        for (t, n), nid in self.by_norm.items():                  # R4
            if t != etype or abs(len(n) - len(n0)) > 12 or (len(n) < 5 and " " not in n):
                continue
            sc = fuzz.token_sort_ratio(n, n0)
            if sc > best_score:
                best, best_score = nid, sc
        return best if best_score >= fuzzy else None

    def entity(self, name: str, etype: str) -> str:
        nid = self.resolve(name, etype)
        if nid:
            return nid
        self.counter += 1
        nid = f"e{self.counter}"
        self.g.add_node("Entity", nid, "memory", name=name, type=etype)
        self.by_norm[(etype, norm(name))] = nid
        return nid

    def has_entity(self, name: str, etype: str) -> bool:
        return self.resolve(name, etype) is not None

    # ---------------- facts ----------------
    def edges_live(self, at: datetime | None = None) -> list[Edge]:
        return [e for e in self.g.edges if at is None or edge_live_at(e, at)]

    def has_fact(self, s_id: str, p: str, o_id: str) -> bool:
        return any(e.type == p and e.dst == o_id and e.props.get("valid_to") is None for e in self.g.out(s_id))

    def had_fact(self, s_id: str, p: str, o_id: str) -> bool:
        """True if the fact exists live OR superseded (closure must not resurrect an invalidated mirror)."""
        return any(e.type == p and e.dst == o_id for e in self.g.out(s_id))

    def admit(self, t: Triple) -> Edge:
        s = self.entity(t.s, t.s_type)
        o = self.entity(t.o, t.o_type)
        e = self.g.add_edge(s, t.p, o, source_turns=list(t.source_turns), session=t.session, speaker=t.speaker,
                            observed_at=t.observed_at, tier=t.tier, confidence=t.confidence, method=t.method, text=t.text,
                            valid_from=t.observed_at, valid_to=None, truth=t.truth)
        return e

    def degree(self, nid: str, at: datetime | None = None) -> int:
        return len(self.g.out(nid, live_at=at)) + len(self.g.inc(nid, live_at=at))

    # ---------------- consolidation ----------------
    def consolidate(self, at: datetime | None = None) -> dict[str, int]:
        """(i) invalidate older contradicted functional facts, (ii) symmetric closure. Returns counts."""
        inv = 0
        closed = 0
        for nid in list(self.g.nodes):
            by_pred: dict[str, list[Edge]] = {}
            for e in self.g.out(nid):
                if e.props.get("valid_to") is None and e.type in FUNCTIONAL:
                    by_pred.setdefault(e.type, []).append(e)
            for p, es in by_pred.items():
                if len(es) > 1:
                    es.sort(key=lambda e: (e.props.get("observed_at") or datetime.min))
                    for old in es[:-1]:
                        if old.dst != es[-1].dst:
                            when = es[-1].props.get("observed_at") or at
                            old.props["valid_to"] = when
                            inv += 1
                            # a superseded symmetric fact takes its mirror with it (no ping-pong between sessions)
                            if p in SYMMETRIC:
                                for m in self.g.out(old.dst, p):
                                    if m.dst == old.src and m.props.get("valid_to") is None:
                                        m.props["valid_to"] = when
        for e in list(self.g.edges):
            if e.props.get("valid_to") is not None or e.props.get("method") == "closure":
                continue
            if e.type in SYMMETRIC and not self.had_fact(e.dst, e.type, e.src):
                self.g.add_edge(e.dst, e.type, e.src, **{**e.props, "method": "closure"})
                closed += 1
            inv_p = INVERSE.get(e.type) if isinstance(INVERSE, dict) else None
            if inv_p and not self.had_fact(e.dst, inv_p, e.src):
                self.g.add_edge(e.dst, inv_p, e.src, **{**e.props, "method": "closure"})
                closed += 1
        self.invalidated += inv
        return {"invalidated": inv, "closed": closed}

    # ---------------- views ----------------
    def verbalise(self, e: Edge) -> str:
        if e.props.get("text"):
            return e.props["text"]
        return f"{self.g.nodes[e.src]['name']} {e.type.replace('_', ' ')} {self.g.nodes[e.dst]['name']}"

    def shape(self) -> dict[str, float]:
        """Graph-shape preconditions for any structural claim (critique E1): how much of the live graph is
        more than a star around the speakers."""
        import networkx as nx
        live = [e for e in self.g.edges if e.props.get("valid_to") is None]
        non_hub = {n for n in self.g.nodes if n not in self.hubs}
        deg: dict[str, int] = {}
        nh_edges = 0
        for e in live:
            deg[e.src] = deg.get(e.src, 0) + 1
            deg[e.dst] = deg.get(e.dst, 0) + 1
            if e.src in non_hub and e.dst in non_hub:
                nh_edges += 1
        touched = [n for n in non_hub if deg.get(n, 0) > 0]
        g = nx.Graph()
        g.add_nodes_from(touched)
        g.add_edges_from((e.src, e.dst) for e in live if e.src in non_hub and e.dst in non_hub)
        return {"non_hub_vertices": len(touched),
                "share_non_hub_deg_ge2": (sum(1 for n in touched if deg[n] >= 2) / len(touched)) if touched else 0.0,
                "share_edges_non_hub_to_non_hub": (nh_edges / len(live)) if live else 0.0,
                "non_hub_components": nx.number_connected_components(g) if touched else 0,
                "max_non_hub_degree": max((deg[n] for n in touched), default=0)}

    def stats(self) -> dict[str, int]:
        return {"entities": len(self.g.nodes), "facts": len(self.g.edges), "live_facts": sum(1 for e in self.g.edges if e.props.get("valid_to") is None),
                "invalidated": self.invalidated}
