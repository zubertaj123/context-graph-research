"""In-memory context graph: nodes, edges, provenance envelope.

This is deliberately tiny. It exists so that the measures, the as-of logic and
the graduation state machine can be read in plain Python before you see the
same thing in Cypher and SPARQL.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

import networkx as nx

ENVELOPE_FIELDS = [
    "source_lane", "source_ref", "status", "authority", "risk_tier", "confidence",
    "version", "content_hash", "valid_from", "valid_to", "authored_by", "attested_by", "owner",
]
LANES = ("boomi", "doc", "caselog", "nba", "catalog", "taxonomy", "agent")
STATUSES = ("candidate", "approved", "deprecated", "rejected")
AUTHORITIES = ("canonical", "candidate", "evidence")
TIERS = ("T0", "T1", "T2", "T3")

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def envelope(
    lane: str,
    ref: str,
    *,
    status: str = "approved",
    authority: str = "canonical",
    risk_tier: str = "T0",
    confidence: float | None = None,
    version: int = 1,
    valid_from: datetime = T0,
    valid_to: datetime | None = None,
    authored_by: str = "generator",
    attested_by: str | None = "steward:alice",
    owner: str = "owner:product-line",
    content: Any = None,
) -> dict[str, Any]:
    """Build the 13-field provenance envelope put on every assertion."""
    assert lane in LANES, lane
    assert status in STATUSES, status
    assert authority in AUTHORITIES, authority
    assert risk_tier in TIERS, risk_tier
    if status == "approved" and attested_by is None:
        raise ValueError("an approved assertion must carry attested_by")
    h = hashlib.sha1(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return {
        "source_lane": lane, "source_ref": ref, "status": status, "authority": authority,
        "risk_tier": risk_tier, "confidence": confidence, "version": version, "content_hash": h,
        "valid_from": valid_from, "valid_to": valid_to, "authored_by": authored_by,
        "attested_by": attested_by, "owner": owner,
    }


@dataclass
class Node:
    id: str
    label: str
    layer: str
    props: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, k: str) -> Any:
        return self.props[k]

    def get(self, k: str, default: Any = None) -> Any:
        return self.props.get(k, default)


@dataclass
class Edge:
    src: str
    type: str
    dst: str
    props: dict[str, Any] = field(default_factory=dict)

    def get(self, k: str, default: Any = None) -> Any:
        return self.props.get(k, default)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.type, self.dst)


class ContextGraph:
    """A labelled property graph with one label per node and typed edges with properties."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}
        self.meta: dict[str, Any] = {"graph_version": "ontol0.1+cs0", "promoted_changesets": 0}

    # --- building -------------------------------------------------------
    def add_node(self, label: str, id: str, layer: str, **props: Any) -> Node:
        if id in self.nodes:
            raise KeyError(f"duplicate node id {id}")
        n = Node(id=id, label=label, layer=layer, props=props)
        self.nodes[id] = n
        return n

    def add_edge(self, src: str, type: str, dst: str, **props: Any) -> Edge:
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError(f"edge {src}-{type}->{dst} references an unknown node")
        e = Edge(src=src, type=type, dst=dst, props=props)
        self.edges.append(e)
        self._out.setdefault(src, []).append(e)
        self._in.setdefault(dst, []).append(e)
        return e

    # --- reading --------------------------------------------------------
    def by_label(self, label: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.label == label]

    def by_layer(self, layer: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.layer == layer]

    def out(self, id: str, type: str | None = None, live_at: datetime | None = None) -> list[Edge]:
        es = self._out.get(id, [])
        if type is not None:
            es = [e for e in es if e.type == type]
        if live_at is not None:
            es = [e for e in es if edge_live_at(e, live_at)]
        return es

    def inc(self, id: str, type: str | None = None, live_at: datetime | None = None) -> list[Edge]:
        es = self._in.get(id, [])
        if type is not None:
            es = [e for e in es if e.type == type]
        if live_at is not None:
            es = [e for e in es if edge_live_at(e, live_at)]
        return es

    def targets(self, id: str, type: str, live_at: datetime | None = None) -> list[Node]:
        return [self.nodes[e.dst] for e in self.out(id, type, live_at)]

    def sources(self, id: str, type: str, live_at: datetime | None = None) -> list[Node]:
        return [self.nodes[e.src] for e in self.inc(id, type, live_at)]

    def edge_types(self) -> set[str]:
        return {e.type for e in self.edges}

    def labels(self) -> set[str]:
        return {n.label for n in self.nodes.values()}

    def census(self) -> dict[str, Any]:
        by_layer: dict[str, int] = {}
        for n in self.nodes.values():
            by_layer[n.layer] = by_layer.get(n.layer, 0) + 1
        return {
            "nodes": len(self.nodes), "edges": len(self.edges), "labels": len(self.labels()),
            "edge_types": len(self.edge_types()), "by_layer": by_layer,
        }

    # --- conversions ----------------------------------------------------
    def to_networkx(self, live_at: datetime | None = None) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for n in self.nodes.values():
            g.add_node(n.id, label=n.label, layer=n.layer, **n.props)
        for e in self.edges:
            if live_at is None or edge_live_at(e, live_at):
                g.add_edge(e.src, e.dst, key=e.type, type=e.type, **e.props)
        return g

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes.values())

    def __len__(self) -> int:
        return len(self.nodes)


def edge_live_at(e: Edge, t: datetime) -> bool:
    vf = e.props.get("valid_from")
    vt = e.props.get("valid_to")
    if vf is not None and vf > t:
        return False
    if vt is not None and vt <= t:
        return False
    return True


def node_live_at(n: Node, t: datetime) -> bool:
    vf = n.props.get("valid_from")
    vt = n.props.get("valid_to")
    if vf is not None and vf > t:
        return False
    if vt is not None and vt <= t:
        return False
    return True
