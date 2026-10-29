"""Time in the context graph: supersede, never delete; answer 'as of' any instant.

Honest limit: edges carry valid_from/valid_to (STARTS_AT, HAS_STEP, NEXT,
ROUTES_TO) but node properties are overwritten in place, so it is edge-level
validity, not full bitemporality. `set_property_versioned` shows the fix and
`declared_vs_extant` in lab 3 will catch that it adds an undeclared pattern.
"""
from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from .model import ContextGraph, Edge, edge_live_at, envelope


def as_of(g: ContextGraph, t: datetime) -> ContextGraph:
    """A view of the graph as it was at instant t (edge-level validity)."""
    v = ContextGraph()
    v.meta = dict(g.meta, as_of=t)
    for n in g.nodes.values():
        v.add_node(n.label, n.id, n.layer, **n.props)
    for e in g.edges:
        if edge_live_at(e, t):
            v.add_edge(e.src, e.type, e.dst, **e.props)
    return v


def canonical_path(g: ContextGraph, flow_id: str, t: datetime | None = None, max_len: int = 20) -> list[str]:
    """Follow STARTS_AT then NEXT edges that are live at t. This is what next_step serves."""
    starts = g.out(flow_id, "STARTS_AT", live_at=t)
    if not starts:
        return []
    path = [starts[0].dst]
    while len(path) < max_len:
        nxt = g.out(path[-1], "NEXT", live_at=t)
        if not nxt:
            break
        path.append(nxt[0].dst)
    return path


def supersede(g: ContextGraph, old: Edge, new_dst: str, at: datetime, changeset: str, lane: str = "agent") -> Edge:
    """Promotion: close the old edge (valid_to) and add its replacement (valid_from). Nothing is deleted."""
    old.props["valid_to"] = at
    old.props["status"] = "deprecated"
    env = envelope(lane, f"intake/{changeset}", status="approved", valid_from=at, content=(old.src, old.type, new_dst))
    env["changeset"] = changeset
    return g.add_edge(old.src, old.type, new_dst, **env)


def set_property_versioned(g: ContextGraph, node_id: str, prop: str, value: Any, at: datetime, changeset: str) -> str:
    """Node-level bitemporality the cheap way: keep every past value as a PropertyVersion node.

    (:Node)-[:HAD_VALUE {valid_from, valid_to}]->(:PropertyVersion {prop, value})
    The current value stays on the node for fast reads; history is one hop away.
    """
    n = g.nodes[node_id]
    prev = n.props.get(prop)
    pv = f"pv:{node_id}:{prop}:{at.date().isoformat()}"
    g.add_node("PropertyVersion", pv, "lifecycle", prop=prop, value=prev,
               **envelope("agent", f"intake/{changeset}", status="deprecated", valid_to=at, content=(node_id, prop, prev)))
    g.add_edge(node_id, "HAD_VALUE", pv, valid_from=n.props.get("valid_from"), valid_to=at)
    n.props[prop] = value
    n.props["version"] = int(n.props.get("version", 1)) + 1
    return pv


def property_as_of(g: ContextGraph, node_id: str, prop: str, t: datetime) -> Any:
    for e in g.out(node_id, "HAD_VALUE"):
        if edge_live_at(e, t):
            return g.nodes[e.dst]["value"]
    return g.nodes[node_id].props.get(prop)


# Kuzu dialect: the filter on a variable-length pattern goes inside the brackets;
# Neo4j would write WHERE ALL(r IN relationships(p) WHERE ...) and [x IN nodes(p) | x.title].
CYPHER_PATH_AS_OF = """
MATCH (f:Flow {id: $flow})-[s:STARTS_AT]->(first:Step)
WHERE s.valid_from <= $t AND (s.valid_to IS NULL OR s.valid_to > $t)
MATCH p = (first)-[n:NEXT*0..10 (r, _ | WHERE r.valid_from <= $t AND (r.valid_to IS NULL OR r.valid_to > $t))]->(last:Step)
RETURN properties(nodes(p), 'title') AS path ORDER BY size(nodes(p)) DESC LIMIT 1"""

SPARQL_NEXT_AS_OF = """
SELECT ?a ?b ?from ?to WHERE {
  ?a tsg:NEXT ?b ~ ?edge .
  ?edge tsg:valid_from ?from .
  OPTIONAL { ?edge tsg:valid_to ?to }
  FILTER(?from <= "%s"^^xsd:dateTime && (!BOUND(?to) || ?to > "%s"^^xsd:dateTime))
  FILTER(STRSTARTS(STR(?a), "%s"))
}"""
