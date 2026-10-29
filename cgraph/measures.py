"""Structural and information-theoretic measures over a context graph.

These are the measures al-Saffar's 2010-2011 papers are built on:

  extant_ontology          class -> predicate -> class count graph mined from instances
  declared_vs_extant       drift between the YAML contract and what the data actually does
  coverage_curve           how many relation types cover 50 / 80 / 95 % of edges
  link_bigram_information  I(r1,r2) = log( p(r1,r2) / (p(r1) p(r2)) ) over consecutive edge pairs
  path_shapes              the highest-gain 2- and 3-hop shapes = candidate retrieval templates
  semantic_information_value  trust x impact of a set of new claims against existing memory (iiWAS'10, simplified)
  reification_bloat        triple overhead of each RDF annotation style
  layer_crossing           how much of the graph is bridge edges

Every function also has a SPARQL or Cypher twin in `QUERIES` so you can run the
same measure in each engine and compare.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable

from .model import ContextGraph, edge_live_at
from .ontology import declared_patterns, load_ontology


def _live(g: ContextGraph, live_at: datetime | None):
    return [e for e in g.edges if live_at is None or edge_live_at(e, live_at)]


# ---------------------------------------------------------------------------
# extant ontology and drift
# ---------------------------------------------------------------------------
def extant_ontology(g: ContextGraph, live_at: datetime | None = None) -> Counter:
    """(from_label, REL, to_label) -> count. al-Saffar (WI-IAT 2011) calls this the
    extant ontology: the schema the data actually exhibits, as opposed to the declared one."""
    c: Counter = Counter()
    for e in _live(g, live_at):
        c[(g.nodes[e.src].label, e.type, g.nodes[e.dst].label)] += 1
    return c


def declared_vs_extant(g: ContextGraph, onto: dict[str, Any] | None = None) -> dict[str, Any]:
    onto = onto or load_ontology()
    declared = declared_patterns(onto)
    extant = extant_ontology(g)
    used = set(extant)
    return {
        "declared": len(declared),
        "extant": len(used),
        "declared_unused": sorted(declared - used),       # vocabulary nobody uses (dead contract)
        "extant_undeclared": sorted(used - declared),     # patterns nobody declared (drift / violation)
        "coverage_of_declaration": round(len(declared & used) / len(declared), 3) if declared else 0.0,
    }


# ---------------------------------------------------------------------------
# coverage curve
# ---------------------------------------------------------------------------
def coverage_curve(g: ContextGraph, by: str = "type", live_at: datetime | None = None) -> dict[str, Any]:
    """Sort relation types (or extant patterns) by frequency; report the cumulative share.
    al-Saffar's observation: a handful of types cover most edges (16 classes -> 80% in his corpora)."""
    if by == "type":
        counts = Counter(e.type for e in _live(g, live_at))
    else:
        counts = extant_ontology(g, live_at)
    total = sum(counts.values())
    curve: list[tuple[Any, int, float]] = []
    cum = 0
    for k, n in counts.most_common():
        cum += n
        curve.append((k, n, round(cum / total, 4)))
    def k_for(th: float) -> int:
        for i, (_, _, c) in enumerate(curve, start=1):
            if c >= th:
                return i
        return len(curve)
    return {"total_edges": total, "distinct": len(counts), "k50": k_for(0.5), "k80": k_for(0.8), "k95": k_for(0.95), "curve": curve}


# ---------------------------------------------------------------------------
# link-type bigram information gain (KDD-MDS 2011 style)
# ---------------------------------------------------------------------------
def link_bigrams(g: ContextGraph, live_at: datetime | None = None) -> Counter:
    """Count (r1, r2) for every path a -r1-> x -r2-> b. The shared node x is what makes it a bigram."""
    inc: dict[str, list[str]] = defaultdict(list)
    out: dict[str, list[str]] = defaultdict(list)
    for e in _live(g, live_at):
        out[e.src].append(e.type)
        inc[e.dst].append(e.type)
    c: Counter = Counter()
    for x in g.nodes:
        for r1 in inc.get(x, ()):
            for r2 in out.get(x, ()):
                c[(r1, r2)] += 1
    return c


def link_bigram_information(g: ContextGraph, live_at: datetime | None = None, min_count: int = 2) -> list[dict[str, Any]]:
    """Pointwise mutual information of consecutive link types.

    I(r1,r2) = log2( p(r1,r2) / (p(r1) * p(r2)) ), with p(r1) the share of bigrams whose first
    link is r1 and p(r2) the share whose second link is r2. Positive = the pair co-occurs more
    than its parts predict = a *motif* worth having a retrieval template for.
    """
    bg = link_bigrams(g, live_at)
    N = sum(bg.values())
    first: Counter = Counter()
    second: Counter = Counter()
    for (r1, r2), n in bg.items():
        first[r1] += n
        second[r2] += n
    rows = []
    for (r1, r2), n in bg.items():
        if n < min_count:
            continue
        p12, p1, p2 = n / N, first[r1] / N, second[r2] / N
        rows.append({"r1": r1, "r2": r2, "count": n, "pmi_bits": round(math.log2(p12 / (p1 * p2)), 3),
                     "p12": round(p12, 5), "p1": round(p1, 5), "p2": round(p2, 5)})
    return sorted(rows, key=lambda r: (-r["pmi_bits"], -r["count"]))


def path_shapes(g: ContextGraph, length: int = 3, live_at: datetime | None = None, top: int = 15) -> list[tuple[tuple[str, ...], int]]:
    """Most frequent typed paths of `length` hops (label-agnostic, type sequence only).
    Compare the top shapes with the retrieval templates in cgraph.retrieval: they should coincide."""
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in _live(g, live_at):
        out[e.src].append((e.type, e.dst))
    c: Counter = Counter()

    def walk(node: str, seq: tuple[str, ...]) -> None:
        if len(seq) == length:
            c[seq] += 1
            return
        for t, nxt in out.get(node, ()):
            walk(nxt, seq + (t,))

    for n in g.nodes:
        walk(n, ())
    return c.most_common(top)


# ---------------------------------------------------------------------------
# semantic information value (al-Saffar & Joslyn, iiWAS 2010; simplified reconstruction)
# ---------------------------------------------------------------------------
def semantic_information_value(memory: Iterable[tuple[str, str, str]], claims: Iterable[tuple[str, str, str]]) -> dict[str, float]:
    """Value a set of new claims (s, p, o) against what the agent already holds.

    trust  = |claims already in memory| / |claims|           (agreement with what we know)
    impact = degree-weighted share of claim endpoints that touch memory (how connected the new
             knowledge is to what we already have; isolated facts have low impact)
    value  = trust * impact

    This is a *reconstruction* of the iiWAS'10 idea for teaching; read the paper for the
    exact formulation before you cite it.
    """
    mem = set(memory)
    cl = list(claims)
    if not cl:
        return {"trust": 0.0, "impact": 0.0, "value": 0.0}
    agreed = sum(1 for c in cl if c in mem)
    deg: Counter = Counter()
    for s, _, o in mem:
        deg[s] += 1
        deg[o] += 1
    touch = sum(deg[s] + deg[o] for s, _, o in cl)
    max_touch = sum(2 * max(deg.values(), default=1) for _ in cl)
    trust = agreed / len(cl)
    impact = touch / max_touch if max_touch else 0.0
    return {"trust": round(trust, 3), "impact": round(impact, 3), "value": round(trust * impact, 4), "new_claims": len(cl) - agreed}


# ---------------------------------------------------------------------------
# RDF hygiene
# ---------------------------------------------------------------------------
def reification_bloat(g: ContextGraph) -> dict[str, dict[str, Any]]:
    from . import rdf
    out: dict[str, dict[str, Any]] = {}
    base = None
    for style in ("plain", "reification", "rdfstar", "reifier"):
        store = rdf.load_store(rdf.to_turtle(g, style))
        n = len(store)
        base = base or n
        out[style] = {"triples": n, "overhead_pct": round(100.0 * (n - base) / base, 1), "literal_share": round(rdf.literal_share(store), 3)}
    annotated = sum(1 for e in g.edges if any(v is not None for v in e.props.values()))
    out["_edges_with_properties"] = {"count": annotated, "share_of_edges": round(annotated / len(g.edges), 3)}
    return out


def layer_crossing(g: ContextGraph) -> dict[str, Any]:
    cross: Counter = Counter()
    total = 0
    for e in g.edges:
        total += 1
        a, b = g.nodes[e.src].layer, g.nodes[e.dst].layer
        if a != b:
            cross[e.type] += 1
    n = sum(cross.values())
    return {"cross_layer_edges": n, "share": round(n / total, 4), "by_type": cross.most_common()}


# ---------------------------------------------------------------------------
# the same measures as queries (run them in lab 5)
# ---------------------------------------------------------------------------
QUERIES = {
    "extant_ontology_cypher": """
MATCH (a)-[r]->(b)
RETURN label(a) AS from_label, label(r) AS rel, label(b) AS to_label, count(*) AS n
ORDER BY n DESC""",
    "extant_ontology_sparql": """
SELECT ?ca ?p ?cb (COUNT(*) AS ?n) WHERE {
  ?a ?p ?b . ?a a ?ca . ?b a ?cb .
  FILTER(?ca != tsg:Layer && ?cb != tsg:Layer)
} GROUP BY ?ca ?p ?cb ORDER BY DESC(?n)""",
    "coverage_cypher": """
MATCH ()-[r]->() WITH label(r) AS p, count(*) AS n ORDER BY n DESC LIMIT 100000
WITH collect(n) AS ns, sum(n) AS total
RETURN size(ns) AS types, round(100.0 * list_sum(ns[1:4]) / total, 1) AS pct_top4,
       round(100.0 * list_sum(ns[1:8]) / total, 1) AS pct_top8, round(100.0 * list_sum(ns[1:16]) / total, 1) AS pct_top16""",
    # Kuzu has no window functions and its lambdas cannot see outer rows, so the marginals
    # f1 (first link) and f2 (second link) are computed with two UNWIND self-joins.
    "bigram_cypher": """
MATCH ()-[r1]->()-[r2]->()
WITH label(r1) AS l1, label(r2) AS l2, count(*) AS f
WITH collect({l1: l1, l2: l2, f: f}) AS rows, sum(f) AS N
UNWIND rows AS a
WITH rows, N, a.l1 AS l1, sum(a.f) AS f1
UNWIND rows AS b
WITH rows, N, l1, f1, b WHERE b.l1 = l1
WITH rows, N, l1, f1, b.l2 AS l2, b.f AS f
UNWIND rows AS c
WITH N, l1, f1, l2, f, c WHERE c.l2 = l2
WITH N, l1, f1, l2, f, sum(c.f) AS f2
RETURN l1, l2, f, round(log2((1.0 * f / N) / ((1.0 * f1 / N) * (1.0 * f2 / N))), 3) AS pmi_bits
ORDER BY pmi_bits DESC, f DESC LIMIT 15""",
    "bigram_sparql": """
SELECT ?p1 ?p2 (COUNT(*) AS ?f) WHERE { ?a ?p1 ?x . ?x ?p2 ?b .
  FILTER(STRSTARTS(STR(?p1), STR(tsg:)) && STRSTARTS(STR(?p2), STR(tsg:)))
  FILTER(?p1 != tsg:layer && ?p2 != tsg:layer) }
GROUP BY ?p1 ?p2 ORDER BY DESC(?f) LIMIT 15""",
    "literal_share_sparql": """
SELECT (SUM(IF(isLiteral(?o), 1, 0)) AS ?literals) (COUNT(*) AS ?triples) WHERE { ?s ?p ?o }""",
}
