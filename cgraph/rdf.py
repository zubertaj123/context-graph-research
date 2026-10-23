"""RDF side: export the context graph to Turtle in four annotation styles, load it
into an embedded Oxigraph store, and run SPARQL (including SPARQL-star).

The four styles answer al-Saffar's question "how much is bloat?" with numbers:
  plain        edge properties are DROPPED (lossy; count what is lost)
  reification  classic rdf:Statement, 4 extra triples per annotated edge
  rdfstar      << s p o >> annotation (RDF-star community group syntax)
  reifier      RDF 1.2: s p o ~ :e . :e prop v .  (rdf:reifies a triple term)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import pyoxigraph as ox

from .model import ContextGraph, Edge, Node
from .ontology import load_ontology

ONTO = load_ontology()
TSG = ONTO["namespace"]
EX = ONTO["instance_namespace"]
PREFIXES = f"""@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix tsg:  <{TSG}> .
@prefix ex:   <{EX}> .
"""
SPARQL_PREFIXES = f"""PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX tsg:  <{TSG}>
PREFIX ex:   <{EX}>
"""


def iri(node_id: str) -> str:
    """Instance IRIs: ex:<id> with the characters Turtle dislikes percent-encoded."""
    safe = node_id.replace("%", "%25").replace(" ", "%20").replace("#", "%23").replace("<", "%3C").replace(">", "%3E").replace('"', "%22").replace("{", "%7B").replace("}", "%7D").replace("|", "%7C").replace("\\", "%5C").replace("^", "%5E").replace("`", "%60")
    return f"<{EX}{safe}>"


def literal(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f'"{v}"^^xsd:integer'
    if isinstance(v, float):
        return f'"{v}"^^xsd:double'
    if isinstance(v, datetime):
        return f'"{v.isoformat()}"^^xsd:dateTime'
    if isinstance(v, (list, tuple)):
        return None  # lists are not exported (Turtle collections are rarely worth it here)
    s = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{s}"'


def _node_triples(n: Node) -> Iterable[str]:
    s = iri(n.id)
    yield f"{s} a tsg:{n.label} ; tsg:layer tsg:layer_{n.layer} ."
    for k, v in n.props.items():
        lit = literal(v)
        if lit is not None:
            yield f"{s} tsg:{k} {lit} ."


def to_turtle(g: ContextGraph, style: str = "reifier", live_at: datetime | None = None) -> str:
    """Serialise the graph. `style` in {plain, reification, rdfstar, reifier}."""
    assert style in ("plain", "reification", "rdfstar", "reifier"), style
    out = [PREFIXES]
    for n in g.nodes.values():
        out.extend(_node_triples(n))
    for i, e in enumerate(g.edges):
        if live_at is not None:
            from .model import edge_live_at
            if not edge_live_at(e, live_at):
                continue
        s, p, o = iri(e.src), f"tsg:{e.type}", iri(e.dst)
        props = [(k, literal(v)) for k, v in e.props.items() if literal(v) is not None]
        if style == "plain" or not props:
            out.append(f"{s} {p} {o} .")
            continue
        if style == "reification":
            st = f"<{EX}stmt/{i}>"
            out.append(f"{s} {p} {o} .")
            out.append(f"{st} a rdf:Statement ; rdf:subject {s} ; rdf:predicate {p} ; rdf:object {o} .")
            for k, lit in props:
                out.append(f"{st} tsg:{k} {lit} .")
        elif style == "rdfstar":
            out.append(f"{s} {p} {o} .")
            for k, lit in props:
                out.append(f"<< {s} {p} {o} >> tsg:{k} {lit} .")
        else:  # RDF 1.2 reifier
            r = f"<{EX}edge/{i}>"
            out.append(f"{s} {p} {o} ~ {r} .")
            for k, lit in props:
                out.append(f"{r} tsg:{k} {lit} .")
    return "\n".join(out) + "\n"


def load_store(turtle: str, store: ox.Store | None = None) -> ox.Store:
    store = store or ox.Store()
    store.load(turtle.encode(), format=ox.RdfFormat.TURTLE)
    return store


def sparql(store: ox.Store, query: str, prefixes: bool = True) -> list[dict[str, Any]]:
    """Run a SELECT and return plain Python rows (IRIs shortened to ex:/tsg: prefixes)."""
    q = (SPARQL_PREFIXES + query) if prefixes else query
    res = store.query(q)
    rows: list[dict[str, Any]] = []
    if isinstance(res, bool):
        return [{"ask": res}]
    for sol in res:
        row: dict[str, Any] = {}
        for var in res.variables:
            v = sol[var]
            row[var.value] = _py(v)
        rows.append(row)
    return rows


def _py(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, ox.NamedNode):
        s = v.value
        if s.startswith(EX):
            return "ex:" + s[len(EX):]
        if s.startswith(TSG):
            return "tsg:" + s[len(TSG):]
        return s
    if isinstance(v, ox.Literal):
        dt = v.datatype.value if v.datatype else ""
        if dt.endswith("#integer"):
            return int(v.value)
        if dt.endswith("#double") or dt.endswith("#decimal"):
            return float(v.value)
        if dt.endswith("#boolean"):
            return v.value == "true"
        return v.value
    if isinstance(v, ox.Triple):
        return (_py(v.subject), _py(v.predicate), _py(v.object))
    return str(v)


def triple_count(store: ox.Store) -> int:
    return len(store)


def literal_share(store: ox.Store) -> float:
    """al-Saffar's hygiene measure: fraction of triples whose object is a literal (30-63% in his corpora)."""
    n = lit = 0
    for q in store:
        n += 1
        if isinstance(q.object, ox.Literal):
            lit += 1
    return lit / n if n else 0.0
