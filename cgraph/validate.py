"""SHACL validation of the exported RDF against the shapes generated from the YAML."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import rdflib
from pyshacl import validate as _validate

from . import rdf
from .model import ContextGraph, envelope
from .ontology import load_ontology, to_shacl


def validate_graph(g: ContextGraph, shapes_ttl: str | None = None, max_report: int = 10) -> dict[str, Any]:
    """Validate the plain-triple projection (SHACL does not see RDF-star annotations)."""
    shapes_ttl = shapes_ttl or to_shacl(load_ontology())
    data = rdflib.Graph().parse(data=rdf.to_turtle(g, "plain"), format="turtle")
    shapes = rdflib.Graph().parse(data=shapes_ttl, format="turtle")
    conforms, report_graph, report_text = _validate(data, shacl_graph=shapes, advanced=True, inference="none", abort_on_first=False)
    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
    results = []
    for r in report_graph.subjects(rdflib.RDF.type, SH.ValidationResult):
        results.append({
            "focus": str(report_graph.value(r, SH.focusNode)).replace(rdf.EX, "ex:"),
            "path": str(report_graph.value(r, SH.resultPath) or "").replace(rdf.TSG, "tsg:"),
            "message": str(report_graph.value(r, SH.resultMessage) or ""),
        })
    return {"conforms": conforms, "violations": len(results), "sample": results[:max_report]}


def inject_violations(g: ContextGraph) -> list[str]:
    """Three planted errors: a Step without human_summary, an approved node with no attester,
    and a Variable with an out-of-enum resolution."""
    s = g.add_node("Step", "step:planted:no_summary", "process", title="planted", rationale="r", customer_effort="low", order=99,
                   **envelope("agent", "planted", content="x"))
    s.props.pop("human_summary", None)
    v = g.add_node("Variable", "var:planted", "process", name="planted", resolution="guess",
                   **envelope("boomi", "planted", content="v"))
    a = g.add_node("Action", "A_PLANTED", "process", name="planted", kind="remedial",
                   **envelope("boomi", "planted", content="a"))
    a.props["attested_by"] = None
    return [s.id, v.id, a.id]
