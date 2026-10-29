"""YAML ontology -> OWL + SHACL (Turtle), plus the declared-ontology accessors.

Pipeline: troubleshooting.yaml -> load.py -> validate.py.
No OWL reasoner is used anywhere; the SHACL shapes are the closed-world contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent


def load_ontology(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else HERE / "ontology.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def _as_list(x: Any) -> list[str]:
    return x if isinstance(x, list) else [x]


def declared_patterns(onto: dict[str, Any]) -> set[tuple[str, str, str]]:
    """Every (from_label, REL_TYPE, to_label) the YAML allows. This is the DECLARED ontology."""
    out: set[tuple[str, str, str]] = set()
    for r in onto["relations"]:
        for a in _as_list(r["from"]):
            for b in _as_list(r["to"]):
                out.add((a, r["type"], b))
    return out


def class_index(onto: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in onto["classes"]}


def relation_index(onto: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["type"]: r for r in onto["relations"]}


# --------------------------------------------------------------------------
# OWL + SHACL generation
# --------------------------------------------------------------------------
_PREFIXES = """@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix tsg:  <%s> .
"""


def to_owl(onto: dict[str, Any]) -> str:
    """A lightweight OWL vocabulary: classes, object properties with domain/range, datatype properties.

    This is OWL as *documentation and interchange*, nothing here
    is ever fed to a reasoner. Note how a relation with several from/to labels becomes a
    unionOf domain/range: that is where RDF/OWL and the YAML contract start to diverge.
    """
    ns = onto["namespace"]
    lines = [_PREFIXES % ns, f"<{ns.rstrip('#')}> a owl:Ontology ; owl:versionInfo \"{onto['version']}\" .", ""]
    for layer in onto["layers"]:
        lines.append(f"tsg:layer_{layer} a tsg:Layer ; rdfs:label \"{layer}\" .")
    lines.append("tsg:Layer a owl:Class .")
    lines.append("")
    for c in onto["classes"]:
        lines.append(f"tsg:{c['name']} a owl:Class ; rdfs:label \"{c['name']}\" ; tsg:layer tsg:layer_{c['layer']} .")
    lines.append("")
    for r in onto["relations"]:
        frm, to = _as_list(r["from"]), _as_list(r["to"])
        dom = f"tsg:{frm[0]}" if len(frm) == 1 else "[ a owl:Class ; owl:unionOf ( " + " ".join(f"tsg:{x}" for x in frm) + " ) ]"
        rng = f"tsg:{to[0]}" if len(to) == 1 else "[ a owl:Class ; owl:unionOf ( " + " ".join(f"tsg:{x}" for x in to) + " ) ]"
        extra = ""
        if r.get("bridge"):
            why = r["why"].replace('"', "'")
            extra = f' ; tsg:bridge true ; rdfs:comment "{why}"'
        lines.append(f"tsg:{r['type']} a owl:ObjectProperty ; rdfs:domain {dom} ; rdfs:range {rng}{extra} .")
    lines.append("")
    seen: set[str] = set()
    for c in onto["classes"]:
        for p in c["required"]:
            if p not in seen and p != "id":
                seen.add(p)
                lines.append(f"tsg:{p} a owl:DatatypeProperty .")
    for p in onto["envelope"]:
        if p not in seen:
            seen.add(p)
            lines.append(f"tsg:{p} a owl:DatatypeProperty ; rdfs:comment \"provenance envelope\" .")
    return "\n".join(lines) + "\n"


def to_shacl(onto: dict[str, Any]) -> str:
    """One sh:NodeShape per class: required properties (minCount 1), enums (sh:in),
    relation targets (sh:class), and the envelope on every node.

    Closed world at write time: a Step without human_summary is a violation, not an unknown.
    """
    ns = onto["namespace"]
    enums = onto["enums"]
    lines = [_PREFIXES % ns]
    rels_by_from: dict[str, list[dict[str, Any]]] = {}
    for r in onto["relations"]:
        for a in _as_list(r["from"]):
            rels_by_from.setdefault(a, []).append(r)
    for c in onto["classes"]:
        name = c["name"]
        lines.append(f"tsg:{name}Shape a sh:NodeShape ; sh:targetClass tsg:{name} ;")
        props: list[str] = []
        for p in c["required"]:
            if p == "id":
                continue
            props.append(f"  sh:property [ sh:path tsg:{p} ; sh:minCount 1 ; sh:maxCount 1 ]")
        for p, enum_name in (c.get("enum") or {}).items():
            vals = " ".join(f'"{v}"' for v in enums[enum_name])
            props.append(f"  sh:property [ sh:path tsg:{p} ; sh:in ( {vals} ) ]")
        # envelope: every node carries status/authority/risk_tier from closed enums
        for p in ("status", "authority", "risk_tier"):
            vals = " ".join(f'"{v}"' for v in enums[p])
            props.append(f"  sh:property [ sh:path tsg:{p} ; sh:minCount 1 ; sh:in ( {vals} ) ]")
        props.append("  sh:property [ sh:path tsg:source_lane ; sh:minCount 1 ; sh:datatype xsd:string ]")
        props.append("  sh:property [ sh:path tsg:valid_from ; sh:minCount 1 ; sh:datatype xsd:dateTime ]")
        # relation targets
        for r in rels_by_from.get(name, []):
            to = _as_list(r["to"])
            if len(to) == 1:
                props.append(f"  sh:property [ sh:path tsg:{r['type']} ; sh:class tsg:{to[0]} ]")
            else:
                alts = " ".join(f"[ sh:class tsg:{t} ]" for t in to)
                props.append(f"  sh:property [ sh:path tsg:{r['type']} ; sh:or ( {alts} ) ]")
        lines.append(" ;\n".join(props) + " .")
        lines.append("")
    # one sanity rule: an approved node must carry attested_by
    lines.append("""tsg:ApprovedNeedsAttesterShape a sh:NodeShape ;
  sh:target [ a sh:SPARQLTarget ;
    sh:select \"\"\"SELECT ?this WHERE { ?this <%sstatus> "approved" }\"\"\" ] ;
  sh:property [ sh:path tsg:attested_by ; sh:minCount 1 ;
    sh:message "approved assertions must name who attested them" ] .
""" % ns)
    return "\n".join(lines)


def write_exports(out_dir: Path | str, onto: dict[str, Any] | None = None) -> tuple[Path, Path]:
    onto = onto or load_ontology()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    owl, shacl = out / "ontology.owl.ttl", out / "shapes.shacl.ttl"
    owl.write_text(to_owl(onto))
    shacl.write_text(to_shacl(onto))
    return owl, shacl
