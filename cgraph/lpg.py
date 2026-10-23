"""LPG side: load the context graph into an embedded Kuzu database (Cypher).

Kuzu is schema-first like Neo4j-with-constraints: one node table per label, one
relationship table per type with the allowed FROM/TO pairs. That is precisely
a 'YAML contract' idea, and it is why undeclared patterns fail to load.

NOTE: `Table`, `Column`, `Case` (and `order`, `type`, `version`) are reserved
words in Kuzu's Cypher. `cypher()` below auto-escapes those three labels, so you
can write MATCH (t:Table) naturally; property names still need backticks:
RETURN s.`order`.
"""
from __future__ import annotations

import csv
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import kuzu

from .model import ContextGraph
from .ontology import load_ontology, _as_list

RESERVED_LABELS = {"Table", "Column", "Case"}


def _kuzu_type(v: Any) -> str:
    if isinstance(v, bool):
        return "BOOLEAN"
    if isinstance(v, int):
        return "INT64"
    if isinstance(v, float):
        return "DOUBLE"
    if isinstance(v, datetime):
        return "TIMESTAMP"
    return "STRING"


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, datetime):
        return v.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    return str(v)


def open_db(path: Path | str, fresh: bool = True) -> kuzu.Connection:
    p = Path(path)
    if fresh:
        for side in (p, Path(str(p) + ".wal"), Path(str(p) + ".lock")):
            if side.exists():
                shutil.rmtree(side) if side.is_dir() else side.unlink()
    db = kuzu.Database(str(p))
    return kuzu.Connection(db)


def load(g: ContextGraph, path: Path | str) -> kuzu.Connection:
    """Create the schema from the data + YAML and bulk-load every node and edge via CSV."""
    onto = load_ontology()
    conn = open_db(path)
    tmp = Path(tempfile.mkdtemp(prefix="cgraph_kuzu_"))
    # --- node tables: union of property names/types seen per label
    cols: dict[str, dict[str, str]] = {}
    for n in g.nodes.values():
        c = cols.setdefault(n.label, {"layer": "STRING"})
        for k, v in n.props.items():
            if v is not None and k not in c:
                c[k] = _kuzu_type(v)
    for label, c in cols.items():
        if "valid_from" in c:
            c.setdefault("valid_to", "TIMESTAMP")   # as-of queries must be able to bind valid_to even when all NULL
        spec = ", ".join(["id STRING PRIMARY KEY"] + [f"`{k}` {t}" for k, t in c.items()])
        conn.execute(f"CREATE NODE TABLE `{label}`({spec})")
        f = tmp / f"{label}.csv"
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id"] + list(c))
            for n in g.by_label(label):
                w.writerow([n.id] + [_cell(n.layer if k == "layer" else n.props.get(k)) for k in c])
        conn.execute(f"COPY `{label}` FROM '{f}' (HEADER=true, PARALLEL=false)")
    # --- rel tables: FROM/TO pairs that occur in the data; undeclared pairs are flagged
    pairs: dict[str, set[tuple[str, str]]] = {}
    rprops: dict[str, dict[str, str]] = {}
    for e in g.edges:
        pairs.setdefault(e.type, set()).add((g.nodes[e.src].label, g.nodes[e.dst].label))
        rp = rprops.setdefault(e.type, {})
        for k, v in e.props.items():
            if v is not None and k not in rp:
                rp[k] = _kuzu_type(v)
    declared = {r["type"]: {(a, b) for a in _as_list(r["from"]) for b in _as_list(r["to"])} for r in onto["relations"]}
    undeclared: list[tuple[str, str, str]] = []
    for t, ps in pairs.items():
        if "valid_from" in rprops[t]:
            rprops[t].setdefault("valid_to", "TIMESTAMP")
        for a, b in ps:
            if (a, b) not in declared.get(t, set()):
                undeclared.append((a, t, b))
        ft = ", ".join(f"FROM `{a}` TO `{b}`" for a, b in sorted(ps))
        pp = "".join(f", `{k}` {ty}" for k, ty in rprops[t].items())
        conn.execute(f"CREATE REL TABLE `{t}`({ft}{pp})")
        for a, b in sorted(ps):
            f = tmp / f"{t}__{a}__{b}.csv"
            with open(f, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["from", "to"] + list(rprops[t]))
                for e in g.edges:
                    if e.type == t and g.nodes[e.src].label == a and g.nodes[e.dst].label == b:
                        w.writerow([e.src, e.dst] + [_cell(e.props.get(k)) for k in rprops[t]])
            conn.execute(f"COPY `{t}` FROM '{f}' (HEADER=true, PARALLEL=false, from='{a}', to='{b}')")
    shutil.rmtree(tmp, ignore_errors=True)
    conn.undeclared_patterns = undeclared  # type: ignore[attr-defined]
    return conn


_LABEL_RE = re.compile(r"(?<=[:|])(" + "|".join(sorted(RESERVED_LABELS)) + r")\b(?!`)")


def escape(query: str) -> str:
    """Backtick the reserved labels so MATCH (t:Table) works."""
    return _LABEL_RE.sub(r"`\1`", query)


def cypher(conn: kuzu.Connection, query: str, **params: Any) -> list[dict[str, Any]]:
    res = conn.execute(escape(query), parameters=params or None)
    if isinstance(res, list):
        res = res[-1]
    names = res.get_column_names()
    rows = []
    while res.has_next():
        rows.append(dict(zip(names, res.get_next())))
    return rows
