from datetime import datetime, timezone

import pytest

from cgraph import data, lpg, measures, memory, rdf, retrieval, temporal, validate


@pytest.fixture(scope="module")
def graph():
    return data.build(scale=1)


def test_census(graph):
    g, key = graph
    c = g.census()
    assert c["nodes"] > 1500 and c["edges"] > 5000
    assert set(c["by_layer"]) == {"process", "catalog", "evidence", "memory", "taxonomy", "lifecycle"}
    assert len(key.step_evidence) == 60


def test_declared_vs_extant_clean(graph):
    g, _ = graph
    d = measures.declared_vs_extant(g)
    assert d["extant_undeclared"] == []


def test_rdf_styles_and_sparql_star(graph):
    g, _ = graph
    counts = {s: len(rdf.load_store(rdf.to_turtle(g, s))) for s in ("plain", "reification", "rdfstar", "reifier")}
    assert counts["plain"] < counts["reifier"] < counts["rdfstar"] < counts["reification"]
    store = rdf.load_store(rdf.to_turtle(g, "reifier"))
    rows = rdf.sparql(store, "SELECT ?r WHERE { ex:var:bios_version tsg:RESOLVED_BY ?c ~ ?e . ?e tsg:rank ?r } ORDER BY ?r")
    assert [r["r"] for r in rows] == [0, 1]


def test_kuzu_roundtrip(graph, tmp_path):
    g, _ = graph
    conn = lpg.load(g, tmp_path / "db")
    assert conn.undeclared_patterns == []
    n = lpg.cypher(conn, "MATCH (s:Step) RETURN count(s) AS n")[0]["n"]
    assert n == 60
    assert lpg.cypher(conn, "MATCH (t:Table) RETURN count(t) AS n")[0]["n"] == 4


def test_as_of(graph):
    g, _ = graph
    fid = "flow:latitude:black_screen_after_sleep"
    before = temporal.canonical_path(g, fid, datetime(2026, 3, 1, tzinfo=timezone.utc))
    after = temporal.canonical_path(g, fid, datetime(2026, 9, 1, tzinfo=timezone.utc))
    assert len(after) == len(before) + 1
    assert "step:latitude:black_screen_after_sleep:2" in after and "step:latitude:black_screen_after_sleep:2" not in before


def test_retrieval_scoping_removes_failure_modes(graph):
    g, key = graph
    idx = retrieval.build_index(g)
    ev = retrieval.evaluate(g, key, idx, k=5)
    assert ev["scoped"]["wrong_model"] == 0 and ev["scoped"]["wrong_locale"] == 0 and ev["scoped"]["stale_revision"] == 0
    assert ev["unscoped"]["wrong_model"] > 0 or ev["unscoped"]["wrong_locale"] > 0
    assert ev["scoped"]["hit@k"] >= ev["unscoped"]["hit@k"]


def test_shacl_catches_planted_violations(graph):
    g, _ = data.build(scale=1)
    assert validate.validate_graph(g)["conforms"]
    validate.inject_violations(g)
    r = validate.validate_graph(g)
    assert not r["conforms"] and r["violations"] == 3


def test_graduation_rules(tmp_path):
    g, key = data.build(scale=1)
    ledger = memory.Ledger(tmp_path / "audit.jsonl")
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    p = memory.Proposal(kind="branch", proposer="agent:copilot", proposes_step="step:xps:wifi_drops:2", derived_from=["memory:xps:wifi_drops"],
                        payload={"insert_before": "step:xps:wifi_drops:2", "new_step": {"id": "step:xps:wifi_drops:1b", "title": "t", "action": "A_RESET_BIOS",
                                 "human_summary": "h", "rationale": "r", "customer_effort": "low"}})
    memory.propose(g, ledger, p, now)
    memory.validate(g, ledger, p, replay=lambda _: {"lift": 0.2, "regression": 0.0})
    assert memory.tier(g, p) == "T2"
    with pytest.raises(memory.GraduationError):
        memory.review(g, ledger, p, approver="agent:copilot", decision="approved", at=now)
    assert memory.review(g, ledger, p, approver="steward:bob", decision="approved", at=now) == "approved"
    res = memory.promote(g, ledger, p, now)
    assert res["new_step"] == "step:xps:wifi_drops:1b" and len(res["superseded"]) == 1
    assert measures.declared_vs_extant(g)["extant_undeclared"] == []
    path_now = temporal.canonical_path(g, "flow:xps:wifi_drops", now)
    assert "step:xps:wifi_drops:1b" in path_now
