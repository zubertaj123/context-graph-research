"""Smoke tests on LoCoMo conversation 0 (needs data/raw/locomo/locomo10.json)."""
from datetime import datetime

import numpy as np
import pytest

from paper2 import measures as M
from paper2.data import load_locomo
from paper2.evaluate import f1_score, exact_match, score_answer, admission_pr
from paper2.extract import OracleObservationExtractor, Triple, sentence_to_triples
from paper2.graph import MemoryGraph
from paper2.policies import AMACLite, GraphGate, OnlineQuantile, SIV2010, cross_validate, make_policy, rouge_l
from paper2.retrieve import BM25, Retriever, tokenize
from paper2.run import run_conversation
from paper2.policies import FixedThreshold


@pytest.fixture(scope="module")
def conv():
    return load_locomo()[0]


def test_loader(conv):
    assert conv.source == "locomo" and len(conv.sessions) >= 10 and len(conv.qa) > 50
    assert conv.observations and all(o.turn_ids for o in conv.observations)
    assert any(q.evidence_turns for q in conv.qa)


def test_sentence_rule():
    ts = sentence_to_triples("Caroline adopted a puppy named Max in March 2023.", "Caroline", ["Caroline", "Melanie"])
    assert ts and ts[0][0] == "Caroline" and ts[0][1] in ("adopt", "has_pet")


def test_graph_admit_resolve_consolidate():
    G = MemoryGraph(hubs=["Caroline"])
    at = datetime(2023, 5, 8)
    G.admit(Triple("Caroline", "lives_in", "Boston", observed_at=at, confidence=0.9))
    G.admit(Triple("Caroline", "lives_in", "boston", observed_at=at, confidence=0.9))     # resolves to the same entity
    assert G.stats()["entities"] == 2
    G.admit(Triple("Caroline", "lives_in", "Chicago", observed_at=datetime(2023, 6, 1), confidence=0.9))
    c = G.consolidate(datetime(2023, 6, 1))
    assert c["invalidated"] == 2 and G.stats()["live_facts"] == 1


def test_measures_ranges():
    G = MemoryGraph(hubs=["Caroline", "Melanie"])
    at = datetime(2023, 5, 8)
    for i in range(5):
        G.admit(Triple("Caroline", "likes", f"thing{i}", observed_at=at))
    G.admit(Triple("Caroline", "lives_in", "Boston", observed_at=at))
    view = M.GraphView(G)
    dup = [Triple("Caroline", "lives_in", "Boston", observed_at=at)]
    new = [Triple("Melanie", "visits", "Paris", observed_at=at)]
    fd, fn = M.features(G, dup, None, view), M.features(G, new, None, view)
    assert fd.t_edge == 1.0 and fn.t_edge == 0.0
    assert fd.anchoring == 1.0 and fn.anchoring == 0.0 and fn.n_nonhub == 1
    assert M.anchoring(G, [Triple("Caroline", "mentions", "Melanie", o_type="Person")])[0] != M.anchoring(G, [Triple("Caroline", "mentions", "Melanie", o_type="Person")])[0]  # NaN: no non-hub vertex
    assert 0 <= fd.impact <= 1 and 0 <= fn.impact_log <= 1
    assert fn.novelty > fd.novelty
    assert M.siva(G, dup, view) == 0.0 and 0 < M.siva(G, new, view) <= 1.0
    contra = [Triple("Caroline", "lives_in", "Chicago", observed_at=at)]
    assert M.siva(G, contra, view, "degree", True) > M.siva(G, contra, view, "plain", False)   # redesign credits the invalidation; 2010 does not


def test_bm25_and_retriever():
    idx = BM25([tokenize("Caroline adopted a puppy"), tokenize("Melanie went to Paris")])
    s = idx.scores(tokenize("who adopted a puppy"))
    assert s[0] > s[1]
    G = MemoryGraph(hubs=["Caroline"])
    at = datetime(2023, 5, 8)
    G.admit(Triple("Caroline", "adopt", "puppy Max", observed_at=at, text="Caroline adopted a puppy named Max", source_turns=["D1:1"]))
    G.admit(Triple("puppy Max", "is", "golden retriever", s_type="Thing", observed_at=at, text="Max is a golden retriever", source_turns=["D1:2"]))
    got = Retriever(G, k=1, cap=5).retrieve("What puppy did Caroline adopt?")
    assert len(got) == 2        # top-1 plus 1-hop expansion through 'puppy Max'
    # speaker named as an object resolves to the hub whatever its type (critique A6.2)
    assert G.resolve("caroline", "NamedEntity") in G.hubs and G.resolve("8 May 2023", "Date") is None
    shape = G.shape()
    assert shape["non_hub_vertices"] == 2 and shape["share_non_hub_deg_ge2"] == 0.5
    # superseded facts stay retrievable with an annotation
    G.admit(Triple("Caroline", "lives_in", "Boston", observed_at=at, text="Caroline lives in Boston"))
    G.admit(Triple("Caroline", "lives_in", "Chicago", observed_at=datetime(2023, 6, 1), text="Caroline lives in Chicago"))
    G.consolidate(datetime(2023, 6, 1))
    r = Retriever(G, k=5, cap=5)
    texts = [r.text(e) for e in r.retrieve("Where does Caroline live?")]
    assert any("superseded on 2023-06-01" in t for t in texts) and any("Chicago" in t and "superseded" not in t for t in texts)


def test_policies_and_gate():
    G = MemoryGraph(hubs=["Caroline"])
    at = datetime(2023, 5, 8)
    t = [Triple("Caroline", "lives_in", "Boston", observed_at=at, text="Caroline lives in Boston")]
    from paper2.data import Turn
    turn = Turn("D1:1", "session_1", "Caroline", "I live in Boston now", at)
    f = M.features(G, t, turn)
    from paper2.policies import FactTokens, amac_features
    extra = amac_features(G, turn, t, ["Hello", "I live in Boston now"], FactTokens())
    assert set(extra) == {"amac_conf", "amac_nov", "amac_prior"} and extra["amac_conf"] > 0.4
    for name in ("admit_all", "random", "heat", "siv_2010", "impact_2010", "graph_gate", "amac_lite", "amac_fit"):
        p = make_policy(name)
        p.reset(conv_stub())
        s = p.score(G, turn, t, f, extra)
        assert isinstance(s, float) or isinstance(s, (int, np.floating))
    g = OnlineQuantile(0.5, warmup=2)
    decisions = [g.admit(x) for x in [0.1, 0.9, 0.5, 0.95, 0.05, 0.6]]
    assert decisions[0] is True and decisions[1] is False      # warm-up alternates instead of admitting everything
    assert 0.9 < rouge_l("a b c", "a b c") <= 1.0 and rouge_l("x", "a b c") == 0.0


def conv_stub():
    from paper2.data import Conversation
    return Conversation(id="stub", speakers=["Caroline"], sessions=[], observations=[], qa=[], source="test")


def test_scoring():
    assert f1_score("Boston, MA", "boston") > 0.6 and exact_match("The Boston", "boston") == 1.0
    from paper2.data import QA
    q = QA("x", "q", "No information available", [], [], "adversarial", adversarial=True)
    assert score_answer("No information available in the conversation.", q)["f1"] == 1.0   # official substring rule
    assert score_answer("She lives in Boston.", q)["f1"] == 0.0
    pr = admission_pr({"a", "b"}, {"a", "b", "c"}, {"a", "c", "z"})
    assert pr["precision"] == 0.5 and abs(pr["recall_reachable"] - 0.5) < 1e-9


def test_fit_logistic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(M.FEATURE_NAMES)))
    y = (X[:, 1] + 0.3 * rng.normal(size=300) > 0).astype(float)
    groups = np.array([f"c{i % 6}" for i in range(300)])
    res = cross_validate(X, y, groups)
    assert res["cv_auc_mean"] > 0.85


def test_run_conversation_offline(conv):
    r = run_conversation(conv, make_policy("admit_all"), FixedThreshold(0.0), OracleObservationExtractor())
    s = r["summary"]
    closed = sum(x["closed"] for x in r["sessions"])
    assert s["admitted_facts"] > 100 and s["live_facts"] <= s["admitted_facts"] + closed   # closure edges are live but not admitted
    assert "shape_share_non_hub_deg_ge2" in s and all("amac_nov" in t and "n_live" in t for t in r["turns"])
    hits = [q["hit"] for q in r["qa"] if q["hit"] == q["hit"]]
    assert hits and np.mean(hits) > 0.3
    assert all(0.0 <= t["siva_2010"] <= 1.0 and 0.0 <= t["siva_cons"] <= 1.0 for t in r["turns"])
    assert len(r["sessions"]) == len(conv.sessions)


def test_gate_holds_budget_for_trending_scores():
    """A score that grows monotonically with graph size defeats a plain quantile rule; the band must hold the rate."""
    for seq in (list(range(300)), list(range(300, 0, -1)), [0.0] * 300):
        g = OnlineQuantile(0.5, warmup=20)
        admitted = sum(g.admit(float(x)) for x in seq)
        assert abs(admitted / 300 - 0.5) < 0.1, (seq[:3], admitted)


def test_matched_size_and_paired_poison(conv):
    from paper2.run import Poisoner, run_matched
    ex = OracleObservationExtractor()
    base = run_conversation(conv, make_policy("admit_all"), FixedThreshold(0.0), ex)["summary"]["live_facts"]
    r, q = run_matched(conv, lambda: make_policy("random"), ex, int(0.5 * base))
    assert abs(r["summary"]["live_facts"] - 0.5 * base) / (0.5 * base) <= 0.06 and 0.02 <= q <= 0.98
    vocab = sorted({t.o for _, ts in ex.candidates(conv) for t in ts})
    inj = []
    for pol in ("random", "admit_all"):
        po = Poisoner(0.2, vocab, seed=1)
        out = run_conversation(conv, make_policy(pol), FixedThreshold(0.0), ex, poisoner=po)
        inj.append([t["poison"] for t in out["turns"]])
    assert inj[0] == inj[1]          # the injected set is paired across policies


def test_symmetric_invalidation_has_no_ping_pong():
    """A superseded symmetric functional fact must not be re-created from its mirror at the next session."""
    G = MemoryGraph(hubs=["Melanie"])
    t1, t2 = datetime(2023, 5, 1), datetime(2023, 6, 1)
    G.admit(Triple("Melanie", "married_to", "Bob", o_type="Person", observed_at=t1))
    G.consolidate(t1)                       # adds the mirror (Bob married_to Melanie)
    G.admit(Triple("Melanie", "married_to", "Carl", o_type="Person", observed_at=t2))
    c1 = G.consolidate(t2)
    c2 = G.consolidate(datetime(2023, 7, 1))
    live = [(G.g.nodes[e.src].props["name"], e.type, G.g.nodes[e.dst].props["name"]) for e in G.g.edges if e.props.get("valid_to") is None]
    assert ("Melanie", "married_to", "Bob") not in live and ("Bob", "married_to", "Melanie") not in live
    assert ("Melanie", "married_to", "Carl") in live and ("Carl", "married_to", "Melanie") in live
    assert c2["invalidated"] == 0 and c2["closed"] == 0 and c1["invalidated"] == 1
