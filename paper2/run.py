"""The experiment loop and CLI.

  python -m paper2.run run --dataset locomo --policy admit_all --offline
  python -m paper2.run fit --features paper2/out/locomo_admit_all/turns.jsonl --exclude-conv conv-26 --structural --out w.json
  python -m paper2.run run --dataset locomo --policy graph_gate --weights w.json --match-live 0.5 --conv conv-26 --offline

Per conversation: stream turns in order -> extract candidates -> structural and A-MAC features -> policy
score -> gate -> admit (exact duplicates suppressed for every policy) -> consolidate at the end of each
session (recording the realised session-level SIVa) -> answer every question from BM25 + 1-hop retrieval
over the admitted graph. Offline mode scores retrieval by evidence recall; online mode also asks a fixed
answerer and scores with the official LoCoMo rules. `--match-live f` bisects the budget q per conversation
until the memory holds f x (admit-all live facts) within 5%, which is the pre-registered size matching.
Every summary.json carries the code, data and weights hashes it was produced with.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper2 import evaluate as ev  # noqa: E402
from paper2 import measures as M  # noqa: E402
from paper2.data import Conversation, load_locomo, load_longmemeval  # noqa: E402
from paper2.extract import HeuristicExtractor, LLMExtractor, OracleObservationExtractor, Triple  # noqa: E402
from paper2.graph import MemoryGraph  # noqa: E402
from paper2.llm import TEMPORAL_HINT, answer_question, make_client  # noqa: E402
from paper2.policies import (AMAC_FEATURE_NAMES, DESCRIPTIONS, FactTokens, FixedThreshold, OnlineQuantile, Policy, amac_features,  # noqa: E402
                             cross_validate, make_policy)
from paper2.retrieve import Retriever  # noqa: E402

OUT = ROOT / "paper2" / "out"
TIER_CONFIDENCE = {"verbatim-user": 1.0, "extractor-inferred": 0.8, "llm-summary": 0.6, "poisoned-lower": 0.4}


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "paper2").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def cached_candidates(extractor, conv: Conversation) -> dict:
    """Candidate triples per turn, cached on disk per (extractor, extract.py hash, conversation). The cache
    holds objects this code produced from the dataset, so it is ours, not untrusted input."""
    import pickle
    from paper2 import extract as _ex
    key = hashlib.sha256((Path(_ex.__file__).read_bytes() + f"{_ex.backend()}:{type(extractor).__name__}:{getattr(extractor, 'tier', '')}:{getattr(extractor, 'confidence', '')}:{conv.id}:{conv.source}".encode())).hexdigest()[:20]
    d = OUT / "cache" / "candidates"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{key}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    cands = {turn.id: triples for turn, triples in extractor.candidates(conv)}
    with open(f, "wb") as fh:
        pickle.dump(cands, fh)
    return cands


# ---------------------------------------------------------------------------
# poisoning arm (H4)
# ---------------------------------------------------------------------------
class Poisoner:
    """Inject fabricated facts at rate r per candidate turn. Design (critique A8): predicates are drawn from the
    predicate distribution of the real candidates seen so far (so a poisoned predicate is not a novelty
    give-away), subjects are the speakers or hub-adjacent known entities, objects come from the corpus
    vocabulary, each fact is verbalised with its predicate and repeated once a few turns later
    (self-consistency). mode='sourceless': confidence equals the real candidates' tier, so no policy can
    see the poison; mode='lower': tier poisoned-lower (0.4), the provenance-weighting condition. The RNG is
    seeded per conversation and consumed identically whatever the policy decides, so the injected set is
    paired across policies."""

    def __init__(self, rate: float, vocab: list[str], seed: int = 0, mode: str = "sourceless", repeat_after: int = 3):
        self.rate, self.vocab, self.seed, self.mode, self.repeat_after = rate, vocab, seed, mode, repeat_after
        self.rng = random.Random(seed)
        self.preds: Counter = Counter()
        self.entities: list[str] = []
        self.confs: list[float] = []
        self.pending: list[tuple[int, Triple]] = []
        self.i = 0

    def reset(self, conv: Conversation):
        self.rng = random.Random(f"{self.seed}:{conv.id}")
        self.preds, self.entities, self.confs, self.pending, self.i = Counter(), [], [], [], 0

    def maybe(self, conv: Conversation, turn, real: list[Triple]) -> list[Triple]:
        for t in real:
            self.preds[t.p] += 1
            self.confs.append(t.confidence)
            if t.o_type != "Date" and t.o not in conv.speakers:
                self.entities.append(t.o)
        self.i += 1
        out = [t for due, t in self.pending if due <= self.i]
        self.pending = [(d, t) for d, t in self.pending if d > self.i]
        if self.rate <= 0 or not self.vocab or self.rng.random() >= self.rate:
            return out
        subj = self.rng.choice(conv.speakers) if (self.rng.random() < 0.5 or not self.entities) else self.rng.choice(self.entities)
        preds, weights = (zip(*self.preds.items()) if self.preds else (["likes"], [1]))
        pred = self.rng.choices(list(preds), weights=list(weights))[0]
        obj = self.rng.choice(self.vocab)
        if self.mode == "sourceless":
            tier, conf = "llm-summary", (sum(self.confs) / len(self.confs) if self.confs else TIER_CONFIDENCE["llm-summary"])
        else:
            tier, conf = "poisoned-lower", TIER_CONFIDENCE["poisoned-lower"]
        t = Triple(s=subj, p=pred, o=obj, s_type="Person" if subj in conv.speakers else "Thing", o_type="Thing", source_turns=[turn.id],
                   session=turn.session, speaker=subj, observed_at=turn.at, tier=tier, confidence=conf, method="poison",
                   text=f"{subj} {pred.replace('_', ' ')} {obj}", truth=False)
        self.pending.append((self.i + self.repeat_after, Triple(**{**t.__dict__, "source_turns": [turn.id]})))
        return out + [t]


# ---------------------------------------------------------------------------
# one conversation
# ---------------------------------------------------------------------------
def run_conversation(conv: Conversation, policy: Policy, gate, extractor, *, offline: bool = True, answer_client=None, k: int = 20,
                     cap: int = 40, poisoner: Poisoner | None = None, tier_weight: bool = False, include_superseded: bool = True,
                     cands: dict | None = None) -> dict:
    t0 = time.time()
    G = MemoryGraph(hubs=conv.speakers)
    policy.reset(conv)
    gate.reset()
    if poisoner:
        poisoner.reset(conv)
    evidence = {tid for q in conv.qa for tid in q.evidence_turns}
    turns_out, sessions_out = [], []
    admitted_turns, candidate_turns = set(), set()
    n_cand = n_admitted = n_dup = n_poison_cand = n_poison_admitted = 0
    cands = cands if cands is not None else cached_candidates(extractor, conv)
    fact_tokens = FactTokens()
    prev_texts: list[str] = []
    position = 0
    for sess in conv.sessions:
        snap_cons, snap_2010 = M.snapshot(G, "degree"), M.snapshot(G, "plain")
        for turn in sess.turns:
            triples = list(cands.get(turn.id, []))
            if poisoner:
                p = poisoner.maybe(conv, turn, triples)
                n_poison_cand += len(p)
                triples += p
            if not triples:
                prev_texts.append(turn.text)
                continue
            candidate_turns.add(turn.id)
            view = M.GraphView(G)
            feats = M.features(G, triples, turn, view)
            extra = amac_features(G, turn, triples, prev_texts, fact_tokens)
            s2010 = M.siva(G, triples, view, "plain", False)          # the 2010 definition: unweighted union ratio
            scons = M.siva(G, triples, view, "degree", True)          # redesign: degree-weighted, consolidation-aware
            score = policy.score(G, turn, triples, feats, extra)
            if tier_weight:
                score *= feats.confidence
            admit = gate.admit(score)
            n_cand += len(triples)
            if admit:
                admitted_turns.add(turn.id)
                for t in triples:
                    s, o = G.resolve(t.s, t.s_type), G.resolve(t.o, t.o_type)
                    if s and o and G.has_fact(s, t.p, o):
                        n_dup += 1
                        continue
                    G.admit(t)
                    n_admitted += 1
                    n_poison_admitted += int(t.truth is False)
            label = turn.id in evidence or any(tid in evidence for t in triples for tid in t.source_turns)
            turns_out.append({"conv": conv.id, "turn": turn.id, "session": sess.id, "position": position, "n_live": view.n_live,
                              "n_triples": len(triples), "score": float(score), "admitted": bool(admit), "label": bool(label),
                              "siva_2010": s2010, "siva_cons": scons, "siva_scaled": scons * max(view.n_live + len(triples), 1),
                              "poison": sum(1 for t in triples if t.truth is False), **feats.to_dict(), **extra})
            position += 1
            prev_texts.append(turn.text)
        cons = G.consolidate(sess.at)
        rc, r10 = M.realised_change(snap_cons, G), M.realised_change(snap_2010, G)
        sessions_out.append({"conv": conv.id, "session": sess.id, **cons, "siva_realised_2010": r10["siva"], "siva_realised_cons": rc["siva"],
                             "siva_realised_scaled": rc["siva_scaled"], "new_edges": rc["new_edges"], "live_edges": rc["live_edges"]})
    retriever = Retriever(G, k=k, cap=cap, include_superseded=include_superseded)
    qa_out = []
    for q in conv.qa:
        got = retriever.retrieve(q.question, at=q.at)
        rec = {"conv": conv.id, "qa": q.id, "category": q.category, "category_id": getattr(q, "category_id", None), "adversarial": q.adversarial,
               **ev.evidence_recall(got, q), "poison_retrieved": sum(1 for e in got if e.props.get("truth") is False)}
        if not offline and answer_client is not None:
            pred = answer_question(answer_client, q.question, [retriever.text(e) for e in got],
                                   cache_key=f"answer:{conv.id}:{q.id}:{policy.name}:{gate.describe()}",
                                   hint=TEMPORAL_HINT if q.category == "temporal" else None)
            rec.update({"pred": pred, "gold": q.answer, **ev.score_answer(pred, q)})
        qa_out.append(rec)
    pr = ev.admission_pr(admitted_turns, candidate_turns, evidence)
    summary = {"conv": conv.id, "policy": policy.name, "gate": gate.describe(), "candidates": n_cand, "candidate_turns": len(candidate_turns),
               "admitted_facts": n_admitted, "duplicates_suppressed": n_dup, "poison_candidates": n_poison_cand, "poison_admitted": n_poison_admitted,
               **G.stats(), **{"shape_" + k: v for k, v in G.shape().items()}, **pr, "seconds": round(time.time() - t0, 2), **policy.cost()}
    return {"summary": summary, "turns": turns_out, "sessions": sessions_out, "qa": qa_out}


def run_matched(conv: Conversation, policy_factory, extractor, target_live: int, tol: float = 0.05, iters: int = 8, **kw) -> tuple[dict, float]:
    """Bisection on the budget q until the memory holds target_live facts within tol (critique B(iii), C3)."""
    lo, hi = 0.02, 0.98
    best, best_q, best_gap = None, None, float("inf")
    cands = cached_candidates(extractor, conv)
    # Bisect offline (no answer calls): every iteration builds a different memory, so answering inside the loop
    # would pay for up to `iters` answer passes per conversation (19 Oct proxy run). The matched memory is
    # re-run once below with answering on; the policies are deterministic given their seed.
    bis_kw = {**kw, "answer_client": None} if kw.get("answer_client") is not None else kw
    for _ in range(iters):
        q = (lo + hi) / 2
        r = run_conversation(conv, policy_factory(), OnlineQuantile(q), extractor, cands=cands, **bis_kw)
        live = r["summary"]["live_facts"]
        gap = abs(live - target_live) / max(target_live, 1)
        if gap < best_gap:
            best, best_q, best_gap = r, q, gap
        if gap <= tol:
            break
        if live < target_live:
            lo = q
        else:
            hi = q
    if bis_kw is not kw:                                        # online arm: answer once, on the matched memory
        r = run_conversation(conv, policy_factory(), OnlineQuantile(best_q), extractor, cands=cands, **kw)
        assert r["summary"]["live_facts"] == best["summary"]["live_facts"], "matched memory not reproduced"
        best = r
    best["summary"]["q_matched"], best["summary"]["target_live"], best["summary"]["size_gap"] = best_q, target_live, best_gap
    return best, best_q


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load(dataset: str, limit: int | None, ids: list[str] | None, variant: str = "s") -> list[Conversation]:
    if dataset == "locomo":
        convs = load_locomo()
    else:
        try:
            convs = load_longmemeval(limit=limit, variant=variant)
        except TypeError:                                   # loader without the variant argument
            convs = load_longmemeval(limit=limit)
    if ids:
        convs = [c for c in convs if c.id in set(ids)]
    if limit and dataset == "locomo":
        convs = convs[:limit]
    return convs


def make_extractor(name: str, dataset: str, client=None):
    if name == "oracle":
        if dataset != "locomo":
            raise SystemExit("the oracle observation extractor exists only for LoCoMo")
        return OracleObservationExtractor()
    if name == "heuristic":
        return HeuristicExtractor()
    if name == "llm":
        if client is None:
            raise SystemExit("--extractor llm needs --extract-model")
        return LLMExtractor(client)
    raise SystemExit(f"unknown extractor {name}")


def cmd_run(a: argparse.Namespace) -> None:
    convs = load(a.dataset, a.limit, a.conv, a.lme_variant)
    gate_client = make_client(a.gate_model)
    policy_factory = lambda: make_policy(a.policy, seed=a.seed, weights=a.weights, client=gate_client)  # noqa: E731
    probe = policy_factory()
    overlap = set(getattr(probe, "fit_convs", [])) & {c.id for c in convs}
    if overlap:
        raise SystemExit(f"weights {a.weights} were fitted on {sorted(overlap)}, which are in this run: refusing (critique A5.4)")
    extractor = make_extractor(a.extractor, a.dataset, make_client(a.extract_model))
    answer_client = None if a.offline else make_client(a.answer_model)
    if not a.offline and answer_client is None:
        raise SystemExit("online mode needs --answer-model (or pass --offline)")
    poisoner = None
    if a.poison_rate > 0:
        vocab = sorted({t.o for c in convs for ts in cached_candidates(extractor, c).values() for t in ts if t.o_type != "Date"})
        poisoner = Poisoner(a.poison_rate, vocab, a.seed, mode=a.poison_mode)
    kw = dict(offline=a.offline, answer_client=answer_client, k=a.k, cap=a.cap, poisoner=poisoner, tier_weight=a.tier_weight,
              include_superseded=not a.live_only)
    tag = f"m{a.match_live}" if a.match_live is not None else (f"q{a.budget}" if a.budget is not None else f"tau{a.threshold}")
    name = a.name or f"{a.dataset}_{a.policy}_{tag}"
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    files = {k: (out / f"{k}.jsonl").open("w") for k in ("turns", "sessions", "qa")}
    summaries, all_qa = [], []
    for conv in convs:
        if a.match_live is not None:
            base = run_conversation(conv, make_policy("admit_all"), FixedThreshold(0.0), extractor, **{**kw, "poisoner": None})
            target = int(round(a.match_live * base["summary"]["live_facts"]))
            r, q = run_matched(conv, policy_factory, extractor, target, **kw)
            gate_desc = f"matched live={a.match_live} (q={q:.3f})"
        else:
            gate = OnlineQuantile(a.budget, a.warmup, window=a.window) if a.budget is not None else FixedThreshold(a.threshold)
            r = run_conversation(conv, policy_factory(), gate, extractor, **kw)
            gate_desc = gate.describe()
        r["summary"]["gate"] = gate_desc
        summaries.append(r["summary"])
        all_qa += r["qa"]
        for key in files:
            for rec in r[key]:
                files[key].write(json.dumps(rec, default=str) + "\n")
        s = r["summary"]
        print(f"{conv.id:>12} cands={s['candidates']:5d} admitted={s['admitted_facts']:5d} live={s['live_facts']:5d} inval={s['invalidated']:3d} "
              f"admit_rate={s['admit_rate']:.2f} P={s['precision']:.2f} R={s['recall']:.2f} hit={np.nanmean([q['hit'] for q in r['qa']]):.3f} "
              f"deg2={s['shape_share_non_hub_deg_ge2']:.2f} nh-edges={s['shape_share_edges_non_hub_to_non_hub']:.2f} {s['seconds']}s", flush=True)
    for f in files.values():
        f.close()
    agg = ev.summarise(all_qa)
    tot_keys = ("candidates", "admitted_facts", "duplicates_suppressed", "live_facts", "invalidated", "poison_candidates", "poison_admitted",
                "llm_calls", "llm_input_tokens", "llm_output_tokens")
    data_file = ROOT / "data" / "raw" / ("locomo/locomo10.json" if a.dataset == "locomo" else f"longmemeval/longmemeval_{'oracle' if a.lme_variant == 'oracle' else 's_cleaned'}.json")
    agg.update({"name": name, "dataset": a.dataset, "policy": probe.name + ("+tier" if a.tier_weight else ""), "policy_label": DESCRIPTIONS.get(probe.name, probe.name),
                "gate": summaries[0]["gate"] if summaries else "", "extractor": a.extractor, "offline": a.offline, "poison_rate": a.poison_rate,
                "poison_mode": a.poison_mode, "weights": a.weights, "include_superseded": not a.live_only, "conversations": summaries,
                "totals": {k: float(np.nansum([s.get(k, 0) for s in summaries])) for k in tot_keys},
                "mean_admit_rate": float(np.nanmean([s["admit_rate"] for s in summaries])),
                "mean_precision": float(np.nanmean([s["precision"] for s in summaries])),
                "mean_recall": float(np.nanmean([s["recall"] for s in summaries])),
                "mean_size_gap": float(np.nanmean([s.get("size_gap", float("nan")) for s in summaries])) if a.match_live is not None else None,
                "shape": {k: float(np.nanmean([s[k] for s in summaries])) for k in summaries[0] if k.startswith("shape_")} if summaries else {},
                "poison_retrieved_per_question": float(np.mean([q["poison_retrieved"] for q in all_qa])) if all_qa else 0.0,
                "poison_retrieved_share": float(np.mean([q["poison_retrieved"] > 0 for q in all_qa])) if all_qa else 0.0,
                "provenance": {"code_hash": code_hash(), "extract_backend": __import__("paper2.extract", fromlist=["backend"]).backend(),
                               "fallback_sentences": getattr(__import__("paper2.extract", fromlist=["FALLBACK_SENTENCES"]), "FALLBACK_SENTENCES", None),
                               "data_file": str(data_file.relative_to(ROOT)), "data_sha256": sha256_file(data_file) if data_file.exists() else None,
                               "weights_sha256": sha256_file(Path(a.weights)) if a.weights and Path(a.weights).exists() else None,
                               "argv": sys.argv[1:], "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}})
    (out / "summary.json").write_text(json.dumps(agg, indent=2, default=str))
    o = agg["overall"]
    print(f"\n{name}: n_qa={agg['n']} evidence-hit={o.get('hit', {}).get('mean', float('nan')):.3f} ci={[round(x, 3) for x in o.get('hit', {}).get('ci95', [])]} "
          f"frac={o.get('frac', {}).get('mean', float('nan')):.3f} admit_rate={agg['mean_admit_rate']:.2f} P={agg['mean_precision']:.2f} R={agg['mean_recall']:.2f} "
          f"live={int(agg['totals']['live_facts'])} poison_admitted={int(agg['totals']['poison_admitted'])}/{int(agg['totals']['poison_candidates'])} "
          f"code={agg['provenance']['code_hash']}")
    for c, m in sorted(agg["by_category"].items()):
        print(f"  {c:>12}: " + " ".join(f"{k}={v['mean']:.3f}(n={v['n']})" for k, v in m.items()))
    print(f"written {out}")


def cmd_fit(a: argparse.Namespace) -> None:
    rows = [json.loads(l) for f in a.features for l in open(f)]
    if a.exclude_conv:
        rows = [r for r in rows if r["conv"] not in set(a.exclude_conv)]
    names = list(AMAC_FEATURE_NAMES) if a.amac else list(M.FEATURE_NAMES)
    exclude = list(a.exclude or [])
    if a.structural:
        exclude = sorted(set(exclude) | set(M.NON_STRUCTURAL))
    X = np.array([[float(r.get(n, 0.0)) for n in names] for r in rows])
    y = np.array([1.0 if r["label"] else 0.0 for r in rows])
    groups = np.array([r["conv"] for r in rows])
    res = cross_validate(X, y, groups, k=min(a.folds, len(set(groups))), seed=a.seed, exclude=exclude, names=names)
    res["kind"] = "amac" if a.amac else ("structural" if a.structural else "full")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[{res['kind']}] n={res['n']} positives={res['positives']} fit on {len(res['fit_convs'])} conversations, grouped CV AUC={res['cv_auc_mean']:.3f}")
    for n, w in sorted(zip(names, res["w"]), key=lambda x: -abs(x[1])):
        if w:
            print(f"  {n:>14} {w:+.3f}")
    print(f"written {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="paper2.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", choices=["locomo", "longmemeval"], default="locomo")
    r.add_argument("--lme-variant", choices=["s", "oracle"], default="s")
    r.add_argument("--policy", default="admit_all", choices=sorted(DESCRIPTIONS))
    r.add_argument("--extractor", choices=["oracle", "heuristic", "llm"], default="oracle")
    r.add_argument("--budget", type=float, default=None, help="online-quantile admit fraction q")
    r.add_argument("--match-live", type=float, default=None, help="bisect q until live facts = this fraction of admit-all's, within 5%% (matched size)")
    r.add_argument("--threshold", type=float, default=0.0)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--window", type=int, default=None)
    r.add_argument("--weights", default=None, help="gate weights JSON from `fit`")
    r.add_argument("--offline", action="store_true", help="score retrieval by evidence recall only; no model calls")
    r.add_argument("--live-only", action="store_true", help="retrieve over live facts only (default: superseded facts stay retrievable, annotated)")
    r.add_argument("--answer-model", default=None, help="anthropic:<model>[:<effort>] or local:<base_url>:<model>[:<temperature>]")
    r.add_argument("--extract-model", default=None)
    r.add_argument("--gate-model", default=None)
    r.add_argument("--k", type=int, default=20)
    r.add_argument("--cap", type=int, default=40)
    r.add_argument("--poison-rate", type=float, default=0.0)
    r.add_argument("--poison-mode", choices=["sourceless", "lower"], default="sourceless")
    r.add_argument("--tier-weight", action="store_true", help="multiply the policy score by the candidates' mean tier confidence (H4)")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--conv", nargs="*", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--name", default=None)
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser("fit")
    f.add_argument("--features", nargs="+", required=True, help="turns.jsonl files from admit_all runs")
    f.add_argument("--out", default=str(OUT / "gate_weights.json"))
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--exclude", nargs="*", default=None, help="features forced to zero weight")
    f.add_argument("--exclude-conv", nargs="*", default=None, help="drop these conversations' rows (leave-one-out cross-fitting)")
    f.add_argument("--structural", action="store_true", help="graph-structural features only (measures.STRUCTURAL)")
    f.add_argument("--amac", action="store_true", help="fit the A-MAC feature set instead")
    f.set_defaults(fn=cmd_fit)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
