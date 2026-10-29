"""Proxy validation analysis: does evidence-hit (offline) track official F1 (online) on LoCoMo?"""
import json, sys, math
from collections import defaultdict
sys.path.insert(0, '.')
import numpy as np
RUNS = [("admit_all", "paper2/out/online_admit_all"), ("graph_gate_struct m0.5", "paper2/out/merged_online_graph_gate_struct_m0.5"), ("random m0.5", "paper2/out/merged_online_random_m0.5")]
def load(p): return [json.loads(l) for l in open(p + "/qa.jsonl")]
print("== policy level: evidence-hit vs F1 / EM / abstained (n questions) ==")
rows = {}
for name, p in RUNS:
    qa = load(p); rows[name] = qa
    hit = np.nanmean([q["hit"] for q in qa]); f1 = np.mean([q["f1"] for q in qa]); em = np.mean([q["em"] for q in qa]); ab = np.mean([float(q.get("abstained") or 0) for q in qa])
    print(f"  {name:24} n={len(qa):4}  hit={hit:.3f}  f1={f1:.3f}  em={em:.3f}  abstained={ab:.3f}")
print("\n== per category F1 (admit_all / struct / random) and hit ==")
cats = sorted({q["category"] for q in rows["admit_all"]})
for c in cats:
    line = f"  {c:12}"
    for name, _ in RUNS:
        qs = [q for q in rows[name] if q["category"] == c]
        line += f" | {name[:10]:10} hit={np.nanmean([q['hit'] for q in qs]):.3f} f1={np.mean([q['f1'] for q in qs]):.3f}"
    print(line)
print("\n== question level, pooled over the three runs: F1 when the gold evidence was retrieved vs not ==")
allq = [q for name, _ in RUNS for q in rows[name] if not (isinstance(q["hit"], float) and math.isnan(q["hit"]))]
h1 = [q["f1"] for q in allq if q["hit"] == 1]; h0 = [q["f1"] for q in allq if q["hit"] == 0]
print(f"  hit=1: n={len(h1)} mean F1={np.mean(h1):.3f}   hit=0: n={len(h0)} mean F1={np.mean(h0):.3f}   difference={np.mean(h1)-np.mean(h0):+.3f}")
from itertools import groupby
print("  per category difference (hit=1 minus hit=0 F1):")
for c in cats:
    a = [q["f1"] for q in allq if q["category"] == c and q["hit"] == 1]; b = [q["f1"] for q in allq if q["category"] == c and q["hit"] == 0]
    if a and b: print(f"    {c:12} {np.mean(a)-np.mean(b):+.3f}  (n {len(a)}/{len(b)})")
print("\n== paired per-conversation differences, struct minus random: hit and F1 (sign agreement is the check) ==")
byc = lambda qa: {k: list(v) for k, v in groupby(sorted(qa, key=lambda q: q["conv"]), key=lambda q: q["conv"])}
S, R = byc(rows["graph_gate_struct m0.5"]), byc(rows["random m0.5"])
agree = 0; n = 0
for c in sorted(S):
    dh = np.nanmean([q["hit"] for q in S[c]]) - np.nanmean([q["hit"] for q in R[c]]); df = np.mean([q["f1"] for q in S[c]]) - np.mean([q["f1"] for q in R[c]])
    agree += (dh > 0) == (df > 0); n += 1
    print(f"  {c}: d_hit={dh:+.3f}  d_f1={df:+.3f}")
print(f"  sign agreement: {agree}/{n} conversations")
dh = [np.nanmean([q["hit"] for q in S[c]]) - np.nanmean([q["hit"] for q in R[c]]) for c in sorted(S)]
df = [np.mean([q["f1"] for q in S[c]]) - np.mean([q["f1"] for q in R[c]]) for c in sorted(S)]
def _rank(x):
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x)+1); return r
rho = np.corrcoef(_rank(np.array(dh)), _rank(np.array(df)))[0, 1]
print(f"  Spearman(d_hit, d_f1) over conversations: {rho:+.3f}")
# cost
tot = defaultdict(float)
for name, p in RUNS:
    s = json.load(open(p + "/summary.json")); t = s.get("totals", {})
    for k in ("llm_calls", "llm_input_tokens", "llm_output_tokens"): tot[k] += float(t.get(k, 0) or 0)
print(f"\n== cost: calls={int(tot['llm_calls'])} in={int(tot['llm_input_tokens'])} out={int(tot['llm_output_tokens'])} -> ${tot['llm_input_tokens']*0.1e-6 + tot['llm_output_tokens']*0.5e-6:.2f} at Haiku 5.5 prices")
