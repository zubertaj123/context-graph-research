"""Write paper/numbers_online.tex: the macros of the 19 Oct 2026 online, Mem0-gate, proxy-validation and
exploratory-H3 runs, plus the hand-check and non-speaker-subject counts from docs 08 and 09. Read-only over
paper2/out; a run that is missing yields the macro value "--" so the paper always compiles.
  .venv/bin/python paper2/paper/make_numbers_online.py
"""
from __future__ import annotations
import collections, glob, json, math, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper2" / "out"
TEX = ROOT / "paper2" / "paper" / "numbers_online.tex"
M: dict[str, str] = {}
# every macro the paper uses is pre-registered as "--" so a missing run never breaks the build
_CODES = ("AdmitAll", "Struct", "Random"); _CATS = ("SingleHop", "MultiHop", "Temporal", "OpenDomain", "Adversarial")
for tag in ("One", "Two"):
    for code in _CODES:
        for k in ("fone", "em", "abst", "hit"): M[f"{k}{tag}{code}"] = "--"
        for cc in _CATS: M[f"fone{tag}{code}{cc}"] = "--"; M[f"hit{tag}{code}{cc}"] = "--"
    for k in ("hitCondOne", "hitCondZero", "hitCondDiff", "hitCondNone", "hitCondNzero", "rhoDhitDfone", "signAgree", "dFoneStructRandom", "pFoneStructRandom", "winsFoneStructRandom", "ciFoneStructRandomLo", "ciFoneStructRandomHi"): M[f"{k}{tag}"] = "--"
    for cc in _CATS[:4]: M[f"hitCondDiff{tag}{cc}"] = "--"
for k in ("memzNconv", "memzAdmitRate", "memzAdmitRateMin", "memzAdmitRateMax", "memzLive", "memzCalls", "memzParseFail", "memzAdd", "memzUpdate", "memzDelete", "memzNone", "memzCallsPerK", "memzTokensPerK", "memzFracOfAdmitAll", "memzHit",
          "hitMemz", "foneMemz", "liveMemz", "hitStructAtMemz", "foneStructAtMemz", "liveStructAtMemz", "hitRandomAtMemz", "foneRandomAtMemz", "liveRandomAtMemz", "hOneDiffFone", "hOneCiLo", "hOneCiHi", "hOneDiffHit"): M[k] = "--"
def put(k, v, fmt="{:.3f}"):
    M[k] = "--" if v is None or (isinstance(v, float) and math.isnan(v)) else (fmt.format(v) if not isinstance(v, str) else v)
def qa(d):
    p = OUT / d / "qa.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None
def summ(d):
    p = OUT / d / "summary.json"
    return json.load(open(p)) if p.exists() else None
# ---- static counts from the documents (08-extraction-check.md, 09-nonspeaker-subjects.md) ----
for k, v in {"extHandCheckOk": "8", "extHandCheckWeak": "4", "extHandCheckWrong": "6", "extHandCheckN": "18", "extFallbackShare": "2.9\\%",
             "extFunctionalTriples": "8", "extInvalidations": "0", "obsSpeakerSubjShare": "84.6\\%", "obsPossSubjShare": "6.0\\%",
             "obsNonSpeakerSubjShare": "2.3\\%", "obsParseFailShare": "6.5\\%", "rawTurnsTwoNE": "1.0\\%", "rawTurnsOneNE": "11.8\\%",
             "lmeTurnsTwoNE": "15.5\\%", "lmeTurnsOneNE": "41.2\\%", "ruleCedgesMin": "4", "ruleCedgesMax": "64", "ruleBedgesMax": "6",
             "nObsSentences": "2{,}541"}.items():
    put(k, v)
# ---- primary permutation test (hit) parsed from sweep.log ----
log = (OUT / "sweep.log").read_text() if (OUT / "sweep.log").exists() else ""
blk = log.split("== primary test")[1].split("== prior gate")[0] if "== primary test" in log else ""
names = {"random": "Random", "siv_2010": "SivTen", "impact_2010": "ImpactTen", "heat": "Heat", "amac_lite": "AmacLite", "amac_fit": "AmacFit", "graph_gate": "GraphGate", "graph_gate_full": "GraphGateFull"}
for line in blk.splitlines():
    m = re.match(r"\s*merged_loo_(\w+?)_m0\.5\s+10\s+([+-][\d.]+)\s+\[([+-][\d.]+),([+-][\d.]+)\]\s+([\d.]+)\s+([\d.]+)\s+\[[^\]]*\]\s+\[[^\]]*\]\s+\d+/\d+\s+(\d+)/10", line)
    if m and m.group(1) in names:
        c = names[m.group(1)]
        put(f"perm{c}Mean", m.group(2)); put(f"perm{c}CiLo", m.group(3)); put(f"perm{c}CiHi", m.group(4))
        put(f"perm{c}P", m.group(5)); put(f"perm{c}Holm", m.group(6)); put(f"perm{c}Wins", m.group(7))
# ---- online passes ----
def online(prefix, tag):
    runs = {"AdmitAll": f"{prefix}_admit_all", "Struct": f"merged_{prefix}_graph_gate_struct_m0.5", "Random": f"merged_{prefix}_random_m0.5"}
    rows = {}
    for code, d in runs.items():
        q = qa(d); rows[code] = q
        if q is None:
            for k in ("fone", "em", "abst", "hit"): put(f"{k}{tag}{code}", None)
            continue
        put(f"fone{tag}{code}", np.mean([r["f1"] for r in q])); put(f"em{tag}{code}", np.mean([r["em"] for r in q]))
        put(f"abst{tag}{code}", np.mean([float(r.get("abstained") or 0) for r in q])); put(f"hit{tag}{code}", np.nanmean([r["hit"] for r in q]))
        for cat, cc in (("single-hop", "SingleHop"), ("multi-hop", "MultiHop"), ("temporal", "Temporal"), ("open-domain", "OpenDomain"), ("adversarial", "Adversarial")):
            qs = [r for r in q if r["category"] == cat]
            put(f"fone{tag}{code}{cc}", np.mean([r["f1"] for r in qs]) if qs else None)
            put(f"hit{tag}{code}{cc}", np.nanmean([r["hit"] for r in qs]) if qs else None)
    allq = [r for q in rows.values() if q for r in q if not (isinstance(r["hit"], float) and math.isnan(r["hit"])) and r["category"] != "adversarial"]
    if allq:
        h1 = [r["f1"] for r in allq if r["hit"] == 1]; h0 = [r["f1"] for r in allq if r["hit"] == 0]
        put(f"hitCondOne{tag}", np.mean(h1)); put(f"hitCondZero{tag}", np.mean(h0)); put(f"hitCondDiff{tag}", np.mean(h1) - np.mean(h0), "{:+.3f}")
        put(f"hitCondNone{tag}", len(h1), "{:d}"); put(f"hitCondNzero{tag}", len(h0), "{:d}")
        for cat, cc in (("single-hop", "SingleHop"), ("multi-hop", "MultiHop"), ("temporal", "Temporal"), ("open-domain", "OpenDomain")):
            a = [r["f1"] for r in allq if r["category"] == cat and r["hit"] == 1]; b = [r["f1"] for r in allq if r["category"] == cat and r["hit"] == 0]
            put(f"hitCondDiff{tag}{cc}", (np.mean(a) - np.mean(b)) if a and b else None, "{:+.3f}")
    S, R = rows["Struct"], rows["Random"]
    if S and R:
        byc = lambda q: {k: [r for r in q if r["conv"] == k] for k in sorted({r["conv"] for r in q})}
        bs, br = byc(S), byc(R)
        dh = [np.nanmean([r["hit"] for r in bs[c]]) - np.nanmean([r["hit"] for r in br[c]]) for c in bs]
        df = [np.mean([r["f1"] for r in bs[c]]) - np.mean([r["f1"] for r in br[c]]) for c in bs]
        rank = lambda x: np.argsort(np.argsort(x)) + 1
        put(f"rhoDhitDfone{tag}", np.corrcoef(rank(np.array(dh)), rank(np.array(df)))[0, 1], "{:+.2f}")
        put(f"signAgree{tag}", sum((a > 0) == (b > 0) for a, b in zip(dh, df)), "{:d}")
        # exact sign-flip permutation on the F1 differences (same statistic as paper2.stats)
        d = np.array(df); k = len(d); obs = d.mean(); cnt = 0
        for mask in range(1 << k):
            s = sum(d[i] if (mask >> i) & 1 else -d[i] for i in range(k))
            cnt += abs(s / k) >= abs(obs) - 1e-12
        put(f"dFoneStructRandom{tag}", obs, "{:+.3f}"); put(f"pFoneStructRandom{tag}", cnt / (1 << k)); put(f"winsFoneStructRandom{tag}", int((d > 0).sum()), "{:d}")
        from math import sqrt
        se = d.std(ddof=1) / sqrt(k); t975 = 2.262  # k-1 = 9 df
        put(f"ciFoneStructRandomLo{tag}", obs - t975 * se, "{:+.3f}"); put(f"ciFoneStructRandomHi{tag}", obs + t975 * se, "{:+.3f}")
online("online", "One"); online("online2", "Two")
# ---- Mem0 gate (offline) and the policies matched to its size ----
ms = [summ(f"mem0_gate_{c}") for c in sorted({Path(d).name.split("mem0_gate_")[1] for d in glob.glob(str(OUT / "mem0_gate_conv-*"))})]
ms = [s for s in ms if s]
if ms:
    convs = [s["conversations"][0] for s in ms]
    put("memzNconv", len(convs), "{:d}"); put("memzAdmitRate", np.mean([c["admit_rate"] for c in convs]))
    put("memzAdmitRateMin", min(c["admit_rate"] for c in convs)); put("memzAdmitRateMax", max(c["admit_rate"] for c in convs))
    tot = lambda k: sum(c.get(k, 0) or 0 for c in convs)
    put("memzLive", tot("live_facts"), "{:d}"); put("memzCalls", tot("llm_calls"), "{:d}"); put("memzParseFail", tot("llm_parse_failures"), "{:d}")
    ev = {k: tot(f"mem0_{k}") for k in ("add", "update", "delete", "none")}
    put("memzAdd", ev["add"], "{:d}"); put("memzUpdate", ev["update"], "{:d}"); put("memzDelete", ev["delete"], "{:d}"); put("memzNone", ev["none"], "{:d}")
    nc = tot("candidates"); put("memzCallsPerK", 1000 * tot("llm_calls") / nc, "{:.0f}"); put("memzTokensPerK", 1000 * (tot("llm_input_tokens") + tot("llm_output_tokens")) / nc, "{:,.0f}")
    aa = summ("locomo_admit_all")
    if aa:
        live_aa = {c["conv"]: c["live_facts"] for c in aa["conversations"]}
        put("memzFracOfAdmitAll", np.mean([c["live_facts"] / live_aa[c["conv"]] for c in convs if c["conv"] in live_aa]))
    allqa = [r for s in ms for r in qa(s["name"])] if all((OUT / s["name"] / "qa.jsonl").exists() for s in ms) else []
    put("memzHit", np.nanmean([r["hit"] for r in allqa]) if allqa else None)
# matched-at-Mem0-size runs (offline + online), if present
for pol, code in (("mem0_gate", "Memz"), ("graph_gate_struct", "StructAtMemz"), ("random", "RandomAtMemz")):
    d = f"merged_h1_{pol}"
    q = qa(d)
    put(f"hit{code}", np.nanmean([r["hit"] for r in q]) if q else None)
    put(f"fone{code}", np.mean([r["f1"] for r in q]) if q and "f1" in q[0] else None)
    put(f"live{code}", int(summ(d)["totals"]["live_facts"]) if summ(d) and "totals" in summ(d) else None, "{:d}")
S, Mz = qa("merged_h1_graph_gate_struct"), qa("merged_h1_mem0_gate")
if S and Mz and "f1" in S[0]:
    byc = lambda q: {k: [r for r in q if r["conv"] == k] for k in sorted({r["conv"] for r in q})}
    bs, bm = byc(S), byc(Mz)
    d = np.array([np.mean([r["f1"] for r in bs[c]]) - np.mean([r["f1"] for r in bm[c]]) for c in bs]); k = len(d); obs = d.mean()
    se = d.std(ddof=1) / math.sqrt(k); put("hOneDiffFone", 100 * obs, "{:+.2f}"); put("hOneCiLo", 100 * (obs - 2.262 * se), "{:+.2f}"); put("hOneCiHi", 100 * (obs + 2.262 * se), "{:+.2f}")
    dh = np.array([np.nanmean([r["hit"] for r in bs[c]]) - np.nanmean([r["hit"] for r in bm[c]]) for c in bs]); put("hOneDiffHit", dh.mean(), "{:+.3f}")
# ---- exploratory H3 on LoCoMo: evidence rate of candidate turns by anchoring ----
turns_p = OUT / "locomo_admit_all" / "turns.jsonl"
if turns_p.exists():
    turns = [json.loads(l) for l in open(turns_p)]
    data = json.load(open(ROOT / "data/raw/locomo/locomo10.json"))
    CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    ev = collections.defaultdict(set)
    for c in data:
        for q in c["qa"]:
            for e in q.get("evidence", []) or []:
                ev[(c["sample_id"], CAT.get(q.get("category"), "?"))].add(e)
    def wilson(k, n, z=1.96):
        p = k / n; dd = 1 + z * z / n; c = (p + z * z / (2 * n)) / dd; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
        return c - h, c + h
    for cat, cc in (("single-hop", "SingleHop"), ("multi-hop", "MultiHop"), ("temporal", "Temporal"), ("open-domain", "OpenDomain"), ("adversarial", "Adversarial"), ("ALL", "All")):
        rows = {0: [0, 0], 1: [0, 0]}
        for t in turns:
            a = t.get("anchoring")
            if a is None or (isinstance(a, float) and math.isnan(a)) or t.get("n_nonhub", 0) == 0 or a not in (0.0, 1.0): continue
            is_ev = (t["turn"] in ev[(t["conv"], cat)]) if cat != "ALL" else any(t["turn"] in ev[(t["conv"], c)] for c in CAT.values())
            rows[int(a)][0] += is_ev; rows[int(a)][1] += 1
        (k0, n0), (k1, n1) = rows[0], rows[1]
        put(f"hThreeNew{cc}", k0 / n0); put(f"hThreeKnown{cc}", k1 / n1); put(f"hThreeDiff{cc}", k0 / n0 - k1 / n1, "{:+.3f}")
        lo0, hi0 = wilson(k0, n0); lo1, hi1 = wilson(k1, n1)
        put(f"hThreeNewCi{cc}", f"[{lo0:.3f}, {hi0:.3f}]"); put(f"hThreeKnownCi{cc}", f"[{lo1:.3f}, {hi1:.3f}]"); put(f"hThreeNnew{cc}", n0, "{:d}"); put(f"hThreeNknown{cc}", n1, "{:d}")
# ---- answer-call cost from the cache (Haiku 5.5, thinking off) ----
tin = tout = n = 0
for f in glob.glob(str(OUT / "cache" / "anthropic_claude-haiku-5-5_low_think_off_temp_default" / "*.json")):
    r = json.load(open(f)); tin += r["input_tokens"]; tout += r["output_tokens"]; n += 1
put("haikuCalls", n, "{:,d}"); put("haikuInTokens", tin / 1e6, "{:.1f}"); put("haikuOutTokens", tout / 1e6, "{:.2f}"); put("haikuCostUsd", tin * 0.1e-6 + tout * 0.5e-6, "{:.2f}")
TEX.write_text("% generated by paper2/paper/make_numbers_online.py; do not edit\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
print(f"wrote {len(M)} macros -> {TEX}; missing:", sum(1 for v in M.values() if v == "--"))
