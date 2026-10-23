"""Write paper/numbers.tex from paper2/out. Keeps every macro name the skeleton defines (unfilled ones stay
\\todo{xx}) and appends a block of extra macros for policies the skeleton did not anticipate (the
leave-one-out fitted gates, A-MAC fitted, Impact alone). Read-only over out/.

  python -m paper2.fill_numbers            # after sh paper2/sweep.sh
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper2" / "out"
TEX = ROOT / "paper2" / "paper" / "numbers.tex"
from paper2.measures import FEATURE_NAMES  # noqa: E402
from paper2.policies import auc  # noqa: E402

POLICY_CODE = {"admit_all": "AdmitAll", "random": "Random", "heat": "Heat", "siv_2010": "SivTen", "graph_gate": "GraphGate", "amac_lite": "AmacLite",
               "llm_gate": "LlmGate", "graph_gate_struct": "GraphGateStruct", "graph_gate_full": "GraphGateFull", "amac_fit": "AmacFit", "impact_2010": "ImpactTen"}
CAT_CODE = {"single-hop": "SingleHop", "multi-hop": "MultiHop", "temporal": "Temporal", "open-domain": "OpenDomain", "adversarial": "Adversarial"}


def load(name: str) -> dict | None:
    p = OUT / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return None
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main(size: str = "0.5") -> None:
    vals: dict[str, str] = {}
    extra: dict[str, str] = {}

    def put(name, v, nd=3, table=vals):
        s = fmt(v, nd)
        if s is not None:
            table[name] = s

    aa = load("locomo_admit_all")
    if aa:
        put("hitLcAdmitAll", aa["overall"]["hit"]["mean"]); put("admitLcAdmitAll", 1.0, 2); put("liveLcAdmitAll", int(aa["totals"]["live_facts"]), 0)
        put("callsLcAdmitAll", 0, 0); put("nCandTurnsLc", int(sum(c["candidate_turns"] for c in aa["conversations"])), 0)
        put("nCandTriplesLc", int(aa["totals"]["candidates"]), 0)
        sh = aa.get("shape", {})
        put("nonHubDegTwoShareLc", sh.get("shape_share_non_hub_deg_ge2")); put("nonHubEdgeShareLc", sh.get("shape_share_edges_non_hub_to_non_hub"))
        put("consFireRateLc", aa["totals"]["invalidated"] / max(aa["totals"]["admitted_facts"], 1))
        for cat, m in aa["by_category"].items():
            if cat in CAT_CODE and "hit" in m:
                put(f"catLc{CAT_CODE[cat]}AdmitAll", m["hit"]["mean"])
        rows = [json.loads(l) for l in open(OUT / "locomo_admit_all" / "turns.jsonl")]
        y = np.array([1.0 if r["label"] else 0.0 for r in rows])
        te = np.array([r["t_edge"] for r in rows]); siv = np.array([r["siv_2010"] for r in rows])
        put("fracTedgeZero", float((te == 0).mean())); put("fracTedgeOne", float((te == 1).mean())); put("nDistinctTedge", len(set(te.round(6))), 0)
        put("fracSingleTriple", float(np.mean([r["n_triples"] == 1 for r in rows])))
        pos = np.array([r["position"] for r in rows]); s2 = np.array([r["siva_cons"] for r in rows]); sp = np.array([r["siva_2010"] for r in rows])
        from paper2.analysis import spearman
        put("corrSivaPosition", spearman(s2, pos)); put("corrSivaSivaPlain", spearman(s2, sp)); put("corrSivaPlainSize", spearman(sp, np.array([r["n_live"] for r in rows])))
        put("corrImpactSize", spearman(np.array([r["impact"] for r in rows]), np.array([r["n_live"] for r in rows])))
        for macro, feat in (("ablAucSivTen", "siv_2010"), ("ablAucAnchor", "anchoring"), ("ablAucSoft", "t_soft"), ("ablAucLogImpact", "impact_log"), ("ablAucNovelty", "novelty")):
            put(macro, auc(y, np.array([float(r[feat]) for r in rows])))
    for pol, code in POLICY_CODE.items():
        d = load(f"merged_loo_{pol}_m{size}")
        if not d:
            continue
        table = vals if f"hitLc{code}" in SKELETON else extra
        put(f"hitLc{code}", d["overall"]["hit"]["mean"], table=table); put(f"admitLc{code}", d["mean_admit_rate"], 2, table=table)
        put(f"liveLc{code}", int(d["totals"]["live_facts"]), 0, table=table); put(f"callsLc{code}", int(d["totals"].get("llm_calls", 0)), 0, table=table)
        put(f"hitCiLoLc{code}", d["overall"]["hit"]["ci95"][0], table=extra); put(f"hitCiHiLc{code}", d["overall"]["hit"]["ci95"][1], table=extra)
        put(f"sizeGapLc{code}", d.get("mean_size_gap"), table=extra)
        for cat, m in d["by_category"].items():
            if cat in CAT_CODE and "hit" in m:
                put(f"catLc{CAT_CODE[cat]}{code}", m["hit"]["mean"], table=(vals if f"catLc{CAT_CODE[cat]}{code}" in SKELETON else extra))
        for s in ("0.25", "0.75"):
            d2 = load(f"merged_loo_{pol}_m{s}")
            if d2:
                put(f"hitLc{code}At{'Quarter' if s == '0.25' else 'ThreeQuarter'}", d2["overall"]["hit"]["mean"], table=extra)
    # poisoning: 10% rate -> 'Ten'; sourceless -> Src; lower tier + multiplier -> Tier
    for pol, code in (("random", "Random"), ("graph_gate", "GraphGate"), ("graph_gate_struct", "GraphGateStruct")):
        for cond, suffix in (("sourceless", "Src"), ("lower_tier", "Tier")):
            d = load(f"merged_poison_{cond}_{pol}_m{size}")
            if d and d["totals"]["poison_candidates"]:
                put(f"poisAdmit{code}Ten{suffix}", d["totals"]["poison_admitted"] / d["totals"]["poison_candidates"], table=(vals if f"poisAdmit{code}Ten{suffix}" in SKELETON else extra))
                put(f"poisRetrievedShare{code}Ten{suffix}", d.get("poison_retrieved_share"), table=extra)
                put("poisN", int(d["totals"]["poison_candidates"]), 0)
    text = TEX.read_text()
    filled = 0
    for name, v in vals.items():      # one macro per line in numbers.tex: replace the whole definition line
        new, n = re.subn(rf"^\\newcommand\{{\\{name}\}}\{{.*$", lambda m: f"\\newcommand{{\\{name}}}{{{v}}}", text, count=1, flags=re.M)
        if n:
            text, filled = new, filled + 1
    text = text.split("\n% ---- Extra macros written by paper2.fill_numbers")[0].rstrip() + "\n"
    if extra:
        text += "\n% ---- Extra macros written by paper2.fill_numbers (policies the skeleton did not anticipate; wire into main.tex as needed)\n"
        text += "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(extra.items()))
    TEX.write_text(text)
    print(f"filled {filled} skeleton macros, wrote {len(extra)} extra macros -> {TEX}")


SKELETON = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", TEX.read_text())) if TEX.exists() else set()

if __name__ == "__main__":
    main(*sys.argv[1:])
