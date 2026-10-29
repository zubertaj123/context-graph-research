"""Figures from the merged leave-one-out runs: value retained vs memory size, and per-conversation
paired differences. Read-only over paper2/out; writes to paper2/out/fig/.

  python -m paper2.figures
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper2" / "out"
LABELS = {"graph_gate": "structural gate (prior weights)", "graph_gate_struct": "structural-only gate (fitted, LOO)",
          "graph_gate_full": "structural + surface gate (fitted, LOO)", "amac_fit": "A-MAC features (fitted, LOO)",
          "amac_lite": "A-MAC features (hand weights)", "heat": "length + revisit heuristic", "impact_2010": "Impact (2010)",
          "siv_2010": "Trust x Impact (2010)", "random": "random"}
ORDER = ["graph_gate", "graph_gate_struct", "graph_gate_full", "amac_fit", "amac_lite", "heat", "impact_2010", "random", "siv_2010"]


def budget_curve(prefix: str = "merged_loo", metric: str = "hit") -> Path | None:
    runs: dict[str, list] = {}
    for f in glob.glob(str(OUT / f"{prefix}_*_m*/summary.json")):
        d = json.loads(Path(f).read_text())
        m = re.match(rf"{prefix}_(.+)_m([0-9.]+)$", d["name"])
        if not m or metric not in d["overall"]:
            continue
        runs.setdefault(m.group(1), []).append((d["totals"]["live_facts"], d["overall"][metric]["mean"], d["overall"][metric]["ci95"], float(m.group(2))))
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for pol in ORDER:
        pts = sorted(runs.get(pol, []))
        if pts:
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[[p[1] - p[2][0] for p in pts], [p[2][1] - p[1] for p in pts]],
                        marker="o", capsize=2, lw=1.2, label=LABELS.get(pol, pol))
    aa = OUT / "locomo_admit_all" / "summary.json"
    if aa.exists():
        d = json.loads(aa.read_text())
        ax.axhline(d["overall"][metric]["mean"], ls="--", c="k", lw=0.8)
        ax.annotate(f"admit all ({int(d['totals']['live_facts'])} facts)", (0.02, d["overall"][metric]["mean"] + 0.01), xycoords=("axes fraction", "data"), fontsize=8)
    ax.set_xlabel("live facts in memory after ingestion (10 conversations)")
    ax.set_ylabel({"hit": "evidence-hit", "frac": "evidence fraction", "f1": "token F1", "mrr": "MRR"}.get(metric, metric))
    ax.set_title("LoCoMo offline arm, leave-one-out gates, memory matched to 25 / 50 / 75% of admit-all (95% question-bootstrap CIs)", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    (OUT / "fig").mkdir(exist_ok=True)
    out = OUT / "fig" / f"budget_curve_{metric}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    return out


def paired_dots(ref: str, others: list[str], metric: str = "hit") -> Path | None:
    def per_conv(run):
        by: dict[str, list] = {}
        for l in open(OUT / run / "qa.jsonl"):
            r = json.loads(l)
            if r.get(metric) == r.get(metric) and r.get(metric) is not None:
                by.setdefault(r["conv"], []).append(float(r[metric]))
        return {c: sum(v) / len(v) for c, v in by.items()}
    base = per_conv(ref)
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(others) + 1.5))
    for i, o in enumerate(others):
        d = per_conv(o)
        diffs = [base[c] - d[c] for c in sorted(set(base) & set(d))]
        ax.scatter(diffs, [i] * len(diffs), s=18, alpha=0.7)
        ax.scatter([sum(diffs) / len(diffs)], [i], marker="|", s=300, c="k")
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels([LABELS.get(re.sub(r"^merged_loo_|_m[0-9.]+$", "", o), o) for o in others], fontsize=8)
    ax.set_xlabel(f"per-conversation difference in {metric}: {LABELS.get(re.sub(r'^merged_loo_|_m[0-9.]+$', '', ref), ref)} minus the policy named")
    ax.grid(alpha=0.3, axis="x")
    (OUT / "fig").mkdir(exist_ok=True)
    out = OUT / "fig" / f"paired_{metric}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    return out


if __name__ == "__main__":
    for m in ("hit", "frac", "mrr", "f1"):
        p = budget_curve(metric=m)
        if p:
            print("written", p)
    ref = "merged_loo_graph_gate_struct_m0.5"
    others = [f"merged_loo_{p}_m0.5" for p in ("random", "siv_2010", "impact_2010", "heat", "amac_lite", "amac_fit", "graph_gate", "graph_gate_full")]
    if (OUT / ref).exists():
        print("written", paired_dots(ref, [o for o in others if (OUT / o).exists()]))
