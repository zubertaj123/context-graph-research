"""Read-only analysis over run outputs: univariate AUC of each structural feature against the
evidence label, SIVa-vs-label association, a policy table across budgets, and the per-conversation
feature-shape / clock diagnostic (critique A4.2, A9).

  python -m paper2.analysis features paper2/out/fit_train_admit_all/turns.jsonl
  python -m paper2.analysis table paper2/out/test_*/summary.json
  python -m paper2.analysis shape paper2/out/fit_train_admit_all/turns.jsonl

Spearman here is the average-rank form (ties get the mean of the ranks they span), computed with numpy.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper2.measures import FEATURE_NAMES  # noqa: E402
from paper2.policies import auc  # noqa: E402

SHAPE_FEATURES = ["siva_2010", "siva_cons", "siva_scaled", "siv_2010", "score"]       # appended to FEATURE_NAMES in `shape`


def average_rank(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties replaced by the mean of the ranks they would occupy (the Spearman convention)."""
    x = np.asarray(x, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts + 1.0           # first rank of each tie group
    return (starts + (counts - 1) / 2.0)[inv.ravel()]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Average-rank Spearman correlation: Pearson on average ranks, so ties (binary labels, t_edge, anchoring)
    are handled. Pairs with a non-finite value are dropped; NaN when either side is constant."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return float("nan")
    rx, ry = average_rank(x), average_rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _load_rows(paths: list[str]) -> list[dict]:
    files = sorted(set(sum((glob.glob(p) or [p] for p in paths), [])))
    return [json.loads(l) for p in files for l in open(p)]


def cmd_features(paths: list[str]) -> None:
    rows = _load_rows(paths)
    y = np.array([1.0 if r["label"] else 0.0 for r in rows])
    print(f"n={len(rows)} positive rate={y.mean():.3f}  (label: the turn is a gold evidence turn for some question)")
    print(f"{'feature':>16} {'AUC':>6} {'rho':>7} {'mean+':>8} {'mean-':>8}  distinct   (rho = average-rank Spearman with the label)")
    for n in FEATURE_NAMES + ["siv_2010", "siva_2010", "siva_cons", "siva_scaled", "score"]:
        if not all(n in r for r in rows):
            continue
        x = np.array([float(r[n]) for r in rows])
        print(f"{n:>16} {auc(y, x):6.3f} {spearman(x, y):7.3f} {x[y == 1].mean():8.3f} {x[y == 0].mean():8.3f}  {len(set(x.round(6)))}")
    # how degenerate are the 2010 measures?
    te = np.array([r["t_edge"] for r in rows])
    print(f"\nT_edge == 0 on {np.mean(te == 0):.1%} of candidate turns; T_edge == 1 on {np.mean(te == 1):.1%}; "
          f"SIVp(2010) == 0 on {np.mean(np.array([r['siv_2010'] for r in rows]) == 0):.1%}")


def shape_rows(rows: list[dict]) -> dict:
    """Per conversation (file order = candidate order): n, share of candidate turns with anchoring = 0, t_edge = 0,
    impact_log = 0 (and siv_2010 = 0), and the average-rank Spearman of every feature with within-conversation
    position (0..n-1). The pooled row uses relative position i/(n-1) so that conversations of different length
    are comparable. Also Spearman(siva_2010, siva_plain) per conversation (critique A4.1)."""
    convs: dict[str, list[dict]] = {}
    for r in rows:
        convs.setdefault(r["conv"], []).append(r)
    feats = [f for f in FEATURE_NAMES + SHAPE_FEATURES if all(f in r for r in rows)]
    zero_cols = [c for c in ("anchoring", "t_edge", "impact_log", "siv_2010") if c in feats]
    out = {"features": feats, "zero_cols": zero_cols, "per_conv": {}, "pooled": {}}
    rel_all, vals_all = [], {f: [] for f in feats}
    for c, rs in convs.items():
        n = len(rs)
        pos = np.arange(n, dtype=float)
        rel = pos / (n - 1) if n > 1 else np.zeros(n)
        vals = {f: np.array([float(r[f]) for r in rs]) for f in feats}
        entry = {"n": n, "zero_share": {z: float(np.mean(vals[z] == 0)) for z in zero_cols},
                 "rho_position": {f: spearman(vals[f], pos) for f in feats}}
        if "siva_2010" in feats and "siva_cons" in feats:
            entry["rho_siva_2010_plain"] = spearman(vals["siva_2010"], vals["siva_cons"])
        out["per_conv"][c] = entry
        rel_all.append(rel)
        for f in feats:
            vals_all[f].append(vals[f])
    rel_all = np.concatenate(rel_all) if rel_all else np.array([])
    vals_all = {f: np.concatenate(v) if v else np.array([]) for f, v in vals_all.items()}
    out["pooled"] = {"n": int(len(rel_all)), "zero_share": {z: float(np.mean(vals_all[z] == 0)) for z in zero_cols},
                     "rho_position": {f: spearman(vals_all[f], rel_all) for f in feats}}
    if "siva_2010" in feats and "siva_cons" in feats:
        out["pooled"]["rho_siva_2010_plain"] = spearman(vals_all["siva_2010"], vals_all["siva_cons"])
    return out


def cmd_shape(paths: list[str]) -> None:
    rows = _load_rows(paths)
    s = shape_rows(rows)
    convs = list(s["per_conv"])
    print(f"n={len(rows)} candidate turns in {len(convs)} conversations; position = order of the candidate turn within its conversation")
    print("\nshare of candidate turns at zero (the degenerate cells, critique B(v), E1):")
    print(f"{'conv':>10} {'n':>5} " + " ".join(f"{z + '=0':>13}" for z in s["zero_cols"]))
    for c in convs + ["pooled"]:
        e = s["per_conv"][c] if c != "pooled" else s["pooled"]
        print(f"{c:>10} {e['n']:>5d} " + " ".join(f"{e['zero_share'][z]:13.3f}" for z in s["zero_cols"]))
    print("\naverage-rank Spearman of each feature with within-conversation position (the clock diagnostic, critique A4.2;"
          " pooled uses relative position i/(n-1)):")
    print(f"{'feature':>16} " + " ".join(f"{c:>8}" for c in convs) + f" {'pooled':>8}")
    for f in s["features"]:
        vals = [s["per_conv"][c]["rho_position"][f] for c in convs] + [s["pooled"]["rho_position"][f]]
        print(f"{f:>16} " + " ".join(f"{v:8.3f}" if v == v else f"{'nan':>8}" for v in vals))
    if "rho_siva_2010_plain" in s["pooled"]:
        vals = [s["per_conv"][c]["rho_siva_2010_plain"] for c in convs] + [s["pooled"]["rho_siva_2010_plain"]]
        print(f"\n{'rho(siva_2010, siva_plain)':>28} " + " ".join(f"{v:8.3f}" if v == v else f"{'nan':>8}" for v in vals)
              + "   (critique A4.1: 2010 weights vs the plain new-triple ratio)")


def cmd_table(paths: list[str]) -> None:
    rows = []
    nan = float("nan")
    for p in sorted(set(sum((glob.glob(x) for x in paths), []))):
        d = json.loads(Path(p).read_text())
        o = d["overall"]
        rows.append((d.get("policy", "?"), d.get("gate", "?"), d.get("mean_admit_rate", nan), int(d["totals"].get("live_facts", 0)),
                     o.get("hit", {}).get("mean", nan), *o.get("hit", {}).get("ci95", [nan, nan]), o.get("frac", {}).get("mean", nan),
                     d.get("mean_precision", nan), d.get("mean_recall", nan), o.get("f1", {}).get("mean", nan),
                     int(d["totals"].get("llm_calls", 0)), d.get("name", Path(p).parent.name)))
    rows.sort(key=lambda r: (str(r[1]), -r[4] if r[4] == r[4] else 0.0))
    print(f"{'policy':>14} {'gate':>22} {'admit':>6} {'live':>6} {'hit':>6} {'ci95':>15} {'frac':>6} {'P':>5} {'R':>5} {'F1':>6} {'llm':>5}  run")
    for r in rows:
        print(f"{str(r[0]):>14} {str(r[1]):>22} {r[2]:6.2f} {r[3]:6d} {r[4]:6.3f} [{r[5]:.3f},{r[6]:.3f}] {r[7]:6.3f} {r[8]:5.2f} {r[9]:5.2f} {r[10]:6.3f} {r[11]:5d}  {r[12]}")


COMMANDS = {"features": cmd_features, "table": cmd_table, "shape": cmd_shape}

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python -m paper2.analysis {{{'|'.join(COMMANDS)}}} <paths...>")
    cmd, *args = sys.argv[1:]
    COMMANDS[cmd](args)
