"""Comparisons between runs on LoCoMo, where the inference unit is the conversation (critique C1, C9).

For a reference run and each other run, over the questions both runs answered:

  primary    per-conversation mean paired difference in the metric (reference - other), one number per
             conversation; exact sign-flip permutation test on those k numbers (all 2^k sign assignments
             for k <= 12, Monte Carlo with a fixed seed above that), two-sided p, with the minimum
             attainable p (2 / 2^k) stated; 95% t-interval with k - 1 degrees of freedom.
  interval   cluster bootstrap over conversations (an interval, not a test; with k clusters it has at most
             C(2k-1, k) distinct resamples).
  secondary  question-level paired bootstrap interval: anti-conservative, because questions within a
             conversation are answered against the same memory and are not independent.
  Holm       applied to the family of permutation p-values produced by one invocation (the reference
             against every other run on the command line, one metric). That is the command-line family,
             not the pre-registered four-hypothesis family; the printout says so.

  python -m paper2.stats paper2/out/<ref-run> paper2/out/<run-2> [paper2/out/<run-3> ...] [--metric hit] [--fig]
  python -m paper2.stats --selftest

The t quantile is computed numerically (regularised incomplete beta by continued fraction, then
bisection), not from scipy; `--selftest` checks it against the printed table for df 1..30 at 0.975 and the
permutation p against hand-countable cases.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np

FIG_DIR = Path(__file__).resolve().parent / "out" / "fig"
EXACT_MAX_K = 12          # enumerate all 2^k sign assignments up to this many clusters
MC_DRAWS = 20000          # Monte Carlo sign assignments above it
BOOT_REPS = 2000          # pre-registered resample count (00-preregistration.md section 10); A9 said pick one


# ----------------------------------------------------------------------------- loading and pairing

def load_qa(run: str, metric: str) -> dict[str, tuple[str, float]]:
    """qa id -> (conversation id, metric value) for every question with a finite metric value."""
    out = {}
    for l in open(Path(run) / "qa.jsonl"):
        r = json.loads(l)
        v = r.get(metric)
        if v is not None and v == v:
            out[r["qa"]] = (r["conv"], float(v))
    return out


def paired(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-question differences a - b over the questions present in both, and each question's conversation."""
    ids = sorted(set(a) & set(b))
    d = np.array([a[i][1] - b[i][1] for i in ids])
    g = np.array([a[i][0] for i in ids])
    return d, g


def _conv_key(c: str):
    m = re.search(r"(\d+)$", c)
    return (0, int(m.group(1)), c) if m else (1, 0, c)


def per_conversation(d: np.ndarray, g: np.ndarray) -> dict[str, float]:
    """Mean paired difference per conversation, keyed by conversation id (the primary statistic's inputs)."""
    return {c: float(d[g == c].mean()) for c in sorted(set(g), key=_conv_key)}


# ----------------------------------------------------------------------------- primary test: exact sign-flip permutation

def sign_flip_test(x, seed: int = 0, n_mc: int = MC_DRAWS, exact_max_k: int = EXACT_MAX_K) -> dict:
    """Two-sided sign-flip permutation test of H0: the paired differences are symmetric about 0.

    x: the k per-conversation mean differences. Statistic: their mean. For k <= exact_max_k every one of the
    2^k sign assignments is enumerated and p is the exact share with |mean| >= |observed| (the observed
    assignment and its mirror always count, so the smallest attainable p is 2 / 2^k). Above that, n_mc
    random assignments from a fixed seed, with the observed assignment added (p = (1 + hits) / (n_mc + 1)).
    Zero differences are kept (flipping them changes nothing), which is the correct exact treatment.
    """
    x = np.asarray(x, dtype=float)
    k = len(x)
    if k == 0:
        return {"p": float("nan"), "min_p": float("nan"), "exact": True, "k": 0, "n_assignments": 0, "stat": float("nan")}
    obs = float(x.mean())
    tol = 1e-12 * max(1.0, abs(obs))
    if k <= exact_max_k:
        n = 2 ** k
        signs = ((np.arange(n)[:, None] >> np.arange(k)) & 1) * 2 - 1        # n x k in {-1, +1}
        stats = (signs * x).mean(1)
        p = float(np.mean(np.abs(stats) >= abs(obs) - tol))
        exact = True
    else:
        n = n_mc
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1, 1]), size=(n, k))
        stats = (signs * x).mean(1)
        p = float((1 + np.sum(np.abs(stats) >= abs(obs) - tol)) / (n + 1))
        exact = False
    return {"p": min(1.0, p), "min_p": 2.0 / 2 ** k, "exact": exact, "k": k, "n_assignments": int(n), "stat": obs}


# ----------------------------------------------------------------------------- Student t without scipy

def _betacf(a: float, b: float, x: float, max_iter: int = 500, eps: float = 1e-15) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz; Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student t with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, df: float) -> float:
    """Inverse t CDF by bisection on t_cdf. Numerical routine, not a table; see --selftest for the check."""
    if df <= 0 or not (0.0 < p < 1.0):
        return float("nan")
    if p == 0.5:
        return 0.0
    target = max(p, 1.0 - p)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < target and hi < 1e12:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    q = 0.5 * (lo + hi)
    return q if p > 0.5 else -q


def t_interval(x, conf: float = 0.95) -> dict:
    """Mean of x with a conf-level t-interval on k - 1 degrees of freedom (NaN half-width for k < 2)."""
    x = np.asarray(x, dtype=float)
    k = len(x)
    m = float(x.mean()) if k else float("nan")
    if k < 2:
        return {"mean": m, "lo": float("nan"), "hi": float("nan"), "half": float("nan"), "df": max(k - 1, 0), "t": float("nan"), "se": float("nan")}
    se = float(x.std(ddof=1) / math.sqrt(k))
    t = t_quantile(0.5 + conf / 2.0, k - 1)
    return {"mean": m, "lo": m - t * se, "hi": m + t * se, "half": t * se, "df": k - 1, "t": t, "se": se}


# ----------------------------------------------------------------------------- bootstrap intervals (not tests)

def cluster_bootstrap(d: np.ndarray, g: np.ndarray, n: int = BOOT_REPS, seed: int = 0) -> dict:
    """Resample conversations with replacement; interval on the pooled mean difference. An interval, not a test.

    `sign_share` is the share of resampled means on the far side of zero (doubled, capped at 1): the quantity
    the earlier version printed as 'p'. It is kept for reference and is not a p-value (critique C, A9).
    """
    rng = np.random.default_rng(seed)
    convs = sorted(set(g), key=_conv_key)
    by = {c: d[g == c] for c in convs}
    means = []
    for _ in range(n):
        pick = rng.choice(convs, size=len(convs), replace=True)
        means.append(np.concatenate([by[c] for c in pick]).mean())
    m = np.array(means)
    k = len(convs)
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
            "clusters": k, "per_cluster": {c: float(by[c].mean()) for c in convs},
            "max_distinct": math.comb(2 * k - 1, k) if k else 0,
            "sign_share": float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))}


def question_bootstrap(d: np.ndarray, n: int = BOOT_REPS, seed: int = 0) -> dict:
    """Resample questions with replacement. Secondary, anti-conservative: questions within a conversation
    are answered against the same memory and are not independent. `sign_share` as in cluster_bootstrap."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return {"ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))], "n": int(len(d)),
            "sign_share": float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))}


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values for one family."""
    order = np.argsort(ps)
    adj = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------- one comparison

def compare(ref: dict, other: dict, reps: int = BOOT_REPS, seed: int = 0, n_mc: int = MC_DRAWS) -> dict:
    d, g = paired(ref, other)
    pc = per_conversation(d, g)
    x = np.array(list(pc.values()))
    return {"per_conv": pc, "n_questions": int(len(d)), "k": len(pc),
            "mean_conv": float(x.mean()) if len(x) else float("nan"),
            "t": t_interval(x), "perm": sign_flip_test(x, seed=seed, n_mc=n_mc),
            "cluster": cluster_bootstrap(d, g, reps, seed), "question": question_bootstrap(d, reps, seed),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "conv_pos": int((x > 0).sum()), "conv_neg": int((x < 0).sum())}


# ----------------------------------------------------------------------------- figure

def dot_plot(res: dict, ref_name: str, other_name: str, metric: str, path: Path) -> Path:
    """One dot per conversation (paired mean difference, reference - other), the mean with its 95% t-interval
    below, a zero line, and the permutation p in the subtitle. Written with the Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, series, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#fcfcfb"
    items = sorted(res["per_conv"].items(), key=lambda kv: kv[1])
    labels = [c for c, _ in items]
    vals = np.array([v for _, v in items])
    k = len(vals)
    ti, perm = res["t"], res["perm"]

    top_in, bottom_in = 1.15, 0.62                       # fixed bands in inches for the title block and the x-axis block
    fig_h = top_in + bottom_in + 0.34 * (k + 1)
    fig, ax = plt.subplots(figsize=(6.6, fig_h), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ys = np.arange(k) + 1.0          # conversations
    y_mean = 0.0                     # summary row below them
    ax.axvline(0, color=ink2, lw=1.0, zorder=1)
    ax.grid(axis="x", color=grid, lw=0.8, zorder=0)
    ax.scatter(vals, ys, s=56, color=series, edgecolor=surface, linewidth=1.5, zorder=3)
    if k >= 2 and np.isfinite(ti["half"]):
        ax.errorbar([ti["mean"]], [y_mean], xerr=[[ti["mean"] - ti["lo"]], [ti["hi"] - ti["mean"]]],
                    fmt="D", color=ink, ecolor=ink, elinewidth=1.4, capsize=3, markersize=6, zorder=3)
        mean_label = f"mean, 95% t-interval ({ti['df']} df)"
    else:
        ax.scatter([res["mean_conv"]], [y_mean], marker="D", s=40, color=ink, zorder=3)
        mean_label = "mean (k < 2: no interval)"
    ax.axhline(0.5, color=grid, lw=0.8, zorder=0)
    ax.set_yticks(np.concatenate([[y_mean], ys]))
    ax.set_yticklabels([mean_label] + labels, color=ink, fontsize=9)
    ax.set_ylim(-0.7, k + 0.7)
    ax.tick_params(axis="x", colors=ink2, labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    lim = max(np.abs(vals).max() if k else 0.0, abs(ti["lo"]) if np.isfinite(ti["lo"]) else 0.0,
              abs(ti["hi"]) if np.isfinite(ti["hi"]) else 0.0, 1e-3) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(f"paired difference in {metric} per conversation (reference − other)", color=ink2, fontsize=8.5)
    kind = "exact" if perm["exact"] else f"Monte Carlo, {perm['n_assignments']} draws"
    fig.text(0.02, 1 - 0.10 / fig_h, f"{ref_name} vs {other_name}: {metric}", ha="left", va="top", color=ink, fontsize=10.5, fontweight="bold")
    line1 = (f"k = {k} conversations, {res['n_questions']} paired questions. Sign-flip permutation p = {perm['p']:.3f} "
             f"({kind}; smallest attainable 2/2^k = {perm['min_p']:.3f}).")
    line2 = f"Mean {res['mean_conv']:+.3f}"
    if np.isfinite(ti["half"]):
        line2 += f", 95% t-interval [{ti['lo']:+.3f}, {ti['hi']:+.3f}] on {ti['df']} df."
    line3 = f"Difference = {ref_name} − {other_name}"
    line4 = f"> 0 favours {ref_name}."
    fig.text(0.02, 1 - 0.36 / fig_h, "\n".join([line1, line2, line3, line4]), ha="left", va="top", color=ink2, fontsize=8, linespacing=1.4)
    fig.subplots_adjust(top=1 - top_in / fig_h, bottom=bottom_in / fig_h, left=0.30, right=0.97)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- self-test

# Two-sided 0.975 t quantiles for df 1..30 (standard table, 3 decimals), used only to check t_quantile.
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def selftest() -> None:
    worst = max(abs(t_quantile(0.975, df) - ref) for df, ref in enumerate(_T975, start=1))
    assert worst < 6e-4, worst
    assert abs(t_quantile(0.975, 1e7) - 1.959964) < 1e-4
    assert abs(t_cdf(2.262, 9) - 0.975) < 1e-3
    # permutation: 5 same-sign clusters -> exactly 2/32; a 4-of-5 split cannot reach 0.05
    assert sign_flip_test([1, 2, 3, 4, 5])["p"] == 2 / 32
    assert sign_flip_test([1, 1, 1, 1, -1])["p"] > 0.05
    assert sign_flip_test([0, 0, 0])["p"] == 1.0
    r = sign_flip_test(np.linspace(1, 2, 15), n_mc=5000)
    assert not r["exact"] and r["p"] < 0.01 and r["n_assignments"] == 5000
    # exact two-sided p for 10 clusters: all positive -> 2/1024
    assert sign_flip_test(np.ones(10) + np.arange(10) * 0.01)["p"] == 2 / 1024
    # the t-interval covers the mean and has the tabulated width
    ti = t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(ti["half"] - 2.776 * math.sqrt(2.5 / 5)) < 1e-3
    # holm step-down: 0.01*3 = 0.03, 0.03*2 = 0.06, then 0.04*1 = 0.04 is raised to the running 0.06
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    print(f"selftest ok (t_quantile max abs error vs table df 1..30 at 0.975: {worst:.1e})")


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="reference run directory first, then the runs to compare it against")
    ap.add_argument("--metric", default="hit")
    ap.add_argument("--reps", type=int, default=BOOT_REPS, help="bootstrap resamples for the two intervals")
    ap.add_argument("--seed", type=int, default=0, help="seed for the bootstraps and for Monte Carlo sign flips above 12 clusters")
    ap.add_argument("--mc", type=int, default=MC_DRAWS, help="Monte Carlo sign assignments when k > 12")
    ap.add_argument("--fig", action="store_true", help="write out/fig/paired_<ref>_vs_<other>.png per comparison")
    ap.add_argument("--fig-dir", default=str(FIG_DIR))
    ap.add_argument("--json", default=None, help="also write every number to this JSON file")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return
    if len(a.runs) < 2:
        ap.error("need a reference run and at least one other run")
    ref_name = Path(a.runs[0]).name
    ref = load_qa(a.runs[0], a.metric)
    rows = []
    for other in a.runs[1:]:
        res = compare(ref, load_qa(other, a.metric), a.reps, a.seed, a.mc)
        res["other"] = Path(other).name
        if a.fig:
            res["fig"] = str(dot_plot(res, ref_name, res["other"], a.metric, Path(a.fig_dir) / f"paired_{ref_name}_vs_{res['other']}.png"))
        rows.append(res)
    adj = holm([r["perm"]["p"] for r in rows])
    for r, h in zip(rows, adj):
        r["perm"]["holm"] = h

    ks = sorted({r["k"] for r in rows})
    print(f"reference: {ref_name}   metric: {a.metric}   (positive = reference better)   unit = conversation")
    print(f"{'vs':>28} {'k':>3} {'mean':>7} {'t CI95 (k-1 df)':>17} {'p_perm':>7} {'holm':>6} {'cluster-boot CI95':>18} "
          f"{'question-boot CI95':>19} {'wins/losses':>12} conv>0")
    for r in rows:
        ti, pm, cb, qb = r["t"], r["perm"], r["cluster"], r["question"]
        tci = f"[{ti['lo']:+.3f},{ti['hi']:+.3f}]" if np.isfinite(ti["half"]) else f"{'n/a':>15}"
        print(f"{r['other']:>28} {r['k']:>3d} {r['mean_conv']:+7.3f} {tci:>17} {pm['p']:7.3f} {pm['holm']:6.3f} "
              f"[{cb['ci95'][0]:+.3f},{cb['ci95'][1]:+.3f}] "
              f"[{qb['ci95'][0]:+.3f},{qb['ci95'][1]:+.3f}]  {r['wins']:5d}/{r['losses']:<6d} {r['conv_pos']}/{r['k']}")
    print()
    print("per-conversation paired mean differences (the primary statistic's inputs):")
    for r in rows:
        print(f"  {r['other']:>28}  " + "  ".join(f"{c}:{v:+.3f}" for c, v in r["per_conv"].items()))
    print()
    exact = all(r["perm"]["exact"] for r in rows)
    print("primary: exact sign-flip permutation on the k per-conversation mean differences"
          + (f" (all 2^k assignments enumerated; two-sided)" if exact else f" (Monte Carlo, {a.mc} assignments, seed {a.seed}; two-sided)")
          + "; smallest attainable p = 2/2^k = " + ", ".join(f"{2 / 2 ** k:.4f} for k={k}" for k in ks)
          + ". The 95% t-interval has k-1 df (t quantile by numerical inversion, not scipy; --selftest checks it against the table).")
    print(f"Holm: applied to the family of {len(rows)} permutation p-values in this invocation ({ref_name} vs each other run, metric {a.metric}). "
          "This is the command-line family, not the pre-registered four-hypothesis family (critique C8); per-category and per-budget "
          "comparisons are exploratory and should be reported as intervals.")
    print("cluster-boot CI95: cluster (conversation) bootstrap of the pooled mean difference; an interval, not a test"
          + f" (at most {', '.join(str(math.comb(2 * k - 1, k)) + ' distinct resamples for k=' + str(k) for k in ks)}).")
    print("question-boot CI95: question-level paired bootstrap; secondary, anti-conservative (questions within a conversation are not independent).")
    print(f"mean = unweighted mean of the per-conversation differences; the pooled question-level mean is in the JSON output (--json) as cluster.mean. "
          f"bootstraps: {a.reps} resamples, seed {a.seed}.")
    if a.fig:
        print("figures: " + ", ".join(r["fig"] for r in rows))
    if a.json:
        Path(a.json).write_text(json.dumps({"reference": ref_name, "metric": a.metric, "comparisons": rows}, indent=2))


if __name__ == "__main__":
    main()
