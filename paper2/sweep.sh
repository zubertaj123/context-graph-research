#!/bin/bash
# Regenerates every number reported for the LoCoMo offline arm, leave-one-conversation-out (critique C1):
# for each of the 10 conversations the learned gates are fitted on the other 9, every policy is run on it
# at matched memory size (live facts = f x admit-all, within 5%), and the per-conversation results are
# merged so that all 10 conversations are test units. About 25 minutes on one CPU, no model calls.
#   bash paper2/sweep.sh            (set SIZES / POLICIES below to shorten)
set -e
cd "$(dirname "$0")/.."
P=.venv/bin/python
SIZES="${SIZES:-0.25 0.5 0.75}"
POLICIES="${POLICIES:-random heat siv_2010 impact_2010 amac_lite amac_fit graph_gate graph_gate_full graph_gate_struct}"
CONVS=$($P -I -c "import json;print(' '.join(c['sample_id'] for c in json.load(open('data/raw/locomo/locomo10.json'))))")
rm -rf paper2/out/loo_* paper2/out/w_loo_* paper2/out/poison_* paper2/out/locomo_admit_all paper2/out/merged_*
mkdir -p paper2/out/w_loo
$P -m paper2.run run --dataset locomo --policy admit_all --offline --name locomo_admit_all | tail -8

run_policy() {  # $1 policy spec, $2 conv, $3 size fraction, $4 extra args, $5 name prefix
  spec=$1; c=$2; f=$3; extra=$4; pre=$5
  case $spec in
    graph_gate_full)   pol=graph_gate; w="--weights paper2/out/w_loo/full_$c.json";;
    graph_gate_struct) pol=graph_gate; w="--weights paper2/out/w_loo/structural_$c.json";;
    amac_fit)          pol=amac_fit;   w="--weights paper2/out/w_loo/amac_$c.json";;
    *)                 pol=$spec; w="";;
  esac
  $P -m paper2.run run --dataset locomo --policy $pol $w --offline --match-live $f --conv $c $extra --name ${pre}_${spec}_m${f}_$c >/dev/null
}

one_conv() {  # all runs for one conversation; invoked in parallel below
  c=$1
  $P -m paper2.run fit --features paper2/out/locomo_admit_all/turns.jsonl --exclude-conv $c --out paper2/out/w_loo/full_$c.json | head -1
  $P -m paper2.run fit --features paper2/out/locomo_admit_all/turns.jsonl --exclude-conv $c --structural --out paper2/out/w_loo/structural_$c.json | head -1
  $P -m paper2.run fit --features paper2/out/locomo_admit_all/turns.jsonl --exclude-conv $c --amac --out paper2/out/w_loo/amac_$c.json | head -1
  for f in $SIZES; do for spec in $POLICIES; do run_policy $spec $c $f "" loo; done; done
  for mode in sourceless lower; do for tw in "" "--tier-weight"; do
    tag=$mode$( [ -n "$tw" ] && echo _tier )
    for spec in random graph_gate graph_gate_struct; do run_policy $spec $c 0.5 "--poison-rate 0.1 --poison-mode $mode $tw" poison_$tag; done
  done; done
  echo "done $c $(date -u +%T)"
}
export P SIZES POLICIES
export -f run_policy one_conv 2>/dev/null || true
JOBS="${JOBS:-3}"
if [ -n "$BASH_VERSION" ]; then
  printf '%s\n' $CONVS | xargs -P "$JOBS" -I{} bash -c 'one_conv {}'
else
  for c in $CONVS; do one_conv $c; done
fi
# merge per-conversation runs into one directory per (prefix, policy, size) so stats.py sees 10 clusters
$P - <<'PY'
import json, glob, re, numpy as np
from pathlib import Path
import sys; sys.path.insert(0, '.')
from paper2 import evaluate as ev
groups = {}
for d in glob.glob("paper2/out/loo_*_m*_conv-*") + glob.glob("paper2/out/poison_*_m*_conv-*"):
    m = re.match(r"paper2/out/((loo|poison_[a-z_]+)_(.+)_m([0-9.]+))_(conv-\d+)$", d)
    if m: groups.setdefault(m.group(1), []).append(d)
for name, dirs in sorted(groups.items()):
    out = Path(f"paper2/out/merged_{name}"); out.mkdir(exist_ok=True)
    qa, summaries = [], []
    for d in sorted(dirs):
        qa += [json.loads(l) for l in open(Path(d) / "qa.jsonl")]
        s = json.load(open(Path(d) / "summary.json")); summaries += s["conversations"]; meta = s
    (out / "qa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qa))
    agg = ev.summarise(qa)
    agg.update({"name": f"merged_{name}", "dataset": "locomo", "policy": meta["policy"], "policy_label": meta.get("policy_label"), "gate": meta["gate"],
                "conversations": summaries, "totals": {k: float(np.nansum([c.get(k, 0) for c in summaries])) for k in ("candidates", "admitted_facts", "live_facts", "invalidated", "poison_candidates", "poison_admitted", "llm_calls")},
                "mean_admit_rate": float(np.nanmean([c["admit_rate"] for c in summaries])), "mean_precision": float(np.nanmean([c["precision"] for c in summaries])),
                "mean_recall": float(np.nanmean([c["recall"] for c in summaries])), "mean_size_gap": float(np.nanmean([c.get("size_gap", np.nan) for c in summaries])),
                "shape": {k: float(np.nanmean([c[k] for c in summaries])) for k in summaries[0] if k.startswith("shape_")},
                "poison_retrieved_share": float(np.mean([q["poison_retrieved"] > 0 for q in qa])), "provenance": meta.get("provenance"), "merged_from": sorted(dirs)})
    (out / "summary.json").write_text(json.dumps(agg, indent=2, default=str))
    print(f"{agg['name']:>45} n={agg['n']} hit={agg['overall']['hit']['mean']:.3f} live={int(agg['totals']['live_facts'])} gap={agg['mean_size_gap']:.3f} poison={int(agg['totals']['poison_admitted'])}/{int(agg['totals']['poison_candidates'])}")
PY
echo; echo "== matched-size table (all 10 conversations, leave-one-out gates) =="; $P -m paper2.analysis table "paper2/out/merged_loo_*/summary.json"
echo; echo "== primary test: sign-flip permutation over 10 conversations, structural gate vs the rest at half size =="
$P -m paper2.stats paper2/out/merged_loo_graph_gate_struct_m0.5 paper2/out/merged_loo_random_m0.5 paper2/out/merged_loo_siv_2010_m0.5 paper2/out/merged_loo_impact_2010_m0.5 paper2/out/merged_loo_heat_m0.5 paper2/out/merged_loo_amac_lite_m0.5 paper2/out/merged_loo_amac_fit_m0.5 paper2/out/merged_loo_graph_gate_m0.5 paper2/out/merged_loo_graph_gate_full_m0.5
echo; echo "== prior gate vs the rest =="
$P -m paper2.stats paper2/out/merged_loo_graph_gate_m0.5 paper2/out/merged_loo_random_m0.5 paper2/out/merged_loo_heat_m0.5 paper2/out/merged_loo_amac_lite_m0.5 paper2/out/merged_loo_amac_fit_m0.5
echo; echo "== graph shape per conversation (admit-all) =="; $P -m paper2.analysis shape paper2/out/locomo_admit_all/turns.jsonl 2>/dev/null || true
echo; echo "== per-feature AUC, all 10 conversations =="; $P -m paper2.analysis features paper2/out/locomo_admit_all/turns.jsonl | tail -22
