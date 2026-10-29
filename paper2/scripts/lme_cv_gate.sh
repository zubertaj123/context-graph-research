#!/bin/bash
# LongMemEval-S, structural-only gate fitted by 5-fold cross-fitting over questions (no LoCoMo weights):
# fold k is fitted on the admit-all turns of the other 400 questions, run at 50% matched memory on its 100.
set -e
cd "$(dirname "$0")/../.."
P=.venv/bin/python
Q=$($P -I -c "import json; d=json.load(open('data/raw/longmemeval/longmemeval_s_cleaned.json')); print(' '.join(str(x['question_id']) for x in d[:500]))")
mkdir -p paper2/out/w_lme
i=0; for q in $Q; do echo "$q" >> paper2/out/w_lme/fold_$((i % 5)).txt; i=$((i+1)); done
for k in 0 1 2 3 4; do
  HOLD=$(cat paper2/out/w_lme/fold_$k.txt | tr '\n' ' ')
  $P -m paper2.run fit --features paper2/out/lme_s_admit_all_full_500/turns.jsonl --exclude-conv $HOLD --structural --folds 5 --out paper2/out/w_lme/structural_fold$k.json | head -1
done
run_fold() { k=$1; HOLD=$(cat paper2/out/w_lme/fold_$k.txt | tr '\n' ' ');
  .venv/bin/python -m paper2.run run --dataset longmemeval --extractor heuristic --policy graph_gate --weights paper2/out/w_lme/structural_fold$k.json --offline --match-live 0.5 --limit 500 --conv $HOLD --name lme_s_graph_gate_struct_m0.5_fold$k > paper2/out/lme_s_gate_struct_fold$k.log 2>&1; echo "fold $k done"; }
export -f run_fold
printf '%s\n' 0 1 2 3 4 | xargs -P 3 -I{} bash -c 'run_fold {}'
$P - <<'PY'
import json, glob
from pathlib import Path
import sys; sys.path.insert(0,'.')
from paper2 import evaluate as ev
qa=[]; convs=[]
for d in sorted(glob.glob("paper2/out/lme_s_graph_gate_struct_m0.5_fold*")):
    qa += [json.loads(l) for l in open(Path(d)/"qa.jsonl")]; s=json.load(open(Path(d)/"summary.json")); convs += s["conversations"]
out=Path("paper2/out/lme_s_graph_gate_struct_m0.5_500"); out.mkdir(exist_ok=True)
(out/"qa.jsonl").write_text("".join(json.dumps(r)+"\n" for r in qa))
agg=ev.summarise(qa); agg.update({"name": out.name, "dataset": "longmemeval", "policy": "graph_gate_struct (5-fold by question)", "conversations": convs,
  "totals": {"live_facts": sum(c["live_facts"] for c in convs)}, "mean_size_gap": sum(c.get("size_gap",0) for c in convs)/len(convs)})
(out/"summary.json").write_text(json.dumps(agg, indent=2, default=str))
print(out.name, "n=", agg["n"], "hit=", round(agg["overall"]["hit"]["mean"],3), "mrr=", round(agg["overall"]["mrr"]["mean"],3), "live=", agg["totals"]["live_facts"])
PY
