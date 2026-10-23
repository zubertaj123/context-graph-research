#!/bin/bash
# H1 at the Mem0 gate's realised size: per conversation f_c = mem0 live facts / admit-all live facts; run the
# structural gate (LOO weights) and random at --match-live f_c, then answer all three memories online (pass-2
# verbalisation). Merges into merged_h1_{mem0_gate,graph_gate_struct,random}.
set -e
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
P=.venv/bin/python
M="anthropic:claude-haiku-5-5:low:nothink"
CONVS=$($P -I -c "import json;print(' '.join(c['sample_id'] for c in json.load(open('data/raw/locomo/locomo10.json'))))")
frac() { $P -c "
import json; m=json.load(open('paper2/out/mem0_gate_$1/summary.json'))['conversations'][0]['live_facts']
aa=[c for c in json.load(open('paper2/out/locomo_admit_all/summary.json'))['conversations'] if c['conv']=='$1'][0]['live_facts']
print(f'{m/aa:.4f}')"; }
run_one() {  # $1 conv
  c=$1; f=$(frac $c)
  $P -m paper2.run run --dataset locomo --policy mem0_gate --gate-model "$M" --threshold 0.5 --conv $c --answer-model "$M" --name h1_mem0_gate_$c 2>&1 | grep -v DataWarning | tail -1 | cut -c1-100
  $P -m paper2.run run --dataset locomo --policy graph_gate --weights paper2/out/w_loo/structural_$c.json --match-live $f --conv $c --answer-model "$M" --name h1_graph_gate_struct_$c 2>&1 | grep -v DataWarning | tail -1 | cut -c1-100
  $P -m paper2.run run --dataset locomo --policy random --match-live $f --conv $c --answer-model "$M" --name h1_random_$c 2>&1 | grep -v DataWarning | tail -1 | cut -c1-100
}
export -f run_one frac; export P M
printf '%s\n' $CONVS | xargs -P 3 -I{} bash -c 'run_one {}'
$P - <<'PY'
import json, glob, re, numpy as np, sys
from pathlib import Path
sys.path.insert(0, '.')
from paper2 import evaluate as ev
for pol in ("mem0_gate", "graph_gate_struct", "random"):
    dirs = sorted(glob.glob(f"paper2/out/h1_{pol}_conv-*"))
    out = Path(f"paper2/out/merged_h1_{pol}"); out.mkdir(exist_ok=True)
    qa, summaries = [], []
    for d in dirs:
        qa += [json.loads(l) for l in open(Path(d) / "qa.jsonl")]
        s = json.load(open(Path(d) / "summary.json")); summaries += s["conversations"]; meta = s
    (out / "qa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qa))
    agg = ev.summarise(qa)
    agg.update({"name": f"merged_h1_{pol}", "dataset": "locomo", "policy": meta["policy"], "policy_label": meta.get("policy_label"), "gate": meta["gate"], "conversations": summaries,
                "provenance": meta.get("provenance"), "merged_from": dirs,
                "totals": {k: float(np.nansum([c.get(k, 0) or 0 for c in summaries])) for k in ("candidates", "live_facts", "admitted_facts", "llm_calls", "llm_input_tokens", "llm_output_tokens")}})
    (out / "summary.json").write_text(json.dumps(agg, indent=2, default=str))
    o = agg["overall"]; print(f"{agg['name']:32} n={agg['n']} live={int(agg['totals']['live_facts'])} hit={o['hit']['mean']:.3f} f1={o['f1']['mean']:.3f} em={o['em']['mean']:.3f}")
PY
