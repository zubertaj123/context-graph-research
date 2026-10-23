#!/bin/bash
# Proxy validation (critique item 9): answer LoCoMo questions with a fixed model on admit-all and two gated
# memories at matched size 0.5 (structural gate, leave-one-out weights; random), all 10 conversations.
# Scores: official LoCoMo F1 / EM next to evidence-hit. One answer call per (policy, question); cached on disk.
set -e
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
P=.venv/bin/python
M="${ANSWER_MODEL:-anthropic:claude-haiku-5-5:low}"
PREFIX="${PREFIX:-online}"
CONVS=$($P -I -c "import json;print(' '.join(c['sample_id'] for c in json.load(open('data/raw/locomo/locomo10.json'))))")
run_conv() {  # $1 policy spec, $2 conv
  spec=$1; c=$2
  case $spec in
    graph_gate_struct) pol=graph_gate; w="--weights paper2/out/w_loo/structural_$c.json";;
    *)                 pol=$spec; w="";;
  esac
  $P -m paper2.run run --dataset locomo --policy $pol $w --match-live 0.5 --conv $c --answer-model "$M" --name ${PREFIX}_${spec}_m0.5_$c 2>&1 | grep -v DataWarning | tail -2
}
export -f run_conv; export P M PREFIX
# admit-all once over all conversations (one process), the two gated policies per conversation, 3 workers
( $P -m paper2.run run --dataset locomo --policy admit_all --answer-model "$M" --name ${PREFIX}_admit_all 2>&1 | grep -v DataWarning | tail -3 ) &
for spec in graph_gate_struct random; do
  printf '%s\n' $CONVS | xargs -P 2 -I{} bash -c "run_conv $spec {}"
done
wait
# merge the per-conversation gated runs
$P - "$PREFIX" <<'PY'
import json, glob, re, numpy as np, sys
PREFIX = sys.argv[1]
from pathlib import Path
sys.path.insert(0, '.')
from paper2 import evaluate as ev
groups = {}
for d in glob.glob(f"paper2/out/{PREFIX}_*_m0.5_conv-*"):
    m = re.match(rf"paper2/out/({PREFIX}_(.+)_m0\.5)_(conv-\d+)$", d)
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
                "conversations": summaries, "provenance": meta.get("provenance"), "merged_from": sorted(dirs),
                "totals": {k: float(np.nansum([c.get(k, 0) for c in summaries])) for k in ("candidates", "live_facts", "llm_calls", "llm_input_tokens", "llm_output_tokens")}})
    (out / "summary.json").write_text(json.dumps(agg, indent=2, default=str))
for p in [f"{PREFIX}_admit_all", f"merged_{PREFIX}_graph_gate_struct_m0.5", f"merged_{PREFIX}_random_m0.5"]:
    s = json.load(open(f"paper2/out/{p}/summary.json")); o = s["overall"]
    print(f"{p:40} n={s['n']} hit={o['hit']['mean']:.3f} f1={o['f1']['mean']:.3f} em={o['em']['mean']:.3f}")
PY
