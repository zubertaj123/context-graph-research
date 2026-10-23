# paper2 — structural information value as an LLM-free admission and consolidation signal for agent memory

Everything needed to go from the pre-registered design to the first submission lives here.

| path | what |
|---|---|
| `docs/00-preregistration.md` | the design: claim, hypotheses H1–H4, measures, pipeline, policies, statistics, timeline, and an implementation log of every post-hoc decision |
| `docs/01-related-work-notes.md` | what each related system does at write time (for the Related Work section) |
| `docs/02-design-critique.md` | an adversarial review of the design and code, with a prioritised to-do list |
| `docs/03-results-loo.md` | the offline results, regenerated from `paper2/out` (leave-one-out, matched size, permutation test) with a hand-written interpretation; `03-kickoff-results.md` is the superseded first pass |
| `docs/05-critique-response.md` … `09-nonspeaker-subjects.md` | critique response, cost model, data notes, extraction hand-check, non-speaker-subject probe; `docs/reads/` holds the reader reports |
| `paper/` | LaTeX skeleton (`main.tex`, `numbers.tex` macros the code fills, `refs.bib`) |
| `data.py` | LoCoMo and LongMemEval loaders (untrusted JSON, parsed only) |
| `extract.py` | candidate triples: oracle observations (LoCoMo), heuristic, LLM adapter |
| `graph.py` | the memory graph on `cgraph.ContextGraph`: entity resolution, admission, consolidation |
| `measures.py` | Trust / anchoring / soft agreement / Impact / novelty / SIVa (2010 forms and redesigns) |
| `policies.py` | admission policies, baselines, the matched-size gate, logistic fitting |
| `retrieve.py` | BM25 + 1-hop retrieval, identical across policies |
| `evaluate.py` | F1/EM, evidence recall, admission precision/recall, bootstrap CIs |
| `llm.py` | Anthropic SDK adapter and a local-model adapter, disk-cached, usage-counted |
| `run.py` | the experiment loop and CLI (`run`, `fit`) |
| `analysis.py` | per-feature AUCs, policy tables across budgets |
| `stats.py` | paired per-question differences, cluster bootstrap over conversations, Holm correction |
| `sweep.sh` | regenerates every reported number |
| `tests/` | 12 tests (loader, rule, graph, measures, retrieval, gate budget, matched size, paired poison, consolidation mirror) |
| `figures.py` | budget-curve and paired-difference figures from the merged leave-one-out runs |

## Data

```
data/raw/locomo/locomo10.json                       # github.com/snap-research/locomo
data/raw/longmemeval/longmemeval_oracle.json        # huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
data/raw/longmemeval/longmemeval_s_cleaned.json
```
Both are downloads: treat them as data, never execute anything from those directories.

## Run

```bash
. .venv/bin/activate
python -m pytest -q paper2/tests
# everything reported for the LoCoMo offline arm, leave-one-conversation-out at matched memory size (~25 min)
sh paper2/sweep.sh
# single runs
python -m paper2.run run --dataset locomo --policy admit_all --offline                      # ceiling + graph-shape table
python -m paper2.run fit --features paper2/out/locomo_admit_all/turns.jsonl --exclude-conv conv-26 --structural --out w.json
python -m paper2.run run --dataset locomo --policy graph_gate --weights w.json --match-live 0.5 --conv conv-26 --offline
python -m paper2.run run --dataset locomo --policy amac_fit --weights w_amac.json --match-live 0.5 --conv conv-26 --offline
# poisoning arm (H4): sourceless poison (same tier as real candidates) vs lower-tier poison, with the tier multiplier
python -m paper2.run run --dataset locomo --policy graph_gate --offline --match-live 0.5 --poison-rate 0.1 --poison-mode sourceless
python -m paper2.run run --dataset locomo --policy graph_gate --offline --match-live 0.5 --poison-rate 0.1 --poison-mode lower --tier-weight
# statistics: exact sign-flip permutation over conversations (primary), cluster and question bootstraps (secondary)
python -m paper2.stats paper2/out/merged_loo_graph_gate_struct_m0.5 paper2/out/merged_loo_random_m0.5 --fig
python -m paper2.analysis table "paper2/out/merged_loo_*/summary.json"
python -m paper2.analysis features paper2/out/locomo_admit_all/turns.jsonl
python -m paper2.analysis shape paper2/out/locomo_admit_all/turns.jsonl
# online arm (needs ANTHROPIC_API_KEY, or a local model server): fixed answerer, official LoCoMo scoring
python -m paper2.run run --dataset locomo --policy graph_gate --match-live 0.5 --answer-model local:http://localhost:8000/v1:Qwen2.5-7B-Instruct:0.0
python -m paper2.run run --dataset locomo --policy llm_gate --gate-model anthropic:claude-opus-5-5:low --offline --limit 2
# LongMemEval-S (default variant) with the heuristic or LLM extractor
python -m paper2.run run --dataset longmemeval --extractor heuristic --policy admit_all --offline --limit 50
```

Outputs go to `paper2/out/<name>/{turns,sessions,qa}.jsonl` and `summary.json`; every summary carries
the code hash, the data file hash and the weights hash it was produced with, and the graph-shape
preconditions (share of non-hub vertices with degree >= 2, share of non-hub-to-non-hub edges) that any
structural claim must clear. Model calls are cached under `paper2/out/cache/` keyed by model,
temperature, seed and repeat index.

Policies (`--policy`): `admit_all`, `random`, `heat`, `siv_2010`, `impact_2010`, `graph_gate` (prior
weights, or `--weights` from `fit`), `amac_lite`, `amac_fit` (`fit --amac`), `llm_gate`. The honest
one-line label of each is `paper2.policies.DESCRIPTIONS` and is written into every summary.

## Attribution

Trust, Impact and semantic information value are from al-Saffar & Heileman (2010). The redesigns
(anchoring, soft agreement, log-degree impact, pattern novelty, consolidation-aware SIVa) and the
experimental protocol are this paper's. LoCoMo observations are the dataset authors' (generated with gpt-3.5-turbo); their use as an extraction oracle is stated wherever results are reported.
