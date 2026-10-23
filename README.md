# Does Structural Change Predict Value?

Semantic information value as an LLM-free admission signal for agent memory: a degeneracy analysis
and a matched-size test on LoCoMo and LongMemEval-S.

Mohammad Zuber Taj, independent researcher. Preprint; the arXiv identifier will be added here once
the submission is announced.

This repository is the public release behind the paper: the code, the frozen pre-registration with
its deviation log, the run artefacts and number files behind every table, and the LaTeX source. It is
a history-free snapshot exported from a private development repository; each release is one commit.

## What the paper does

Agent memory systems decide what to store with a language-model call per candidate fact. The paper
asks whether the structure of the memory graph carries that information, and gives the first
implementation and quantitative test, in an agent-memory setting, of al-Saffar and Heileman's 2010
semantic information value (trust times degree-weighted impact, and the realised graph-edit change of
a write). At fact granularity the 2010 formulation is degenerate; a redesigned, LLM-free gate over
seven graph features beats random, the 2010 measures and A-MAC's lexical features at matched memory
size on LoCoMo, and beats random on LongMemEval-S, but on LoCoMo no better than a turn-length
heuristic. The paper also validates the offline evidence-hit proxy against answers from a fixed
model, runs the verbatim Mem0 gate as the LLM baseline, and tests structural novelty under poisoning.

## Layout

| path | what |
|---|---|
| `paper2/paper/` | `main.tex`, the number files the code writes (`numbers.tex`, `numbers_online.tex`, `numbers_lme.tex`), `refs.bib`, the compiled PDF and the arXiv source archive (`arxiv-src.tar.gz`) |
| `paper2/docs/00-preregistration.frozen-2026-10-18.md` | the pre-registration as frozen before the first full run (SHA-256 quoted in the paper) |
| `paper2/docs/00-preregistration.md` | the living copy with the implementation log of every later decision |
| `paper2/docs/03-results-loo.md` | the offline results note, regenerated from `paper2/out` |
| `paper2/docs/` | related-work notes, design critique and response, cost model, data notes, extraction hand-check, the non-speaker-subject probe, and the reader reports in `reads/` |
| `paper2/*.py` | loaders, extractor, memory graph, measures, policies (including the verbatim Mem0 gate in `mem0_prompt.py`), retrieval, evaluation, statistics, the experiment CLI |
| `paper2/sweep.sh`, `paper2/scripts/` | the scripts that regenerate every reported number |
| `paper2/out/` | merged leave-one-out runs, fitted weights, figures and logs; per-conversation run directories and the model-response cache are not shipped |
| `cgraph/` | the small context-graph library the memory graph is built on (`cgraph.model.ContextGraph`) |
| `tests/`, `paper2/tests/` | smoke tests for the library and the experiment pipeline |

`paper2/README.md` has the full command list.

## Run

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && python -m spacy download en_core_web_sm
mkdir -p data/raw/locomo data/raw/longmemeval
# LoCoMo:       github.com/snap-research/locomo            -> data/raw/locomo/locomo10.json
# LongMemEval:  huggingface.co/datasets/xiaowu0162/longmemeval-cleaned -> data/raw/longmemeval/
python -m pytest -q tests paper2/tests
sh paper2/sweep.sh                       # offline LoCoMo arm, about 25 minutes, no model calls
```

The online arm (fixed answerer, Mem0 gate) needs `ANTHROPIC_API_KEY` in a `.env` file at the project
root; see `paper2/scripts/README.md`. Every run summary carries the code hash, the data-file hash and
the weight-file hash it was produced with.

## Licence and attribution

Code: MIT (see `LICENSE`). Paper, documents and run artefacts: CC BY 4.0, except that the `qa.jsonl`
files of the online and H1 runs quote LoCoMo gold answers and a model's answers to LoCoMo questions, and
those files remain under LoCoMo's licence (CC BY-NC 4.0). LoCoMo is used under its own licence
(CC BY-NC 4.0) and no dialogue text is redistributed; the Mem0 prompt in `paper2/mem0_prompt.py` is copied verbatim from mem0ai 1.0.11
(Apache-2.0) with attribution. Trust, impact and the two semantic information values are the work of
Sinan al-Saffar and Gregory Heileman; the redesigns and the experimental protocol are this paper's.

## Citation

```bibtex
@misc{taj2026structural,
  author = {Taj, Mohammad Zuber},
  title  = {Does Structural Change Predict Value? Semantic Information Value as an LLM-Free Admission
            Signal for Agent Memory: A Degeneracy Analysis and a Matched-Size Test on LoCoMo and LongMemEval-S},
  year   = {2026},
  note   = {Preprint; arXiv identifier to follow},
  url    = {https://github.com/zubertaj123/context-graph-research}
}
```
