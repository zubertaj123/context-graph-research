# Response to the design critique (`02-design-critique.md`)

Status per item of the critique's to-do list (section F), 18 Oct 2026. "done" means implemented and
covered by `paper2/tests` or by `sh paper2/sweep.sh`; "partial" means the code path exists but a
decision or a data run is still owed; "open" means not started.

| # | item | status | where |
|---|---|---|---|
| 1 | category mapping, adversarial by category, coverage erratum, "no LLM" phrasing, code/data/weights hashes in every summary, results note brought into line | done, verified (loader skeptic ok) | `data.py`, `run.py` (`provenance`), pre-registration §6.1 erratum and log items 10–12, results note banner |
| 2 | entity layer: head-noun / NER objects, lemmatiser, closed predicate vocabulary; type-insensitive speaker resolution; declared ER rules; anchoring undefined without non-hub vertices; graph-shape precondition table; ER precision sample | partial | `extract.py` (spaCy `en_core_web_sm` active; rule fallback via `PAPER2_EXTRACT_BACKEND=rule`), `graph.py` R1–R4 and `shape()`, `measures.anchoring` NaN; hand-check in `08-extraction-check.md` (8 ok / 4 weak / 6 wrong of 18 primary triples before the second repair; after it all 8 functional triples in the corpus are true; graph still has 0 non-hub–non-hub edges); **200-triple two-annotator sample still owed** |
| 3 | SIVa: size-invariant companion, |Ec| and position recorded for the H2 model; predicate normalisation so the functional rule can fire; inverse closure; consistent degree basis; §11 claim struck until supported; SIVa naming aligned with the 2010 definitions | partial | `measures.siva_scaled`, `siva_2010` (true 2010) vs `siva_cons` (redesign), `turns.jsonl` fields `n_live`, `position`; `graph.consolidate` inverse closure; after the second extractor repair the functional rule fires on 0 of 10 LoCoMo conversations (`08-extraction-check.md`), so there is nothing to hand-check here; **the consolidation term can only be tested on LongMemEval-S / the LLM arm** |
| 4 | leave-one-conversation-out cross-fitting; weights record fit conversations and runs refuse overlap; true structural-only fit; feature vector and prior weights declared as deviations | done, run (`03-results-loo.md`) | `sweep.sh`, `policies.cross_validate` (`fit_convs`), `run.py` overlap check, `fit --structural`, pre-registration log 12 |
| 5 | exact sign-flip permutation on per-conversation paired means + t-interval; dot plots; question bootstrap labelled secondary; four families; average-rank Spearman | done (code), verified; open (family definitions in the paper) | `stats.py` (skeptic ok), `analysis.py`, `figures.paired_dots` |
| 6 | size matching by bisection to ±5% live facts; warm-up without unconditional admission; admitted share by session decile | partial | `run.py --match-live` (realised size gaps 1–4%, `03-results-loo.md` §2), `OnlineQuantile` warm-up alternation; **session-decile admission profile not yet reported** |
| 7 | official LoCoMo scorer | done, verified | `evaluate.py` (vendored per-category rules, Porter stemming, two abstention substrings, MRR, Wilson intervals; skeptic verdict ok with two cosmetic residuals) |
| 8 | online-arm plumbing: LongMemEval-S default, local small model at temperature 0, seed / temperature / repeat in the cache key, token accounting per 1,000 candidates | done (code), verified | `data.py` variant (skeptic ok), `llm.py` (skeptic ok; residual: local spec parser accepts a temperature without a model), `run.py --lme-variant` |
| 9 | validate the proxy once against F1 | open (needs a model endpoint and a budget decision) | `run.py --answer-model` |
| 10 | baselines: A-MAC feature definitions with weights fitted by the same learner; honest labels; BM25 context for the LLM gate; 5 random seeds | partial | `policies.amac_features`, `amac_fit`, `DESCRIPTIONS`, `LLMGate`; **Mem0-verbatim prompt and the 5-seed random aggregation still owed** |
| 11 | poisoning arm: sourceless condition, real-vocabulary predicates, verbalised with predicate, self-consistent repeats, hub-adjacent subjects, paired injection, QA-harm metric | partial, run | `run.Poisoner`, results in `03-results-loo.md` §4; **QA harm needs the online arm** |
| 12 | rename drifted hypotheses; freeze the pre-registration | done | results note, `00-preregistration.frozen-2026-10-18.md` + sha256 in log item 9 |

Objections E1–E3 and what now answers them:

- **E1 (no structure to measure).** The precondition table is printed with every run
  (`shape_share_non_hub_deg_ge2`, `shape_share_edges_non_hub_to_non_hub`, components, max non-hub
  degree) and goes into the paper before any structural number. Whether the spaCy entity layer gives
  the graph enough shared vertices is an empirical question the regenerated sweep answers; if the
  share of non-hub vertices with degree ≥ 2 stays near zero, the wording falls back to "turn-level
  admission features over an entity-centric fact store" as the critique proposes.
- **E2 (SIVa is a clock; consolidation inert).** `siva_scaled` and the recorded `n_live` / `position`
  make the H2 model estimable with the clock as a covariate; the functional rule now has a predicate
  vocabulary to fire on. The section-11 claim stays struck until the LLM-extraction arm shows
  invalidations with checked precision.
- **E3 (inference, size, reproducibility).** Ten test clusters via leave-one-out, sign-flip
  permutation with its attainable minimum stated, bisection to ±5% live facts, hashes in every
  summary, and a results note regenerated only from files on disk.
