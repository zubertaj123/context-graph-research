# Paper 2 pre-registration: structural information value as an admission and consolidation signal for agent memory

Status: draft v0.1, 18 Oct 2026. Freeze this document (git tag `prereg-v1`) before the first full
run; anything changed afterwards is reported as a deviation in the paper.

## 1. Working title and one-sentence claim

**Does structural change predict value? A first quantitative test of semantic information value
(trust x impact, graph-edit delta) as an LLM-free admission and consolidation signal for agent
memory graphs.**

Claim to be tested: graph-structural features of a candidate memory fact (agreement with what the
memory already holds, anchoring to known entities, degree-weighted impact, pattern novelty) and
the realized structural change a write causes carry measurable information about the fact's
downstream value for question answering, at zero gate-time LLM calls.

## 2. What is already known (the honest related-work position)

The measure is not new: Trust, Impact, SIVp and SIVa are al-Saffar and Heileman (DocEng 2008,
IEEE IS 2008, iiWAS 2010); the novelty term is Joslyn and al-Saffar's 2011 motif information gain
over the extant ontology; agreement-based trust is Knowledge-Based Trust (Dong et al. 2015); graph
priors for fact admission at web scale are Knowledge Vault (2014); informativeness of triples is
RELIN; the RDF delta is Berners-Lee and Connolly's Delta note. LLM-free arithmetic promotion on
LoCoMo exists (MemoryOS heat); LLM-light linear admission exists (A-MAC: confidence = max ROUGE-L,
novelty = 1 - max cosine, recency, type prior, one LLM utility call); LLM gates exist (Mem0
ADD/UPDATE/DELETE/NOOP; Graphiti; Nemori); learned gates exist (Memory-R1). GAM defines a graph
divergence as the consolidation trigger and declines to compute it as "expensive and sensitive to
lexical noise"; Kumiho lists degree-based entrenchment as future work. Fei et al. (2026) show
agreement- and degree-based selectors are reallocatable by sourceless writes.

What no verified work does (as of 15 Oct 2026; re-search before submission): admit or consolidate
agent-memory facts with a graph-informational measure; measure a write's realized change as a
weighted graph delta and relate it to QA value; test the 2010 prior that connected-to-known beats
disconnected-new on assistant memory; measure how much an agreement-based admission value is
reallocatable by poisoning and how much a provenance tier buys back.

Contribution framing, therefore: the first implementation and quantitative evaluation of SIV in
an agent-memory setting, with a degeneracy analysis of the 2010 formulation at fact granularity
and a redesign that separates agreement from anchoring; not a new measure.

## 3. Pre-registered hypotheses

- **H1 (non-inferiority).** An LLM-free logistic gate over graph features recovers at least a
  pre-set fraction of the QA benefit of a Mem0-style LLM gate at matched memory size. Margin:
  the graph gate's QA score is within 2.0 F1 points (LoCoMo) and 2.0 accuracy points
  (LongMemEval-S) of the LLM gate's, with the 95% paired-bootstrap interval of the difference
  excluding a loss larger than the margin. Memory size matched within 5%.
- **H2 (primary, novel).** Realized structural change SIVa of admitting a fact correlates
  positively with the per-question accuracy delta of the questions whose evidence the fact
  carries (Spearman rho, 95% bootstrap interval excluding 0), on a pre-sized sample of
  (fact, question) pairs with repeated answering (section 7).
- **H3 (prior inversion).** On single-session-user and information-extraction questions, facts
  whose candidate vertices are all new to memory (anchoring = 0) carry more QA value per fact
  than facts whose vertices are all known, i.e. the 2010 "disconnected new knowledge costs 1"
  prior is inverted for assistant memory. Reported per LongMemEval question type and LoCoMo
  category; H3 is confirmed if the inversion holds on the two single-session types.
- **H4 (reallocatability).** Injecting self-consistent sourceless false facts raises the
  admission rate of false facts under the agreement-based gate by a measurable amount; a
  provenance-tier multiplier recovers most of it. The marginal value of the structural term
  under poisoning is reported either way.

A null result on H2 is publishable: it would be the first measured evidence that structural
change does not predict value for conversational memory, against GAM's and the deck's assumption.

## 4. Memory model

Entity-centric graph on `cgraph.model.ContextGraph`: nodes are entities (speaker or user, person,
place, organisation, object or activity, date), edges are facts `(s, p, o)` with properties:
`source_turn`, `session`, `speaker`, `observed_at`, `tier` (verbatim-user, extractor-inferred,
llm-summary, poisoned), `confidence`, `method`, `valid_from`, `valid_to` (set by consolidation,
never deleted). One store only (no Neo4j mirror). RDF export via `cgraph.rdf` (reifier style)
for the SPARQL formulation in the paper.

Candidate unit `Gt`: the set of triples extracted from one turn (1 to 5 triples). The 2010 paper's
document-sized `Gt` corresponds to a *session* here; consolidation (section 6) runs per session.

## 5. Measures (all LLM-free, all linear time)

Let `Gc = (Vc, Ec)` be memory, `Gt = (Vt, Et)` the candidate, `H` the hub set (speakers, "user",
"assistant").

- Trust, 2010 formulation (ablation): `T_edge = |Et ∩ Ec| / |Et|`. Degenerate for |Et| in 1..3
  (takes 2 to 4 values; duplicates score 1, news scores 0). Reported as-is.
- Anchoring (redesign): `A = |(Vt \ H) ∩ Vc| / |Vt \ H|`, the share of non-hub candidate vertices
  already known.
- Soft agreement (redesign): `T_soft = mean over e in Et of max over e' in Ec with same subject of
  sim(e, e')`, with sim = token-set Jaccard over (p, o) in the offline arm and embedding cosine in
  the full arm. Separates "already known" from "about known things".
- Impact, 2010 formulation (ablation): `I = sum deg(v) for v in Vt ∩ Vc / sum deg(v) for v in Vc`.
  Hub-dominated.
- Impact, redesign: `I' = sum log(1 + deg(v)) for v in (Vt ∩ Vc) \ H / sum log(1 + deg(v)) for v
  in Vc \ H`, with a per-type variant.
- Novelty: `N = mean over (type(s), p, type(o)) in Gt of -log p_hat(pattern)` under the current
  extant ontology with add-one smoothing; `N_p = -log p(p)` for the predicate alone.
- SIVp (2010): `T_edge x I`. SIVp (redesign): logistic over `[T_edge, A, T_soft, I', N, tier,
  recency, |Et|]` fitted on a held-out split with 5-fold cross-validation.
- SIVa (realized change): define `Gc' = c(Gc ∪ Gt)` where the consolidation `c` (i) merges
  entities by exact and fuzzy match, (ii) invalidates a contradicted functional fact (same s,
  functional p, different o: old edge gets valid_to), (iii) closes a small inverse and symmetric
  predicate set. `SIVa = (w(Ec' \ Ec) + w(invalidated)) / w(Ec')` with 2010 weights `t x deg(v)`
  for edges attached to existing vertices and 1 for new components; `SIVa_plain` with unit
  weights and plain union is the degenerate control (a new-triple ratio).

## 6. Pipeline, held identical across policies

1. Extraction (fixed per dataset, not under test):
   - LoCoMo oracle arm: the dataset's own per-session observations (2,541 fact sentences with
     evidence dialog ids; they cover all evidence turns of 158/282 single-hop, 270/321 multi-hop,
     50/96 temporal, 653/841 open-domain and 359/446 adversarial questions) converted to triples
     by a deterministic rule (speaker as subject, lemmatised verb as predicate, object chunk).
     This arm has no LLM in extraction at all.
   - LLM arm (both datasets): one small instruct model, temperature 0, schema-constrained triples
     with a confidence; 3 seeds via sampling at temperature 0.3 for the seed-variance estimate.
2. Entity resolution: exact match on normalised surface form, then RapidFuzz token-sort ratio
   >= 90, then embedding cosine >= 0.85 (full arm). ER precision reported on a 200-mention sample;
   an oracle-ER arm on LoCoMo (observations name canonical speakers) bounds the noise.
3. Admission policy (the only thing that varies): see section 8.
4. Consolidation `c` per session (section 5).
5. Retrieval: identical for every policy: BM25 over verbalised admitted facts, top-k = 20, plus
   1-hop expansion over the admitted graph from the retrieved facts' entities, capped at 40 facts.
6. Answering: one fixed model, temperature 0, prompt fixed; the offline arm replaces answering by
   evidence recall (does the retrieved set contain a gold evidence turn's fact?).
7. Scoring: LoCoMo token F1 and exact match per category with the official script's
   normalisation (adversarial category scored by abstention); LongMemEval-S accuracy with the
   official judge prompt per question type, plus F1 as an LLM-free fallback; admission precision
   and recall against gold evidence turns (LoCoMo `evidence`, LongMemEval `has_answer` turns);
   memory size in facts and tokens; LLM calls and tokens per 1,000 candidates for the write path.

## 7. The H2 design (SIVa vs value), sized in advance

Sample 300 (fact, question) pairs per dataset stratified by SIVa tertile, where the fact's source
turn is a gold evidence turn of the question. For each pair answer the question k = 5 times with
the fact admitted and 5 times withheld (all else equal); value = mean accuracy delta. Spearman
rho between SIVa and value with a 2,000-resample bootstrap; also a logistic model of value on
SIVa controlling for fact length and category. Cost: 300 x 2 x 5 x 2 datasets = 6,000 answer
calls per model, affordable. Pre-registered success criterion: rho > 0 with the 95% interval
excluding 0 on at least one dataset and no significant negative on the other.

## 8. Policies and baselines

| Policy | LLM calls at gate | Notes |
|---|---|---|
| admit-all | 0 | upper bound on recall, the strong cheap baseline on LoCoMo |
| random at matched size | 0 | per-policy size matching; 5 draws |
| recency / heat (MemoryOS style) | 0 | heat = visits + interaction length + exp(-age); promotion threshold swept |
| SIVp-2010 threshold | 0 | the measure as published; threshold swept; cold-start rule for new components |
| graph-feature logistic gate (ours) | 0 | features of section 5; weights by 5-fold CV on a disjoint split |
| A-MAC-lite | 0 | confidence = max ROUGE-L, novelty = 1 - max cosine, recency, type prior; utility term dropped (LLM-free) |
| A-MAC | 1 | as above with the utility LLM call |
| Mem0-style LLM gate | 1 to 2 | the open-source ADD/UPDATE/DELETE/NOOP prompt, reproduced verbatim |
| learned gate (Memory-R1 style) | 0 at inference | optional, if training data and compute allow |

All compared at matched memory size where the policy has a threshold, and on the size-vs-score
curve otherwise.

## 9. Poisoning arm (H4)

Inject, per conversation, self-consistent false facts about existing hub-adjacent entities at
rates 0%, 2% and 10% of candidates, with the `poisoned` tier hidden from the gate in the
"sourceless" condition and visible as a low tier in the "tiered" condition. Report admission rate
of injected facts and QA harm (questions whose gold evidence is contradicted) per policy, and the
structural term's marginal value with and without the tier multiplier.

## 10. Statistics

Paired bootstrap over questions (1,986 LoCoMo, 500 LongMemEval-S), 2,000 resamples; Wilson
intervals for rates; three extraction seeds where extraction is stochastic; Holm correction over
the four hypotheses; effect sizes with intervals, no bare p-values. The non-inferiority margin
and the H2 sample size above are fixed before the first full run.

## 11. Deviations from the original proposal, forced by the reviewers

Dropped: the Neo4j mirror (no scientific gain); citing A-MAC's 0.583 (it is admission-label F1
with no QA); "fraction of the LLM calls" as a headline (extraction still needs an LLM; efficiency
is reported as write-path calls and tokens next to LightMem's accounting). Added: the oracle
extraction arm, the degeneracy analysis, the consolidation step that makes SIVa distinct from a
count ratio, the prior-inversion hypothesis, the tier-vs-structure ablation under poisoning,
MemoryOS and A-MAC-lite as LLM-free baselines.

## 12. Timeline to the first submission (ESWC 2027 workshop or poster, 18 Mar 2027)

| Week | Deliverable |
|---|---|
| 1-2 | loaders, oracle extraction, memory graph, measures, offline evidence-recall harness (this scaffold) |
| 3-4 | LLM extraction and answering adapters; LoCoMo end to end with admit-all and the LLM gate; cost check |
| 5-6 | all policies on LoCoMo; LongMemEval-S extraction (3 seeds); ER precision sample |
| 7-8 | H1 and H3 results with intervals; H2 pair sampling and repeated answering |
| 9 | poisoning arm (H4); ablations (2010 vs redesigned measures) |
| 10 | figures frozen; draft; related work from `docs/adversarial-verdicts.md` and the kickoff reads |
| 11 | blunt reader pass; code release tagged; Zenodo DOI |
| 12 | submit three days early |

## 13. Ethics, data and attribution

Public data only (LoCoMo, LongMemEval-cleaned, both MIT-licensed per their repositories;
re-check). No Virtusa or Dell material. The measures are al-Saffar and Heileman's and are cited
as such; if Dr. al-Saffar is reachable, offer acknowledgement or co-authorship before submission.
