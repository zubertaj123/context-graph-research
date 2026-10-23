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
     evidence dialog ids; they cover all evidence turns of 158/282 multi-hop, 270/321 temporal,
     50/96 open-domain, 653/841 single-hop and 359/446 adversarial questions) converted to triples
     by a deterministic rule (speaker as subject, lemmatised verb as predicate, object chunk).
     [Erratum, 18 Oct 2026: the frozen text named the categories in the wrong order (LoCoMo
     category 1 is multi-hop, 2 temporal, 3 open-domain, 4 single-hop); the counts were right.]
     Zero model calls happen at the gate in this arm; the candidates themselves are the
     benchmark's released session observations (generated by gpt-3.5-turbo per the official
     scripts; the frozen text and the first critique said GPT-4), produced with the whole session visible and
     covering 40% of turns and 76% of gold evidence turns, so "LLM-free" refers to the write-path
     decision, never to the pipeline. [Rephrased 18 Oct 2026; the frozen text said "no LLM in
     extraction at all", which was false.]
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

Public data only (LoCoMo under CC BY-NC 4.0 [corrected 18 Oct 2026; the frozen text said MIT],
LongMemEval-cleaned under MIT;
re-check). No Virtusa or Dell material. The measures are al-Saffar and Heileman's and are cited
as such; if Dr. al-Saffar is reachable, offer acknowledgement or co-authorship before submission.


## 14. Implementation log (decisions taken after the first admit-all pass, before any comparison)

Logged here so the comparisons in `docs/03-kickoff-results.md` can be read against what was
decided when. None of these decisions used gold labels or policy comparisons.

1. **Copula removed from the functional predicate set** (`paper2/graph.py`). The first admit-all
   pass showed every invalidation came from `is` ("is managing kids" vs "is going swimming"), i.e.
   false contradictions. Section 5's consolidation rule is unchanged; the predicate set is now the
   profile predicates only.
2. **Lemma and possessive fixes in the sentence rule** (`paper2/extract.py`): "values" was becoming
   "valu", "Caroline's dog" was becoming the predicate "caroline'". Now `value` and
   `(Caroline, has_dog, …)`.
3. **Gate tie rule and proportional control** (`paper2/policies.py`). The 2010 product is 0 on
   ~89% of candidates; with a plain `score >= quantile` rule the ties admitted everything, so the
   matched-size comparison was void. Ties at the cut are now admitted only while the realised rate
   is below the budget, and the quantile level is corrected by the gap between the budget and the
   realised rate. Realised admit rates are reported alongside every result.
4. **Plain SIVa weights** were multiplying by degree by mistake; fixed before any SIVa number was
   read (caught by `paper2/tests/test_paper2.py::test_measures_ranges`).
5. **Constant-feature guard and clipping in the logistic gate** (`paper2/policies.py`). Tier
   confidence is constant (0.9) in the oracle arm, so its standardised value was undefined; a
   constant column now gets weight 0, and standardised features are clipped to ±5 so a value outside
   the fit range (a new tier) cannot dominate the score.
6. **`--tier-weight`** (`paper2/run.py`): the pre-registered H4 multiplier, score × mean tier
   confidence, implemented as a switch applicable to any policy so that the effect of provenance
   weighting is measured separately from the effect of structure.
7. **Structural-only fit** (`fit --exclude text_len recency`): added after the first fit showed turn
   length as the top feature. The structural-only gate is the headline policy; the full gate is
   reported as "structural + surface".
8. **Budget band in the gate** (`paper2/policies.py`). A score that trends monotonically with graph
   size (the 2010 Impact: every new score is a new extreme) defeats any quantile rule, and the
   proportional correction saturated (realised 0.43 at q = 0.25, 0.61 at q = 0.75). When the realised
   rate leaves q ± 0.05 the gate now admits or rejects unconditionally until it is back inside. All
   comparisons in `docs/03-kickoff-results.md` were rerun with this gate from `paper2/sweep.sh`.
9. **Frozen copy.** The directory is not a git repository, so the promised `prereg-v1` tag does not
   exist. Sections 1–13 as they stood on 18 Oct 2026 are frozen verbatim in
   `docs/00-preregistration.frozen-2026-10-18.md` (sha256 `4d9abb17179ac73bf63267bd358fd9502f205d0e7f8f175b52fae303d58bb5e5`); everything after this line, and
   every later edit to sections 1–13, is a declared deviation.
10. **Category erratum and extraction phrasing** (section 6.1): see the bracketed notes in place.
    Every per-category number produced before 18 Oct 2026 17:00 UTC carried swapped names.
11. **Hypothesis labels.** The results note's "H1, first look" was a univariate AUC diagnostic, not
    the pre-registered H1 (non-inferiority to an LLM gate); its "H3" was the consolidation claim of
    section 11, not the pre-registered H3 (prior inversion). The note now uses the pre-registered
    labels only.
12. **Entity layer, consolidation vocabulary, size matching, statistics** rebuilt after the design
    critique (`docs/02-design-critique.md`): head-noun / NER objects with a closed predicate
    vocabulary (`paper2/extract.py`), type-insensitive speaker resolution and declared ER rules
    (`paper2/graph.py`), bisection on the budget to match live facts within 5%
    (`paper2/run.py --match-live`), leave-one-conversation-out cross-fitting with weights that record
    their fit conversations, exact sign-flip permutation on per-conversation paired means as the
    primary LoCoMo test (`paper2/stats.py`), and a graph-shape precondition table printed with every
    run. The feature vector is 13-dimensional (pre-registered: 8); the graph-structural subset is
    declared in `paper2/measures.py` (`STRUCTURAL`). The "prior" gate weights were hand-set before
    any fitting but are not in the frozen document.
13. **SIVa naming corrected to the 2010 definitions** (reader report `docs/reads/measure-lineage.md`,
    `related-2026.md`, both from the iiWAS 2010 PDF). In the paper, SIVa = (|Ec'| − |Ec|) / |Ec'| with
    Gc' = Gt ∪ Gc, an unweighted new-edge ratio under plain union, and the t·degree(v) edge weights
    belong to SIVp. Section 5 above called the unweighted union ratio "SIVa_plain, the degenerate
    control" and the weighted, consolidation-aware quantity "SIVa (2010 weights)": the labels were
    the wrong way round. From now on `siva_2010` is the 2010 definition, `siva_cons` is this paper's
    degree-weighted consolidation-aware redesign, `siva_scaled` its size-invariant companion. The
    Trust degeneracy on single-fact candidates and the speaker-hub dominance of Impact are intrinsic
    to the published formulas, not artefacts of this implementation (the extraction rule made them
    worse, critique B(iv)).
14. **Related-work position corrected** (`docs/reads/related-2026.md`): Fei et al. (2026) do not
    discuss overlap trust; their criterion is reallocatability of what determines top-k membership
    under sourceless structural writes. Section 2's claim that they "show agreement-trust is
    reallocatable" is withdrawn; the inference that sourceless writes to Gc can inflate later
    candidates' Trust is this paper's own and is what the `sourceless` poisoning condition tests.
    As of 18 Oct 2026 no paper found scores candidate triples by structural novelty or impact at
    admission, measures realised per-write graph change against per-question accuracy, or tests
    connected-vs-disconnected candidate priors; nearest work uses structure for retention/decay
    (Selective Forgetting, EngramRAG, STRATA, HeLa-Mem) or retrieval, and treats trust as
    source-typing (CPB, GovMem, Louck).
15. **Data facts corrected** (`docs/reads/data-locomo.md`): LoCoMo is licensed CC BY-NC 4.0, not MIT
    (section 13 corrected in place); its observations were generated by gpt-3.5-turbo with the
    CONVERSATION2FACTS prompt, not by GPT-4 (section 6.1 corrected in place). The non-commercial clause
    permits the research use and the redistribution of derived run outputs with attribution; the
    data files themselves are not redistributed with the code. Full data notes: `docs/07-data-notes.md`.
16. **Results regenerated on 18 Oct 2026** (`docs/03-results-loo.md`, from `bash paper2/sweep.sh`): all ten
    conversations as test units, gates fitted leave-one-out, memory matched to 25 / 50 / 75% of admit-all
    live facts within 5%, exact sign-flip permutation as the primary test. Outcome in one paragraph: the
    structural-only gate beats random, the 2010 product, Impact alone and the A-MAC-feature policies
    (Holm p ≤ 0.031, 8–10 of 10 conversations), matches the length + revisit heuristic (+0.027, Holm
    p = 0.146), and is not improved by surface features; the 2010 product is the worst policy at every
    budget (Trust = 0 on 77% of candidates); the memories are still speaker stars (0% non-hub-to-non-hub
    edges), so the wording "graph-structural" is replaced by "entity-centric admission features" for
    this arm; consolidation never fires on LoCoMo observations (0 invalidations), so H2, H3 and the
    section-11 claim move entirely to LongMemEval-S and the LLM-extraction arm; in the poisoning arm the
    tier multiplier cuts poison admission from 32–34% to 10% and raises evidence-hit by 6–10 points,
    while sourceless poison is admitted slightly more often than random.
17. **LongMemEval-S offline arm run on 20 Oct 2026** (`docs/03-results-loo.md` section 8): heuristic
    extractor, 500 questions, memory matched to 50%. The unfitted prior-weight gate equals random on
    evidence-hit (+0.008, p = 0.80) and leads only on MRR (+0.038, p = 0.057); the length + revisit
    heuristic is 8 to 9 points below random (p ≤ 0.001). Declared: the LoCoMo gate advantage over random
    does not transfer with hand-set weights; a gate fitted on LongMemEval-S by 5-fold cross-fitting over
    questions is the next pre-specified run, with structural features only.
18. **LongMemEval-S cross-fitted gate, 20 Oct 2026** (`docs/03-results-loo.md` section 8.3): structural
    features only, 5-fold by question, no LoCoMo weights; +0.073 evidence-hit over random (p = 0.003,
    Holm 0.005, 479 questions), +0.154 over the length heuristic, MRR +0.080. H1's "matched size" claim
    now has two datasets; the deviation to declare is that gate weights are fitted per dataset (the
    pre-registration said one fit on a held-out split).
19. **Full review pass of the draft, 22 Oct 2026.** Four independent reviewers (numbers, related work,
    LaTeX/structure, prose/consistency), each finding re-verified against the number files, the run
    outputs and the related-work notes before application; 106 edits applied. Corrections of substance:
    the trust ratio takes up to eleven values for |Et| ≤ 5 (not six); the 2010 product and Impact alone
    are the two worst policies at every budget (the product is not last at every budget); the poisoning
    arm injects at 10% per dialogue turn, so poison is about a quarter of the candidate stream (the
    text said "10% of candidate turns"); the LongMemEval-S sign-flip test is Monte Carlo (20,000
    assignments), not exact; the LongMemEval-S memory load is about 335 facts per question-memory
    against 262 per LoCoMo conversation, not "thirteen times"; the deviations table now lists log
    items 13–18; the bibliography is verified against arXiv, Crossref and DOI records.
