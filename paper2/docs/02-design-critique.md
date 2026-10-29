# Design critique: paper2 pre-registration and implementation

Reviewer stance: hostile but fair. Written 18 Oct 2026 against the files as they stood at 16:52-16:59
UTC (`policies.py`, `run.py` 16:52; `00-preregistration.md` incl. the new section 14, 16:54;
`03-kickoff-results.md` 16:56; `out/` regenerated 16:58-16:59). The code moved under review three
times; where a finding depends on a version I say which. Every number below that is not quoted from a
file was recomputed in memory with the current code (`run_conversation` on all 10 LoCoMo conversations,
no files written). The test suite passes (9 tests, 0.8 s).

Summary verdict in five lines:

1. The "graph" in the oracle arm is two stars. Entities ≈ facts (conv-26: 185 non-hub entities, none
   with degree > 1, zero non-hub–non-hub edges). Every structural feature is therefore constant, a
   clock, or an entity-resolution artefact (section B(v), E1).
2. SIVa with 2010 weights is rank-identical to the plain new-triple ratio (Spearman 0.99) and is a
   clock (Spearman −0.91 with position in the conversation). Consolidation fired once in 2,555
   admissions. The pre-registered claim that consolidation "makes SIVa distinct from a count ratio"
   is false in this arm, and H2 as designed is confounded by position (A4, E2).
3. The LoCoMo category labels are swapped in the loader and in the pre-registration text; the saved
   results and the results note disagree with each other and with the code (A1, A10).
4. The statistics in `03-kickoff-results.md` (Holm p < 0.001 from a 5-conversation cluster bootstrap)
   are not attainable by any valid conversation-level test; the design has 5 (at best 10) independent
   units on LoCoMo (section C).
5. The baselines are mostly honest sketches with the wrong names; the LLM gate is not Mem0's, A-MAC-lite
   has the authors' weights and a constant recency term, heat is a turn-length score (section D).

Where something is fine it says "no issue".

---

## A. Does the implementation match the pre-registration?

### A1. Data and labels

- **A1.1 LoCoMo category names are swapped.** `data.py:33` maps `{1: single-hop, 2: multi-hop,
  3: temporal, 4: open-domain}`. The data says otherwise: every sampled category-2 question is
  "When did/was …" with a date answer (temporal); category-3 questions need world knowledge ("Which
  US state do Audrey and Andrew potentially live in?", answers with a `;`-separated rationale);
  category-1 questions aggregate 2-4 evidence turns ("What exercises has John done?"); category-4
  questions have one evidence turn. The official `task_eval/evaluation.py` confirms the shape:
  category 1 is scored with comma-split partial F1 (the multi-hop rule), category 3's gold is cut at
  the first `;`. Correct mapping: 1 multi-hop (282), 2 temporal (321), 3 open-domain (96), 4 single-hop
  (841), 5 adversarial (446). Consequences: the per-category table in `summary.json`, section 1 of
  `03-kickoff-results.md`, and the coverage sentence in the pre-registration (§6.1: "158/282 single-hop,
  270/321 multi-hop, 50/96 temporal, 653/841 open-domain") all carry the wrong names; the real
  numbers are 158/282 multi-hop, 270/321 temporal, 50/96 open-domain, 653/841 single-hop. The worst
  evidence-hit category (0.391, n = 92) is open-domain, not temporal. H3 is "reported per LoCoMo
  category": with this bug every H3 row would be mislabelled.
- **A1.2 Adversarial flag by key presence**, `data.py:130`: `adv = "adversarial_answer" in q and
  "answer" not in q`. Two of the 446 category-5 questions carry both keys and would be scored by
  token F1 against `answer`. Use `category == 5`.
- **A1.3 LongMemEval default is the oracle file**, `data.py:163` (`longmemeval_oracle.json`, 1.9
  sessions per question, no distractors). The pre-registration (§3 H1, §10) says LongMemEval-S.
  `longmemeval_s_cleaned.json` is on disk and the loader's field handling is compatible; the default
  must change or every "LongMemEval" number is on a haystack-free variant.
- **A1.4 One malformed evidence id** (`'D'`) in the raw data; harmless, but `evidence_turns` should
  drop ids that match no turn so `admission_pr` recall denominators are exact.
- No issue: session ordering, timestamp parsing (0 of 272 sessions without a date), observation
  parsing, `has_answer` handling for LongMemEval.

### A2. Scoring rule vs "the official script's normalisation" (pre-reg §6.7)

- **A2.1 Abstention list**, `evaluate.py:96-97`: eleven phrases. The official script checks exactly two
  substrings (`'no information available'`, `'not mentioned'`). The implementation is more lenient
  on the adversarial category; the paper cannot say "official" scoring.
- **A2.2 Normalisation**, `evaluate.py:100-104`: removes punctuation and `a|an|the`. The official
  `normalize_answer` also strips commas and the token `and`. Small, but it changes F1 on list answers.
- **A2.3 Multiple golds**, `evaluate.py:132-133`: splits gold on `;` and takes the max for every
  category. Official: category 1 splits prediction and gold on commas and averages per-part partial
  F1; category 3 keeps only the text before the first `;`; categories 2 and 4 use plain F1. These
  are different numbers. Vendor the official functions or reimplement them category by category.
- No issue: `exact_match`, `evidence_recall`, `admission_pr` (precision over admitted turns, recall
  and `recall_reachable` against gold, `evidence_coverage`).

### A3. Measures (pre-reg §5)

- **A3.1 T_edge, anchoring, T_soft, Impact (2010), Impact' (log-degree, hubs excluded), novelty with
  add-one smoothing, N_p**: formulas at `measures.py:110-178` match §5. No issue on the formulas.
  The problem is what they are computed over (B(v)).
- **A3.2 Anchoring returns 0 when the candidate has no non-hub vertex**, `measures.py:125-126`. A
  turn whose only triple is `(Caroline, mentions, Melanie)` is thereby counted as "all new", which
  pollutes the anchoring = 0 cell that H3 depends on. Return NaN and exclude.
- **A3.3 Feature vector is 12-dimensional, pre-registered as 8**, `measures.py:23-24` vs §5 "logistic
  over [T_edge, A, T_soft, I', N, tier, recency, |Et|]". Added without declaration: `impact` (2010),
  `novelty_pred`, `n_new_vertices`, `text_len`; `tier` is represented by `confidence`. `text_len` is
  the number of tokens in the raw dialogue turn (`measures.py:204`), not a property of the candidate
  or the graph, and it is the strongest univariate label correlate (Spearman +0.22; label rate by
  tertile 0.28 / 0.31 / 0.50). Pre-reg §14.7 now logs the structural-only fit as a post-hoc decision;
  that is a deviation and must be reported as one, not as "the headline policy".
- **A3.4 "Structural-only" still contains non-structural features.** `fit --exclude text_len recency`
  (`run.py:248`, `gate_weights_structural_*.json`) keeps `n_edges`, `n_new_vertices`, `confidence`.
  The largest weight in both structural fits is `n_edges` (−0.36 / −0.12), i.e. "how many GPT-4
  observations were written about this turn", followed by `n_new_vertices`. Neither is graph
  structure. The honest structural set is `{t_edge, anchoring, t_soft, impact, impact_log, novelty,
  novelty_pred}`; a third fit with exactly those is missing.
- **A3.5 The "pre-registered prior" weights are not in the pre-registration.** `policies.py:110-114`
  calls `PRIOR` "the pre-registered prior"; §5 and §8 contain no weights. The prior gate is the best
  structural policy in `03-kickoff-results.md` §3; its provenance must be stated as "hand-set before
  fitting, not in the frozen document". Note also that the prior applies to raw, unstandardised
  features (`mean = 0, std = 1` at `policies.py:118-119`), so with `novelty` ≈ 4 and `anchoring` almost
  always 0 the prior gate is effectively `0.3·novelty − 2·t_soft − 2·t_edge + 0.2·n_edges`: a
  novelty-minus-redundancy score. Say so.
- **A3.6 Recency is "hours since the subject was last mentioned"** (`measures.py:181-190`). The
  subject is a speaker in ~100% of oracle candidates, and `turn.at` is the session timestamp, so
  recency is 0 within a session and equals the session gap for the first candidate of a session per
  speaker (78% zeros). It is a "first fact of the session" indicator, not a recency signal. Fine as
  a feature; wrong as a description.
- No issue: `GraphView` single pass, `Features.vector` ordering guarded by the `names` assertion at
  `policies.py:124`.

### A4. SIVa and consolidation (pre-reg §5 last bullet, §6.4)

- **A4.1 SIVa_2010 ≡ SIVa_plain.** `measures.py:231` weights a new edge by `confidence ×
  max(deg(s), deg(o), 1)`. For the dominant candidate shape `(speaker, p, new object)` this is
  `0.9 × deg(speaker)`, and `mass_old` (`measures.py:217`, via `weight()` at `measures.py:64-68`) is
  the same quantity summed over live edges, which all touch a speaker. The hub degree cancels.
  Measured on all 10 conversations with the current code: Spearman(siva_2010, siva_plain) = 0.99.
  The pre-registration's "SIVa_plain is the degenerate control" therefore controls for nothing.
- **A4.2 SIVa is a clock.** Spearman(siva_2010, within-conversation position) = −0.91;
  siva_plain −0.92. Both are ≈ k/|Ec|. Any H2 correlation between SIVa and value is a correlation
  between position and value unless |Ec| at admission is controlled (B, C, E2).
- **A4.3 Consolidation does not fire.** `FUNCTIONAL` (`graph.py:27-28`) contains LLM-style
  predicates (`lives_in`, `works_at`, `has_spouse`…). The oracle rule emits bare verb lemmas
  (`extract.py:97`; conv-26 top predicates: `is` 25, `value` 11, `find` 9, `mentions` 8, `attend` 7).
  Intersection with `FUNCTIONAL` on conv-26: empty. Fresh run, all 10 conversations: 1 invalidation.
  The symmetric closure (`graph.py:108-111`) never fires either. So `c(Gc ∪ Gt) = Gc ∪ Gt`, which is
  exactly the degenerate case §11 says was fixed by "the consolidation step". `03-kickoff-results.md`
  §5 acknowledges this; the pre-registration text (§5, §11) does not yet carry the deviation.
- **A4.4 Inverse closure is pre-registered and absent**: §5 says "(iii) closes a small inverse and
  symmetric predicate set"; only `SYMMETRIC` exists (`graph.py:29`).
- **A4.5 Mixed degree bases in realised SIVa**, `measures.py:253-272`: invalidated mass uses
  before-session degrees (`snap["w"]`), new-edge mass and the denominator use after-session degrees.
  Immaterial while consolidation is inert; wrong once it is not.
- **A4.6 The "lemmatised verb" is a suffix stripper**, `extract.py:48-55`: `planning → plann`,
  `goes → goe`, `moved → mov` but `moves → move`, `running → runn`. The same verb yields different
  predicates by inflection, which (a) splits `(type, p, type)` patterns and inflates novelty, (b)
  guarantees that `FUNCTIONAL` and `PRIORITY_PREDICATES` (`policies.py:30-31`) match erratically.
  §6.1 says "lemmatised"; it is not.
- No issue: `consolidate()` keeps the newest fact live and sets `valid_to` on the older ones
  (`graph.py:101-107`); the per-session timing (`run.py:109`) matches §4.

### A5. Gate, fitting, size matching

- **A5.1 OnlineQuantile (current version, `policies.py:236-273`)**: the tie rule and proportional
  control fix the degenerate-score problem (siv_2010 now admits exactly 0.500 at q = 0.5). Not fixed:
  non-stationary scores. Realised admit rates at q = 0.5 across the 10 conversations with the
  current code: random 0.479, amac_lite 0.462, siv_2010 0.500, heat 0.512, graph_gate(prior) 0.544,
  impact_2010 0.551; live facts 1,167 to 1,448 (24% spread). Pre-reg §3 H1: "memory size matched
  within 5%". The same spread appears in `out/test_*_q50` (637 to 747 live facts). Nothing enforces
  the 5%.
- **A5.2 Warm-up admits the first 20 candidate turns unconditionally** (`policies.py:262-264`):
  8-13% of each conversation's candidates (152 to 307 candidate turns per conversation) are admitted
  by every policy regardless of score, all from session 1-2. Policies are therefore compared on the
  remaining 87-92% while being scored on all questions, and early-session evidence is favoured.
- **A5.3 Fitted weights are evaluated on a graph trajectory they were not trained on.** Training
  features come from an admit-all run (`run.py:239` help text; `fit_A_admit_all/turns.jsonl`). At test
  time the gate's features are computed on the graph *it* built (`run.py:85-86`), which is roughly
  half the size, so `novelty`, `impact`, `t_soft`, `siva_*` are shifted. Not addressed anywhere.
- **A5.4 `cross_validate` returns weights fit on all rows it is given** (`policies.py:328-330`).
  Disjointness of fit and test conversations is enforced only by the operator passing the right
  `turns.jsonl`. The current A/B split is disjoint (no issue), but `run` should refuse a weights file
  whose fit conversations intersect the run's conversations (store `groups` in the weights JSON).
- **A5.5 Class weighting and L2 are undeclared** (`policies.py:279-290`: inverse-frequency class
  weights, `l2 = 1e-2`, 2,000 full-batch steps, lr 0.05). Fine, but they belong in the paper's
  appendix; `fit_logistic` has no convergence check.
- **A5.6 "Weights by 5-fold CV on a disjoint split" (§8) vs what was done**: grouped 5-fold inside a
  5-conversation fit split, then one model on all five. The CV AUC (0.58-0.67) is reported from
  folds of 1 conversation each. Fine as a diagnostic; it is not the selection procedure §8 implies
  (no hyper-parameter was selected by it).
- No issue: grouped folds by conversation (`policies.py:316-318`); gate reset per conversation
  (`run.py:67`); duplicate suppression applied identically to every policy (`run.py:97-100`).

### A6. Pipeline (pre-reg §6)

- **A6.1 Entity resolution has no embedding stage** (§6.2 "then embedding cosine ≥ 0.85 (full arm)");
  `graph.py:47-58` is exact + RapidFuzz only. No ER precision sample, no oracle-ER arm (both promised
  in §6.2).
- **A6.2 Type-gated resolution creates shadow speaker nodes.** `extract.py:128` types every object
  with `entity_type()` (`extract.py:58-63`: capitalised → `NamedEntity`); hubs are created as
  `Person` (`graph.py:44`); `resolve()` requires equal types (`graph.py:53`). So `(Caroline, mentions,
  Melanie)` creates a second "Melanie" node of type `NamedEntity`. Measured: 20 shadow speaker nodes
  (2 per conversation); 269 of the 270 turns with 0 < anchoring < 1, 271 of 284 turns with
  `t_edge > 0`, and 271 of 289 turns with `impact_log > 0` contain a `mentions` triple. The
  non-degenerate part of all three "structural" signals is this artefact (B(v), E1).
- **A6.3 Retrieval is over live facts only**, `retrieve.py:60`; `run.py:117` never passes the
  question time. §4 says `valid_to` is "set by consolidation, never deleted" — true in storage, false
  for retrieval. Once consolidation fires (LLM arm, knowledge-update questions), the superseded value
  that temporal questions ask for becomes unretrievable. LongMemEval `QA.at` exists and is unused.
- **A6.4 1-hop expansion is inert**: it added anything on 11.5% of questions, at most 4 facts (cap is
  40; mean retrieved 19.6). Because objects never recur, there is nothing to expand through except
  the shadow speaker nodes. The paper must not attribute any retrieval effect to "the graph".
- **A6.5 Answering is not "one fixed model, temperature 0"** (§6.6): `llm.py:65-78` defaults to
  `claude-opus-5-5` with adaptive thinking and no temperature; `make_client` (`llm.py:103-114`) never
  passes a temperature to `OpenAICompatClient`. The pre-registered "one small instruct model" has no
  CLI path that sets temperature 0.
- **A6.6 Seeds collide in the cache**: §6.1 "3 seeds via sampling at temperature 0.3". The cache
  namespace is `local:{model}` (`llm.py:88`) and the hash (`llm.py:38`) contains no seed or
  temperature, so three "seeds" return one cached answer. Same for H2's "k = 5 answers": identical
  prompt and `cache_key` (`run.py:121`) → one call, five copies.
- **A6.7 Observations are not write-time candidates.** LoCoMo observations are per-session summaries
  written with the whole session visible ("Melanie is going swimming with the kids after the
  conversation"). Streaming them turn by turn leaks within-session look-ahead into the "online" write
  path. Must be stated (B(i)).
- **A6.8 Tier → confidence mapping is inconsistent**: `TIER_CONFIDENCE["extractor-inferred"] = 0.8`
  (`run.py:39`) but the oracle extractor sets 0.9 (`extract.py:115`), poison 0.6.
- No issue: BM25 top-20 + 1-hop + cap 40 (`retrieve.py:50, 69-82`) as in §6.5; per-policy identical
  retrieval; candidate unit per turn (`extract.py:123-131`; |Et| reaches 6 once, §4 says 1-5).

### A7. Baselines (pre-reg §8) — details in section D

- Mem0-style gate "reproduced verbatim": not reproduced (`policies.py:155-159` is a custom
  store/skip prompt; context = the 10 *oldest* same-subject facts, `policies.py:173`).
- A-MAC (with utility call) and Memory-R1: absent. A-MAC-lite: authors' weights, constant recency.
- "SIVp-2010 … cold-start rule for new components": no cold-start rule (`policies.py:94-99`).
- "random at matched size; 5 draws": one seed per run; no aggregation (`run.py:157`).
- `impact_2010` as a standalone policy is not in §8 (fine as an ablation; label it so).

### A8. Poisoning arm (pre-reg §9)

- **A8.1 Not self-consistent, not hub-adjacent**: `Poisoner.maybe` (`run.py:50-57`) picks a speaker
  as subject (the hub itself), a random corpus object, and a predicate from `FUNCTIONAL` that no real
  candidate ever uses. Each injected fact appears once; nothing "self-consistent" is built.
- **A8.2 Poison is separable by vocabulary**: `novelty_pred` is maximal for a never-seen predicate,
  and the verbalisation is `"{subj} {obj}"` (`run.py:57`) with no predicate, which changes BM25
  behaviour. `03-kickoff-results.md` §4's conclusion ("a fabricated fact is exactly what the
  structural gate prefers") is therefore an artefact of the injection, not a property of structure.
- **A8.3 There is no sourceless condition.** Poison confidence is 0.6 vs 0.9 for every real oracle
  candidate (`run.py:39, 56`), and `confidence` is a feature (`measures.py:24`). Under the fitted gate
  the constant guard zeroes the weight (`policies.py:304-308`) so the feature is inert; under
  `--tier-weight` (`run.py:90-91`) it is a perfect separator. §9 requires a condition where the gate
  cannot see the tier at all: set poison confidence equal to the real tier, and inject predicates
  drawn from the real vocabulary.
- **A8.4 "QA harm (questions whose gold evidence is contradicted)"** is not computed; only
  `poison_retrieved` per question (`run.py:118-119`).

### A9. Statistics (pre-reg §10)

- `evaluate.py:153`: 1,000 resamples (pre-registered 2,000); `stats.py` uses 5,000. Trivial; pick one.
- No Wilson intervals anywhere for rates (`admission_pr`, poison admission).
- `stats.py:103` reports a bootstrap sign-proportion as a "p"; this is not a test (section C).
- `analysis.py:23-27` Spearman uses raw argsort ranks without tie handling; wrong for binary labels
  and for `t_edge`/`anchoring` (88% ties).

### A10. Provenance of reported numbers

- `out/locomo_admit_all/summary.json` was first written at 16:44 by code that still had `is` in
  `FUNCTIONAL` (441 invalidations, evidence-hit 0.568); the current code gives 1 invalidation and
  0.648. It was regenerated at 16:58. The 16:56 version of `03-kickoff-results.md` §3 quoted a run
  that was no longer on disk ("full (fitted) 0.484, live 671" against `test_graph_gate_fit_q50` at
  0.453 / 659; heat 0.426/717 vs 0.429/714; amac_lite 0.414/613 vs 0.422/637); the 17:01 version
  matches `out/`. Headline numbers moved twice within one hour with no record of which code produced
  which. Every `summary.json` needs the code hash and the weights-file hash; every number in a doc
  needs the run directory it came from; the paper's tables must be generated from `out/` by one
  script, never typed.
- The pre-registration (§3) says "freeze … git tag `prereg-v1`"; the directory is not a git repository,
  so there is no frozen version. Section 14 (implementation log) was appended to the same file after
  the first runs; a reviewer cannot tell what was frozen.

### A11. Hypothesis drift between the pre-registration and the results note

- Pre-reg H1 = non-inferiority to a Mem0-style LLM gate within 2 F1 points at matched size. The
  results note's "H1, first look" is univariate AUC against the evidence label. Different claim.
- Pre-reg H3 = prior inversion (anchoring = 0 vs 1). The results note's "H3" (§5, §7.5) is
  "consolidation-aware SIVa beats the plain ratio". Different hypothesis under the same label. Rename
  one before anything is written up.
- No issue: H2 and H4 are still the same statements.

---

## B. Validity threats specific to this design

### B(i). The oracle arm uses GPT-4 observations: what happens to "LLM-free"

- Every candidate in the oracle arm is a GPT-4 sentence; `n_edges` (a top-weighted feature in both
  structural fits) is literally "how many GPT-4 observations were written for this turn"; `text_len`
  is the dialogue turn's length; `t_soft` and `novelty` are computed over GPT-4's paraphrases. The
  gate makes zero calls, the pipeline that produces its inputs made one call per session with the
  whole session visible. The extractor also *pre-selects*: among the 2,375 turns that have an
  observation the evidence rate is 0.451; among the 3,507 that do not it is 0.101. Most of the
  "admission" has happened before any policy runs.
- The observation→turn link (`extract.py:127-128`, `source_turns = ob.turn_ids`) is GPT-4's own
  evidence attribution, and evidence-hit is scored on `source_turns` (`evaluate.py:137`). The hit
  criterion and the candidates' provenance come from the same annotation; a real extractor would set
  `source_turns = [turn.id]`, which is also fine, but the oracle arm's hit rate is optimistic relative
  to any real extractor and this must be said.
- Required phrasing: "zero LLM calls at the gate, given an extractor; in the oracle arm the extractor
  is the benchmark's released GPT-4 session observations, which were produced with the whole session
  visible and which cover 76% of gold evidence turns and 40% of all turns; `n_edges` and the
  candidate text are LLM outputs". Never "LLM-free pipeline", never "no LLM in extraction" (§6.1 says
  "This arm has no LLM in extraction at all": false, the LLM ran offline before the benchmark shipped).

### B(ii). The turn-level evidence label used to fit the gate

- **Direct leakage into the policy**: none. `Policy.score` receives `(G, turn, triples, feats)`
  (`run.py:89`); `evidence` is used only for `label` and `admission_pr`. No issue.
- **Experiment-level leakage**: present if the weights' fit conversations overlap the run's
  conversations; currently avoided by hand (A5.4). Enforce it in code.
- **Is "turn is a gold evidence turn for some question" a legitimate proxy for value?** Partially,
  and with known biases: (a) it measures annotator attention, which correlates with turn length
  (0.28 → 0.50 label rate across `text_len` tertiles) and specificity, so a gate fit to it learns
  "long, specific turns", which is why `text_len` dominates the full fit; (b) it is binary per turn
  while value is per (fact, question) and conditional on what else is in memory — a multi-hop
  question's evidence turn has value only jointly; (c) adversarial questions' evidence turns are
  labelled positive, but admitting them has zero or negative value (the answer is abstention). This
  last effect is small here: only 2 of 1,076 positives are *exclusively* adversarial evidence. (d) It
  is dataset-specific: there is no such label at deployment, so the gate cannot be refit in the wild.
- **What the label makes the "structural" coefficients mean**: `t_edge > 0` has label rate 0.229
  vs 0.483, and anchoring is negatively associated (Spearman −0.16). The results note reads this as
  "redundancy with memory predicts low value" (§7.3). It does not: 271/284 `t_edge > 0` turns and
  269/270 partially anchored turns are turns containing `(speaker, mentions, other speaker)`, i.e.
  "turns that talk about the other speaker are less question-worthy". Nothing about agreement or
  anchoring has been measured yet.
- Recommendation: keep the label for a *diagnostic* AUC table only; fit nothing to it that is then
  called "the structural gate". If a learned gate is wanted, fit it to a retrieval-based target on
  the fit split (e.g. "fact was retrieved for a question it is evidence for") and say so.

### B(iii). The online-quantile gate as a size-matching device

- **Warm-up** (`policies.py:262-264`): 20 unconditional admissions per conversation, 8-13% of
  candidates, all early-session. Biases every policy toward session-1 facts and makes the realised
  rate q' = (20 + q·(n−20))/n ≠ q; the control then has to over-reject later to compensate, which
  rejects late-session facts of *every* policy. Fix: warm-up on a score buffer without admitting
  (admit by the fixed prior for the first 20), or report results excluding the warm-up turns.
- **Non-stationarity**: `novelty` (Spearman +0.40 with position), `t_soft` (+0.31), `siva_*`
  (−0.91) drift with graph size. A running quantile over the history systematically mis-sizes drifting
  scores; the proportional control (`policies.py:265-266`) corrects the rate but not *which* turns are
  admitted: for an increasing score the gate admits late turns, for a decreasing one early turns, at
  the same realised rate. Policies are therefore compared with different temporal admission profiles,
  which matters because evidence is not uniform in time. Report admitted-share by session decile per
  policy, or normalise each score by its running mean/std before the quantile.
- **Ties**: fixed for exact ties (`policies.py:268`), but `siv_2010` is 0 on 88% of candidates, so
  at q ≤ 0.88 the "policy" is first-come-first-served among the zeros, i.e. a recency policy wearing
  the 2010 label. Say so when reporting `siv_2010`.
- **Size is not matched**: 24% spread in live facts at q = 0.5 (A5.1). The pre-registration says 5%.
  Either bisection on q per policy to hit a target live-fact count, or compare on the size-vs-hit
  curve by interpolation at a common size. The `budget_curve` figure is the right object; the q = 0.5
  table is not.
- **Within-conversation reset** (`run.py:67`): the gate forgets across conversations, which is right
  for LoCoMo (independent people) and wrong for LongMemEval (one conversation per question, 50+
  sessions, the first 20 candidates of every haystack admitted).

### B(iv). Consolidation rule and the deterministic sentence-to-triple rule: where they bias SIVa and Trust

- **Functional invalidation with verb-lemma predicates** (when it did fire, pre-16:46 code): "is
  managing kids" vs "is going swimming" counted as contradictions; 441 invalidations, 17% of
  admissions, nearly all false. SIVa's `inval_mass` term then rewarded candidates that *re-mention* a
  frequent predicate, and retrieval lost the superseded facts. The current fix (drop `is`) makes the
  rule inert instead of correct. Either way the SIVa consolidation term is untested.
- **Object chunks as entities** (`extract.py:98`: up to 6 content tokens): "LGBTQ support group
  recently found" is one entity. Consequences: Trust (T_edge) requires an identical 6-token chunk →
  88% zeros; T_soft's Jaccard is over bags that contain adverbs and verbs; Impact' is zero because no
  non-hub vertex is shared; SIVa's degree weighting cancels (A4.1). The degeneracy the paper will
  report as a property of the 2010 measures "at fact granularity" is, in this arm, largely a property
  of this extraction rule. The degeneracy analysis must separate the two (recompute T_edge with head
  nouns / NER objects and with soft matching; then report what survives).
- **Possessive rule** (`extract.py:80-81, 85-87, 97`): "Melanie's kids are excited" → `(Melanie,
  has_kids, "excited about summer break")`. The predicate absorbs the noun, the object is the
  predicate's complement. Novelty then counts `has_kids` as a new pattern every time the noun differs.
- **Predicate = first non-stop token after the subject** (`extract.py:89-94`): for "Caroline is
  planning to continue her education" the predicate is `is` (25 of 192 conv-26 triples) with object
  "planning continue education explore career". `is`-triples are excluded from `FUNCTIONAL` by hand,
  so 13% of all facts can never be consolidated whatever happens.
- **Bias on Trust direction**: duplicates are almost exclusively `mentions` edges (64 of 67 in the
  first three conversations), so T_edge = 1 means "re-mentions the other speaker", and the fitted
  negative weight on `t_edge` is not a finding about agreement (B(ii)).

### B(v). Entity resolution via RapidFuzz token-sort ≥ 90

- **Wrong merges** (checked values): `8 may 2023` / `9 may 2023` = 90 (merged: wrong, dates), `march
  2023` / `march 2024` = 90 (merged: wrong), `paris` / `parish` = 91 (merged), `daughter` /
  `daughters` = 94 (fine). Dates one day or one year apart collapse into one node; for the temporal
  category (321 questions, the second largest) this is a direct error source.
- **Failed merges**: `lgbtq support group recently` / `support group` = 63, `puppy max` / `max` = 50,
  `counseling` / `counseling job` = 83, `boston` / `boston ma` = 80, `the beach` / `beach` = 71,
  `new york` / `new yorker` = 89 (correctly not merged, but by one point). With 6-token object chunks
  essentially nothing merges: conv-26 has 185 non-hub entities for 185 facts, none with degree > 1.
- **Type gating defeats hub resolution** (A6.2): speakers named as objects become `NamedEntity`
  shadow nodes; 2 per conversation; they are the only non-hub vertices that recur, and they generate
  the only non-zero anchoring / Impact' / T_edge values. The test at `test_paper2.py:159-167` passes
  only because it never sets `o_type`, so both "Boston" and "boston" default to `Thing`; the real
  pipeline would type them `NamedEntity` and `Thing` and never merge them.
- **Length filter** `abs(len(n) − len(key)) > 12` (`graph.py:53`) blocks "Max" vs "puppy named Max"
  type merges regardless of ratio. Fine as a speed hack; it is an undeclared ER rule.
- Net effect on the hypotheses: H3 compares 2,086 anchoring = 0 turns against 19 anchoring = 1 turns
  (17 of which have no `mentions` triple). H3 cannot be tested in this arm; §6.2's "ER precision on a
  200-mention sample" is the minimum before any structural number is interpreted.

### B(vi). Evidence recall as the offline proxy for QA quality

- **Ceiling and category dependence**: only 76% of evidence turns have an observation
  (`evidence_coverage`), and only 158/282 multi-hop, 50/96 open-domain questions have *all* their
  evidence covered. Admit-all hit (0.648) is the ceiling; per-category ceilings differ, so
  per-category comparisons across policies are comparisons of fractions of different ceilings.
- **Hit ≠ answerable**: the retrieved fact is a compressed GPT-4 sentence; dates are routinely
  relative ("last Saturday", "two weeks ago") while temporal gold answers are absolute ("24 June,
  2023"), so a temporal question can hit and still score F1 = 0. Multi-hop needs all parts
  (`frac`), not any (`hit`); `frac` for the 282 multi-hop questions is 0.329 while `hit` is 0.621.
- **Adversarial questions invert the sign**: hit is 0.655 on adversarial vs 0.646 on the rest, and a
  retrieved near-evidence fact is what lures an answerer away from abstention. Evidence recall must
  be reported excluding category 5, and the abstention metric exists only in the online arm.
- **Set size and ordering are ignored**: 40 facts with the evidence at rank 38 scores the same as 5
  facts with it at rank 1; F1 with a fixed answerer will not. Report the rank of the first evidence
  fact (MRR) as well as hit.
- **Divergence direction is policy-dependent**: a gate that admits long, specific turns (the full
  gate) retrieves longer facts, which help extractive answers; a gate that admits novel short facts
  may hit as often and answer worse. The proxy is not neutral across the compared policies. The
  results note's own item 7.6 ("validate evidence recall against F1 at least once") is the right
  requirement and should be a blocking gate for any claim.

---

## C. Statistical plan

### What is wrong with the plan as pre-registered and as executed

- **Unit of analysis.** Pre-reg §10: "paired bootstrap over questions (1,986 LoCoMo)". On LoCoMo the
  memory is built once per conversation and every question in that conversation is answered against
  the same admitted set; a policy's effect is a conversation-level treatment with 10 units, 5 in the
  current split. The question-level bootstrap treats 1,986 correlated outcomes as independent and its
  intervals (±0.03 on hit in `out/test_*`) are far too narrow. `stats.py` now adds a cluster bootstrap,
  which is the right instinct, but with 5 clusters it has ≤ 126 distinct resamples and its "p"
  (`stats.py:103`, share of resampled means on one side of zero) is not a hypothesis test. The
  values reported in `03-kickoff-results.md` §3 (Holm p < 0.001, 0.006, 0.029 from 5 clusters; several
  rows report p < 0.001 with only 4 of 5 conversations positive) cannot be produced by any exact
  conversation-level procedure: the smallest two-sided p attainable by a sign-flip test on 5 paired
  clusters is 2/32 = 0.0625, and a 4-of-5 split cannot be significant at any conventional level. The
  note's own conclusion ("every structural gate beats random and the 2010 product … with
  Holm-corrected p < 0.001") rests on these values.
- **Multiple comparisons.** `stats.py` applies Holm over whatever runs are on the command line (7
  comparisons in the note). The pre-registered family is "the four hypotheses". Neither matches the
  table actually produced (3 budgets × 8 policies × 5 categories × 2 directions).
- **H1's margin is not reachable.** Per-conversation admit-all hit ranges 0.486-0.642 (SD ≈ 0.05);
  paired per-conversation policy differences have SD ≈ 0.03-0.05 in the current runs. With n = 10
  the 95% half-width is ≈ 2.1-3.6 hit points; with n = 5, ≈ 4-6. A 2.0-point non-inferiority margin
  cannot be established on LoCoMo at this sample size whatever the effect.
- **H2's repeated answering is one answer** at temperature 0 with a shared cache key (A6.6); the
  "value" variable will be in {−1, 0, 1} with most pairs at 0, and Spearman with these ties is
  degenerate.

### Recommendation, concretely

1. **LoCoMo inference unit = conversation.** Use leave-one-conversation-out cross-fitting for every
   learned gate so that all 10 conversations are test units (fit on 9, score the 10th). Statistic:
   per-conversation mean paired difference in the metric. Primary test: exact sign-flip permutation
   over the 2^10 = 1,024 assignments (minimum two-sided p ≈ 0.002), with a 95% t-interval (9 df) on
   the per-conversation differences. Plot the 10 per-conversation differences; that plot *is* the
   evidence. Report the question-level paired bootstrap as secondary and label it anti-conservative.
   Optional model: `logit(hit_qp) ~ policy + (1 | question) + (policy | conversation)`; the policy
   contrast is within-question, its uncertainty comes from the conversation-level random slope.
2. **LongMemEval-S inference unit = question** (500 independent haystacks, each its own memory): the
   paired bootstrap over questions is legitimate there; stratify by the 6 question types (30-133
   each); treat it as the primary dataset for H1 and H3, LoCoMo as the replication with 10 units.
   Extraction seeds (3) are a nested level: report between-seed SD of each estimate and use the
   seed-mean per question as the unit.
3. **Size matching before any test**: bisection on q per policy to within ±5% live facts of the
   reference (or interpolate the size-vs-metric curve at the reference size), per conversation.
   Report realised live facts in every row (already done; keep it).
4. **H1**: one-sided 97.5% lower confidence bound of (graph gate − LLM gate) > −2.0 points, computed
   on LongMemEval-S accuracy (n = 500, paired SD of a 0/1 difference ≈ 0.4 → SE ≈ 1.8 points →
   half-width ≈ 3.5 points). Still underpowered at 2.0; either widen the margin to 4-5 points with a
   justification, or pool both datasets, or state that H1 is descriptive. Say which before running
   the LLM gate.
5. **H2**: pairs share facts and questions → cluster-bootstrap by fact; use Kendall τ-b or an
   ordinal model instead of Spearman on a 5-vs-5 accuracy delta; mandatory covariates: log|Ec| at
   admission (the SIVa clock), within-conversation position, fact length, category; sample the 300
   pairs stratified jointly by SIVa tertile *and* position tertile, otherwise the strata are the same
   thing. Repeated answering requires sampling (temperature 0.7 and a repeat index in the cache key);
   with a thinking model the repeats are not controlled by temperature at all.
6. **H3**: a feasibility gate before the test: at least 100 candidates per anchoring cell per dataset
   after the ER fix; otherwise H3 is reported as untestable in that arm, not as "confirmed on 19
   turns". Per LongMemEval type, Wilson intervals on value-per-fact.
7. **H4**: inject the same poison set (seeded) under every policy so admission decisions are paired
   per poisoned candidate; test with an exact binomial on discordant pairs (McNemar); Wilson intervals
   on admission rates; the harm metric needs the online arm.
8. **Multiplicity**: define four families with one primary endpoint each (H1: LongMemEval-S accuracy
   difference at matched size; H2: τ-b on one dataset, with "no significant negative on the other"
   as a secondary condition; H3: the two single-session types, both must hold — an intersection-union
   test, no correction inside the family; H4: poison admission-rate difference under the sourceless
   condition). Holm over the four primary p-values as pre-registered. Everything else (per category,
   per budget, 8 baselines) is exploratory: intervals, no p-values.
9. **Code**: replace `analysis.spearman` with average-rank Spearman; replace `stats.py:103` with the
   permutation p; store fit conversations in the weights JSON and refuse overlap at run time.

---

## D. Baseline fairness

- **admit_all**: no issue. It is also the cheapest policy and the best on every metric in the oracle
  arm (2,554 facts ≈ 50k tokens per 10 conversations); the paper must answer "why gate at all" with a
  token budget at which gating matters (LongMemEval-S haystacks), or the comparison is academic.
- **random**: no issue with the implementation; `run` takes one seed, §8 says 5 draws. Report the
  mean and spread over 5 seeds.
- **heat** (`policies.py:67-91`). MemoryOS's heat is a *promotion* score over mid-term memory pages
  with `N_visit` = retrieval hits, `L_interaction` = dialogue length, `R = exp(−Δt/μ)`. Here `visits`
  counts prior candidate turns mentioning the same normalised object string (`policies.py:82-83`),
  which is ≈ 0 for 88% of candidates because objects never recur; `rec` is `exp(−Δt/1e7)` with Δt in
  seconds between sessions (days to weeks apart → 0.9-1.0 whenever the entity was seen before, 0
  otherwise); the remaining term is `text_len / 50`. Net: a turn-length score with a small
  "re-mentions the other speaker" bonus. Honest label: "length + revisit heuristic (MemoryOS-inspired)".
  A version MemoryOS's authors would recognise needs resolved entity ids (after the ER fix), a
  visit counter driven by the read path (requires a query stream, which admission does not have — say
  this is why heat is not an admission policy), and their two-stage FIFO → promotion structure where
  "admission" is promotion at session end.
- **amac_lite** (`policies.py:135-148`). A-MAC (per the verdicts' verified reading): confidence =
  max ROUGE-L against *prior turns*, novelty = 1 − max SBERT cosine against *existing memories*,
  recency = exp(−λτ), type prior from rules, weights by grid search with 5-fold CV, plus one LLM
  utility call. Here: confidence = ROUGE-L of the GPT-4 observation against *its own source turn*
  (a paraphrase-faithfulness number, not agreement with prior turns, `policies.py:145`); novelty = 1 −
  same-subject token Jaccard (no embedding, no cross-subject search); recency = the constant 1.0
  (`policies.py:148`), which shifts every score by 0.2 and changes no ranking; type prior = membership
  in a hand-written predicate list that the oracle rule matches erratically by inflection
  (`plans → plan` yes, `planned → plann` no); weights 0.3/0.3/0.2/0.2 are the authors', not A-MAC's.
  Honest label: "two-signal lexical score (paraphrase ROUGE-L, Jaccard novelty) with a hand-set
  predicate prior". A fair version: A-MAC's feature definitions (ROUGE-L vs prior turns, embedding
  novelty over all live facts), weights fitted on the *same* fit split with the same grouped CV used
  for the graph gate (then the comparison is "same learner, different features", which is the only
  comparison a reviewer will accept), and the utility-call arm as pre-registered.
- **llm_gate** (`policies.py:151-184`). Not Mem0's prompt (no ADD/UPDATE/DELETE/NOOP, no retrieved
  similar memories); the "known facts" context is the first 10 out-edges of the subject, i.e. the 10
  *oldest* facts (`policies.py:173`), never the relevant ones, so the instruction "do not store …
  facts already implied by memory" cannot be followed. Honest label: "LLM store/skip judgement with
  stale context". A Mem0-faithful version: retrieve top-s facts by BM25/embedding similarity to the
  candidate, use Mem0's open-source update prompt verbatim, map ADD/UPDATE → admit and NOOP/DELETE →
  reject, report the model and the write-path tokens per 1,000 candidates (the accounting exists,
  `policies.py:182-184`). Also note the score mapping `c if store else 1 − c` with a default of 0.5 on
  parse failure (`policies.py:177-180`): under the quantile gate, parse failures are admitted or
  rejected depending on the running cut; log the failure rate.
- **siv_2010** (`policies.py:94-99`). The product is computed as specified, but (a) no cold-start
  rule (§8), so 88% of candidates score exactly 0 and the gate becomes first-come-first-served among
  them (B(iii)); (b) T_edge's degeneracy here is partly manufactured by the 6-token object chunks and
  the shadow speaker nodes (B(iv), B(v)). The 2010 authors would say their Gt is a document graph
  with IRI-aligned vertices and that this arm never gave the measure aligned vertices to work with.
  Honest label: "the 2010 product at turn granularity under our deterministic extraction"; the
  degeneracy analysis must show what T_edge does with head-noun entities and soft matching before
  calling the measure degenerate at fact granularity.
- **impact_2010**: in a two-star graph, `shared = {speaker}` and Impact = deg(speaker) / 2|Ec| =
  the speaker's share of facts. Report it as such; it is not in §8.
- **graph_gate (prior)**: honest only if the weights' provenance is stated (A3.5) and the raw-feature
  arithmetic is shown (novelty-minus-redundancy).

---

## E. The three most damaging objections, and the cheapest neutralisation

**E1. "You measure graph structure on a graph that has none."** Entities ≈ facts; zero non-hub
vertices with degree > 1 in conv-26; zero non-hub–non-hub edges; 1-hop expansion changes the
retrieved set on 11.5% of questions by ≤ 4 facts; anchoring, Impact' and T_edge are zero on 88% of
candidates and their non-zero values are 95%+ `mentions`-of-the-other-speaker artefacts of a typing
bug. "Structural features carry information about value" is untested; the paper would be reporting a
turn-length and GPT-4-observation-density effect under a graph vocabulary.
Cheapest fix (≈ 1-2 days, LLM-free): object = head noun / named entity (spaCy `en_core_web_sm`
noun-chunk root + NER), lemmatiser instead of the suffix stripper, type-insensitive resolution for
hub names; then a *precondition table* per arm (share of non-hub vertices with degree ≥ 2, share of
non-hub–non-hub edges, components, expansion firing rate) printed before any structural result, with
a stated threshold below which structural claims are not made. Wording fallback if the fix is not
done: "turn-level admission features over an entity-centric fact store"; drop "graph-structural" from
the title and H1.

**E2. "SIVa is a clock, and your consolidation never runs, so H2 and the SIVp/SIVa distinction are
void in this arm."** Spearman(SIVa, position) = −0.91; Spearman(SIVa_2010, SIVa_plain) = 0.99; 1
invalidation in 2,555 admissions; inverse closure absent. Any positive H2 result will be read as
"earlier facts matter more", any null as uninformative.
Cheapest fix: (a) make SIVa size-invariant (weight new edges relative to the mean live-edge weight,
or report SIVa × |Ec|) and in the H2 model control for log|Ec| at admission and for position; (b) give
the oracle rule a 30-line predicate normalisation map (`live/move/moved → lives_in`, `work/works →
works_at`, `married → married_to`, `study → studies`, …) so the functional rule can fire, then
hand-check 100 invalidations for precision; (c) until (b) is done, strike "the consolidation step that
makes SIVa distinct from a count ratio" from §11 and report SIVa_plain alone.

**E3. "Your inference cannot support the claims, your sizes are not matched, and your reported
numbers do not reproduce."** Five conversations per direction; Holm p-values that no exact
conversation-level test can produce; 24% spread in live facts under a "matched size" heading against
a pre-registered 5%; the results note quotes runs that are no longer in `out/`; the loader's category
names are swapped and the pre-registration repeats them.
Cheapest fix (≈ 1 day): fix `data.py:33` and the §6.1 sentence as a declared erratum; regenerate
`out/` with the code hash in every `summary.json`; leave-one-conversation-out cross-fitting so all 10
conversations are test clusters; exact sign-flip permutation and per-conversation dot plots as the
primary LoCoMo statistic; bisection on q to match live facts within 5%; LongMemEval-S (not the oracle
file) as the primary inferential dataset.

---

## F. Prioritised to-do list

| # | Item | Effort (h) | Blocks submission? |
|---|---|---|---|
| 1 | Fix the LoCoMo category mapping (`data.py:33`), set `adversarial` by `category == 5` (`data.py:130`), correct the §6.1 coverage sentence as a declared erratum, drop the "no LLM in extraction at all" phrasing, regenerate `out/` and add code + weights hashes to every `summary.json`; bring `03-kickoff-results.md` numbers into agreement with `out/` or delete them | 2 | yes |
| 2 | Entity layer: head-noun/NER objects and a real lemmatiser (spaCy), type-insensitive hub resolution, exclusion of candidates with no non-hub vertex from anchoring; graph-shape precondition table per arm; ER precision on a 200-mention sample | 12-16 | yes, for any structural claim (H1 wording, H3) |
| 3 | SIVa: size-invariant definition or explicit |Ec| control; predicate normalisation map so the functional rule fires; inverse closure; hand-check 100 invalidations; recompute A4.1-A4.3 and strike the §11 claim until the numbers support it | 6 | yes, for H2 and the consolidation claim |
| 4 | Leave-one-conversation-out cross-fitting; store fit conversations in the weights JSON and refuse overlapping runs; a true structural-only fit with only `{t_edge, anchoring, t_soft, impact, impact_log, novelty, novelty_pred}`; declare the 12-feature vector and the prior weights as deviations in §11 | 4 | yes |
| 5 | Statistics: exact sign-flip permutation on per-conversation paired means + 9-df t-interval as the primary LoCoMo test; per-conversation dot plots; question-level bootstrap labelled secondary; four hypothesis families with one primary endpoint each; fix `analysis.spearman` ties and `stats.py:103` | 5 | yes |
| 6 | Size matching: bisection on q per policy to ±5% live facts (or curve interpolation at a common size); warm-up without unconditional admission; report admitted share by session decile per policy | 4 | yes, for every "matched size" table |
| 7 | Official LoCoMo scorer: comma-split partial F1 for category 1, first-`;` gold for category 3, the two abstention substrings, comma/`and` removal; or vendor `evaluation.py` | 2 | yes, for the online arm |
| 8 | Online arm plumbing: LongMemEval-S as the default path; fixed small answerer with temperature 0 and no adaptive thinking; seed and temperature in the cache namespace; repeat index in the H2 cache key; token accounting per 1,000 candidates | 4 | yes, for H1/H2 and any F1 number |
| 9 | Validate the proxy once: run the online arm on the admit-all memories and on two gated memories, report per-question correlation of hit/frac/MRR with F1 per category, excluding category 5 from evidence recall | 6 + API cost | yes, before any claim rests on evidence-hit |
| 10 | Baselines: A-MAC-lite with A-MAC's feature definitions and weights fitted on the fit split; Mem0-faithful LLM gate (similarity-retrieved context, verbatim prompt, op mapping); relabel heat; 5 random seeds; honest labels in every table | 8 | yes for the fairness claim; otherwise relabel (1 h) |
| 11 | Poisoning arm per §9: sourceless condition (poison confidence = real tier, predicates from the real vocabulary, verbalised with a predicate), self-consistent repeats, hub-adjacent third-party subjects, paired injection across policies, the QA-harm metric; redo §4 of the results note | 6 | yes, for H4 |
| 12 | Rename the drifted hypotheses in the results note (its "H1" is an AUC diagnostic, its "H3" is a different hypothesis), and put the pre-registration under version control with the tag it promises | 1 | yes (cheap, but a reviewer who reads both documents will stop there) |

Not on the list because it is already acknowledged and correctly scoped in `03-kickoff-results.md`
§7: moving H3 (prior inversion) to the LLM-extraction arm, and reporting anchoring / soft agreement
with the sign the data gives — with the correction from B(ii) that the current sign is a `mentions`
artefact and has not yet been measured.
