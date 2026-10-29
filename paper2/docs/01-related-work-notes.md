# Related-work notes for paper 2: what each system does at write time

Purpose: let Section 2 of `paper/main.tex` state, for every related system, exactly what happens
when a candidate fact arrives (admission or consolidation mechanism), whether that step calls a
language model, and what the system measures. Everything below is taken from
`docs/literature-map.md` (sections 0-9, plus section 10 added 18 Oct 2026 from the kick-off reads),
`docs/adversarial-verdicts.md` (T2 section, where two reviewers read the primary sources directly),
`docs/topic-proposals-full.md` (T2), the PDFs under `docs/papers/al-saffar/`, and the seven reader
reports under `paper2/docs/reads/` dated 18 Oct 2026 (`baselines-code.md`, `data-locomo.md`,
`data-longmemeval.md`, `measure-lineage.md`, `related-2026.md`, `models-cost.md`, `venues-t2.md`).

Conventions. A claim marked **(VERIFY)** is one none of those documents supports; it is either from
memory or an inference, and must be checked against the paper before it is written into the
related-work section. A claim marked **(verified: source)** was settled on 18 Oct 2026 by a reader
who read the named primary source (raw code on the default branch, a PDF extracted with pdftotext,
or the downloaded data file); the reader report that holds the fact is named when the source is
not obvious. Published numbers of other systems are quoted only where a reader or the verdicts
recorded them, and should be re-read from the source before any appears in the paper; paper-level
numbers that a reader took from WebFetch summaries of arXiv HTML are medium-confidence by the
reader's own account. No number in this file comes from this project's experiment runs. The
Virtusa/Dell deck is not a source for anything here.

---

## A. The measures: al-Saffar and Heileman, 2008-2011

### al-Saffar & Heileman 2008 (DocEng 2008; IEEE IS 2008)
Two papers introduce subjective information valuation: the value of an information piece to a
consumer is a function of (i) the impact it would have on a graph representing the consumer's
world view and (ii) the trust the consumer places in the information's source. Bibliographic
details: "Semantic impact graphs for information valuation", DocEng 2008, São Paulo, 16-19 Sept
2008, pp. 209-212, DOI 10.1145/1410140.1410181; "Semantics-based information valuation", IEEE IS
2008, Varna, 6-8 Sept 2008, pp. 651-658, DOI 10.1109/IS.2008.4670438 (verified: Arizona experts
pages, `measure-lineage.md`). Both full texts are closed access (ACM returned 403, Semantic Scholar
status CLOSED); only the abstracts were read. The DocEng abstract scores a candidate "by how much it
would affect a knowledge base reflecting the recipient's world view", at statement level, estimable
via a causal graph without full logical inference; the IS abstract says value "can be predicted and
computed as potential causal impact of extractable semantics" (verified: abstracts). The earlier
documents' statement that the DocEng paper already uses the phrase *context graph* could not be
re-checked against the full text **(VERIFY)**. In 2008 trust is explicitly specified provenance
trust, not agreement: the only primary statement is the 2010 paper's self-citation, "In our previous
work we required the consumer to explicitly (maybe with the help of an assisting application)
specify degrees of trust they have in the provenance of the information source of Gt [3]", and the
2008 graphs are "a special kind of graph" (semantic impact / causal graphs) rather than plain RDF
(verified: iiWAS 2010 Sec. 4.2.1). Write-time mechanism: none in the agent sense. No LLM (2008).
Evaluation: qualitative. Relevance: the provenance-tier multiplier of H4 is a return to the 2008
design, as the novelty reviewer noted. Do not attribute formulas to the 2008 papers (reader's note).

### al-Saffar & Heileman 2010, "Computing Information Value from RDF Graph Properties" (iiWAS 2010)
Verified from the PDF (`docs/papers/al-saffar/al-Saffar_InfValue-iiWAS10.pdf`; re-read by two
readers with pdftotext, `measure-lineage.md` and `related-2026.md`; pp. 347-354, ACM
978-1-4503-0421-4): a target graph G_t (the document) and a context graph G_c (the consumer).
Actual value SIVa = ΔG_c (eq. 1), with G_c' = l(G_c, G_t) for a learning function l, "some learning
function representing the set of inference rules through which the information consumer may learn
from assertions in Gt and generate new assertions to update Gc" (eq. 2), computable when l is known
or G_c' is given; cast as a dissimilarity (eqs. 3-4) and computed with uniform edit costs as
(|E_c'| − |E_c|)/|E_c'| (eq. 5), where |E_c'| counts the edges of G_c' plus the G_c edges whose
labels changed, under the assumption that "knowledge does not decrease so relationships between
entities change but do not disappear" (verified). Predicted value uses plain union G_c' = G_t ∪ G_c
(eq. 6) with non-uniform costs w(e') = t × degree(v) for a new edge attached to a vertex v in
V_c ∩ V_t and cost 1 for edges of new disconnected components (eq. 7); Trust = |E_t ∩ E_c| / |E_t|
(eq. 8, Definition 2 "degree of agreement", normalised by |E_t| "to account for the number of
assertions made by the target document", asymmetric; "ten statements ... agree with eight ... trust
is 80 percent"; "Trust is necessary but not sufficient"); Impact = Σ degree(v in V_c ∩ V_t) /
Σ degree(v in V_c) (eq. 9); SIVp = f(Impact, Trust), "We use multiplication" (eq. 10). The paper
calls eq. 7 "a particular weights assignment that models trust and impact"; eqs. 7 and 10 are two
instantiations never shown to coincide (verified). Linear time because IRI labels make vertex
alignment trivial. The stated intuition: "learning of new unknown (disconnected) concepts is not as
important as learning related to already known concepts which in turn is less important than
learning concepts that are centrally connected" (verified). Cold start (Sec. 4.2.2): a G_t with no
overlapping vertices adds a disconnected component whose edges cost 1 under eq. 7, while eq. 8 gives
Trust = 0 and eq. 10 gives SIVp = 0 (verified). The context graph "can be created through a software
agent" and Fig. 2 is captioned "Learning as belief revision", so the agent framing is in the
original. Implementation: Firefox plug-in (AJAX/XUL) + Java servlet on Tomcat + DBpedia in Virtuoso
+ MySQL label cache, re-ranking web pages; G_t built "by scanning for terms appearing as literal
objects in the rdf:label property" (verified). Evaluation: none quantitative; the conclusion states
that traditional evaluation techniques do not apply to subjective valuation and that "human-based
survey evaluation techniques may be required" (verified). Stated main shortcoming: NLP, i.e. mapping
relationships from text ("usually have difficulty in mapping relationships which end up being the
edges"). No LLM. No code: the author's PNNL page lists five PDFs and nothing else (verified:
`measure-lineage.md`). Relevance: the measure under test; the degeneracy at fact granularity (Trust
in {0, 1/k, ..., 1}; hub-dominated Impact; SIVa under union = new-triple ratio) is intrinsic to the
published formulas (verified: `related-2026.md`), with the project's extraction rule making it worse
(`02-design-critique.md` B(iv)). The pre-registration's deviation log item 13 adopts the 2010 names:
`siva_2010` is eq. 5 under plain union, the degree-weighted consolidation-aware quantity is this
paper's redesign.

### al-Saffar, Joslyn & Chappell 2011 (WI-IAT 2011) and Joslyn, al-Saffar, Haglin & Holder 2011 (KDD MDS workshop)
The *extant ontology* is the class→predicate→class count graph computed by one GROUP BY over a
triple store, used for visualisation and *ontological scaling* (rolling classes up a hierarchy).
The motif paper measures the significance of typed subgraph motifs (link-type bigrams, trigrams)
as information gain relative to the frequency expected under independence of the constituent
link types. Both verified from the PDFs. No LLM. Relevance: our novelty term N is pattern
surprisal under the extant ontology with add-one smoothing; it is a port, not a new measure. The
KDD-MDS paper is unindexed (Crossref and Semantic Scholar miss it); cite the PNNL PDF. The kick-off
reads did not re-read these two papers.

---

## B. Graph deltas, informativeness, graph priors and agreement-based trust

### Berners-Lee & Connolly, "Delta" (W3C Design Issues, 2001, revised later)
An RDF graph delta as insert and delete set differences: "If G1 and G2 are ground RDF graphs, then
the ground graph delta of G1 and G2 is a pair (insertions, deletions) where insertions is the set
difference G2-G1 and deletions is G1-G2"; a weak delta applies only to the graph it was computed
from, a strong delta is context-independent via functional analogs of fully labelled graphs (blank
nodes labelled through owl:FunctionalProperty / owl:InverseFunctionalProperty paths), so blank nodes
turn it into a graph-isomorphism problem; ontology terms diff:replacement, diff:insertion,
diff:deletion, namespace `http://www.w3.org/2004/delta#`; the insertion/deletion form is implemented
in cwm; the algorithms are "not particularly efficient, having being designed as proof of concept"
(verified: https://www.w3.org/DesignIssues/Diff via WebFetch, `measure-lineage.md`). No write-time
mechanism, no LLM. Relevance: SIVa is a weighted instance of this twenty-year-old delta (reading
eq. 5 under its no-deletion assumption, the numerator is |insertions| plus label-changed edges, which
is the formalisation `measure-lineage.md` proposes for T2's graph-edit delta: a reader's suggestion,
not a statement in either source); the "IRI alignment makes isomorphism trivial"
observation is theirs and al-Saffar's, not ours.

### RELIN (Cheng, Tran & Qu, ISWC 2011) and the entity-summarisation lineage (Liu et al., JWS 2021)
Informativeness of a triple from self-information over graph frequencies, combined with relatedness
in a random-surfer model, for entity summarisation: x(t+1) = (M·Δ + J·Λ) x(t) with Δ_qq = 1 − λ,
Λ_qq = λ (eqs. 1-4); relatedness M_pq ∝ Rel(Prop(f_p), Prop(f_q)) × Rel(Val(f_p), Val(f_q)) with
Rel = PMI(s_i, s_j) = log(P(s_i, s_j) / (P(s_i) P(s_j))) estimated from Google hit counts (eqs. 5-6);
informativeness SelfInfo(o) = −log P(o) (eq. 7), P(f_p | f_q) = |{e : f_p, f_q ∈ FS(e)}| /
|{e : f_q ∈ FS(e)}| (eq. 8), J_pq = −log P(f_p | f_q) (eq. 9), approximable by SelfInfo(f_p); 10
iterations; λ ∈ {0, 0.15, 0.5, 0.85, 1}, best 0.85 at k = 10 (verified: Springer PDF, LNCS 7031,
pp. 114-129, DOI 10.1007/978-3-642-25073-6_8, `measure-lineage.md`). The 2021 survey (arXiv
1910.08252v2) restates si(⟨p, v⟩) = −log(|{e' : ⟨e', p, v⟩ ∈ Desc(e')}| / |E|) (eq. 12) and RELIN
(eqs. 17-20) but prints PMI without the log; cite the original for the formula (verified). Read-side,
no admission. No LLM. Relevance: a decade of Semantic Web precedent for valuing triples by frequency
statistics, which the paper's novelty term must be positioned against.

### SDType / SDValidate (Paulheim & Bizer, 2013/2014)
SDType (ISWC 2013, LNCS 8218, pp. 510-525): conf(T(r)) = ν Σ_p w_p P(T(r) | ∃p.(r)) with
w_p = Σ_t (P(t) − P(t | ∃p.))², ν = 1/Σ_p w_p, p and p⁻¹ weighted independently; "thresholds between
0.4 and 0.6 typically yield statements at a precision between 0.95 and 0.99"; F-measure 88.5%
(verified: Springer PDF, `measure-lineage.md`). SDValidate rejects implausible relation assertions
from the per-property distribution of object types: for x p o, "compares all the types of o to the
distribution of p, and computes a confidence score based on the deviation of those types from the
distribution" (verified: Melo & Paulheim, SWJ, as the citable secondary description). Venue
confirmed: IJSWIS 10(2):63-86, 2014, DOI 10.4018/ijswis.2014040104, CC-BY per Semantic Scholar
(verified: Semantic Scholar API record). The original article could not be retrieved (IGI Global
returned 429 on every path, the author's host redirects, archive.org was blocked), so the
"relative predicate frequency" first step and a cosine-similarity confidence with a fixed threshold
rest on secondary snippets only **(VERIFY)**. No LLM.

### Knowledge Vault (Dong et al., KDD 2014)
For each candidate triple, Pr(triple is true) from graph-based priors over the existing KG (Path
Ranking Algorithm random walks with per-predicate logistic regression, AUC 0.884; an MLP
Pr(G(s,p,o) = 1) = σ(βᵀ f(A[u_s, w_p, v_o])), L = K = 60, AUC 0.882; fused priors AUC 0.911) fused
with extractor confidences: per extractor "the square root of the number of sources that the
extractor extracted this triple from, and the mean score of the extractions from this extractor"
(sqrt(n) "to reduce the effect of very commonly expressed facts"; log(1+n) similar), one
boosted-decision-stump classifier per predicate, Platt scaling on a validation set, labels by the
Local Closed World Assumption; triples above a calibrated threshold (p ≥ 0.9 in the paper) are
"confident facts", and prior + extractor raises them from about 100M to about 271M (33% new vs
Freebase); Table 5 AUC on 10k test triples: prior 0.943, extractor 0.872, both 0.959 (LCWA labels)
(verified: author-hosted PDF, `measure-lineage.md`; the verdicts had already verified the threshold).
The fusion features contain no per-source trustworthiness term (verified). No LLM (2014). Relevance:
directly contradicts any claim that "no system values a candidate fact before promotion with a
graph-informational measure"; the paper's claim must be scoped to agent memory. The evaluation is at
web scale, calibrated, with no agent.

### Knowledge-Based Trust (Dong et al., 2015; arXiv 1502.03519, 12 Feb 2015; publication venue PVLDB 8(9):938–949 per `paper2/paper/refs.bib`, itself marked VERIFY; no reader report confirms it **(VERIFY)**)
Source trustworthiness A_w = "the probability that a value it provides for a fact is correct",
estimated from the correctness of the facts extracted from the source with a joint model that
separates extraction error from source error: single-layer ACCU p(X_sdv = 1 | V_d = v*, A_s) = A_s if
v = v*, (1 − A_s)/n otherwise, EM with A_s initialised 0.8 and M-step A_s = the average posterior
probability of the facts s provides (eqs. 1-4); multi-layer model with extractor recall R_e and
false-positive rate Q_e, presence and absence votes Pre_e = log R_e − log Q_e, Abs_e = log(1−R_e) −
log(1−Q_e) (eqs. 5-15); value votes VCV(w) = log(n A_w/(1−A_w)) and a softmax over summed votes
(eqs. 19-21); eq. 28, A_w = Σ p(C_wdv = 1 | X) p(V_d = v | X) / Σ p(C_wdv = 1 | X), "the key equation
behind Knowledge-based Trust estimation"; applied to 2.8B triples, 119M webpages and 5.6M websites
receive a KBT, 52% above 0.8, "almost orthogonal" to PageRank (verified: arXiv PDF,
`measure-lineage.md`). No LLM. Relevance: agreement-based trust is an old idea; the 2010 Trust ratio
is its naive version and lacks an extraction-error model, a limitation to state in Threats to
validity. KBT presumes several independent sources per data item, which a two-speaker conversation
rarely offers unless turns, speakers or extractor runs are treated as sources (reader's risk note).

---

## C. Agent memory systems and their write paths

### Mem0 and Mem0g (Chhikara et al., 2025; arXiv 2504.19413)
Write path as published: a two-phase extract-then-update pipeline. Extraction produces candidate
memories from a conversation summary plus the last m = 10 messages and the new pair; the update
phase retrieves the top s = 10 most similar stored memories and "the LLM itself determines which of
four distinct operations to execute": ADD, UPDATE, DELETE or NOOP, as a tool call; GPT-4o-mini for
all operations (verified: arXiv HTML, `baselines-code.md`, `data-locomo.md`). Mem0g (the graph
variant) adds an LLM-based update resolver for conflicts and embedding-threshold node matching. LLM
at write time: yes, one extraction call plus one update call per message batch (verified: code).
No structural score on writes.

Write path as coded (verified: `baselines-code.md`, raw `mem0/memory/main.py` at the named tags):
the classic ADD/UPDATE/DELETE/NONE gate exists through tag v1.0.11 (also at v0.1.100, v0.1.118,
v1.0.0, v1.0.2, v1.0.3, v1.0.5-v1.0.10) and is absent from v1.1.0 onwards; PyPI `mem0ai` 1.0.11 was
uploaded 2026-04-06, 2.2.1 on 2026-09-25, no 1.1.0 is listed. Current main (pyproject 2.2.1) is
ADD-only: `get_last_messages(limit=10)`, `vector_store.search(top_k=10)`, one LLM call with system
`ADDITIVE_EXTRACTION_PROMPT`, md5 dedup, batch insert with event ADD; `get_update_memory_messages` is
defined but never called. The classic flow (v1.0.2, L386-600): `parse_messages` → system prompt
`USER_MEMORY_EXTRACTION_PROMPT` (`AGENT_MEMORY_EXTRACTION_PROMPT` when `agent_id` is set and
assistant messages exist; `FACT_RETRIEVAL_PROMPT` only via the legacy helper), user "Input:\n{parsed}",
`json_object` → `facts` → per fact embed and `search(limit=5)` → union, dedupe, UUIDs remapped to
"0", "1", ... → one LLM call whose single user message is `get_update_memory_messages(old, facts,
custom)` → dispatch ADD / UPDATE / DELETE / NONE ("NOOP for Memory"). `FACT_RETRIEVAL_PROMPT` and
`DEFAULT_UPDATE_MEMORY_PROMPT` are byte-identical between v1.0.2 and main (the update prompt opens
"You are a smart memory manager which controls the memory of a system. You can perform four
operations ..."). LLM defaults: temperature 0.1, top_p 0.1, max_tokens 2000, OpenAI default
`gpt-4.1-nano-2025-04-14`. Paper says NOOP and tool call; code says NONE and JSON mode; temperature
0.1 is not deterministic.

Measures: LoCoMo with an LLM-judge score "J" per category and overall, F1 and BLEU-1 with A-MEM's
set-based utils, memory tokens per conversation and latency (verified: `evaluation/` at commit
aae5989, the last commit before that directory was removed from main). Judge gpt-4o-mini,
temperature 0, JSON mode, lenient prompt adapted from MemGPT ("as long as it touches on the same
topic as the gold answer, it should be counted as CORRECT"), category 5 skipped (1,540 questions),
answer model gpt-4o-mini, mean ± sd over 10 judge runs (verified: `data-locomo.md`). Table 2 overall
J: full-context 72.90, Mem0g 68.44, Mem0 66.88, Zep 65.99, LangMem 58.10, OpenAI 52.90, A-Mem 48.38,
best RAG 60.97; Mem0 per category J 67.13 / 51.15 / 72.93 / 55.51 as "single-hop / multi-hop /
open-domain / temporal"; 1,764 memory tokens retrieved per query vs 26,031 full context; store about
7k tokens per conversation (14k for Mem0g); search p50 0.148 s, total p50 0.708 s (verified: arXiv
HTML). The paper's column names follow Mem0's own label convention, not the official one (see
"Scoring conventions on LoCoMo"); its Table 1 appears to relabel A-MEM's published columns the same
way (the reader's inference, marked unverified in `data-locomo.md`; **(VERIFY)**). Zep's
rebuttal disputes Mem0's Zep run (65.99 vs Zep's own 75.14 ± 0.17): Mem0 assigned the user role to
both speakers, appended timestamps to message text instead of `created_at`, and ran searches
sequentially (verified: Zep blog). Relevance: the LLM-gate baseline (`llm_gate`). The
pre-registration (§8) promises the *verbatim* ADD/UPDATE/DELETE/NOOP prompt; `02-design-critique.md`
D records that the current `llm_gate` is a custom store/skip prompt with stale context, and
`05-critique-response.md` item 10 lists the Mem0-verbatim prompt as still owed. Pin `mem0ai==1.0.11`
or rebuild the gate from the prompts above.

### Zep / Graphiti (Rasmussen et al., 2025; arXiv 2501.13956; getzep/graphiti)
Write path: episodes are ingested; entity and edge extraction by model; entity and edge
deduplication are LLM-judged after embedding and full-text candidate retrieval; edge invalidation
is LLM-based ("employs an LLM to compare new edges against semantically related existing edges");
edges carry bitemporal fields (t_valid, t_invalid; t_created, t_expired) and contradictions set
t_invalid rather than deleting. The only structural write step is community assignment by
plurality of neighbours. The Graphiti repository routes dedupe and edge resolution through LLM
prompts, optionally through a "decision model" answering typed questions with probabilities
(duplicate and contradiction thresholds), still a model call. LLM at write time: yes, several.
Measures: the paper contains no LoCoMo evaluation, only DMR and LongMemEval_S: gpt-4o-mini
full-context 55.4% vs Zep 63.8%, gpt-4o 60.2% vs 71.2%, GPT-4o judge with the LongMemEval
question-specific prompts, average context 1.6k tokens vs 115k (verified: arXiv HTML,
`data-longmemeval.md`; its per-type gpt-4o single-session-user row is identical to the gpt-4o-mini
row, a likely typo). Zep's separate `locomo_eval` code uses Mem0's judge prompt with gpt-4o-mini,
filters category 5, ingests each turn as "speaker: text" with `created_at` from the session date,
and its README reports overall 75.14 with per-category labels that `data-locomo.md` shows are
permuted relative to the data (see "Scoring conventions on LoCoMo"); the released per-question
grades file (1,540 items) joins to every locomo10.json question (verified). Relevance: the LLM-gate
family; the bitemporal never-delete convention is also ours (valid_from / valid_to); the grades file
gives a per-question correctness vector for a strong commercial baseline.

### A-MAC, "Adaptive Memory Admission Control for LLM Agents" (Zhang et al., 2026; arXiv 2603.04549)
Write path: admission as a linear score over five factors, S(m) = Σ w_i f_i, admit iff S ≥ θ
(verified: `scorer.py`). Utility = one LLM call (Ollama `qwen2.5:latest`, 5-turn context, a single
float in [0, 1], fallback 0.5; the temperature argument is never passed); Confidence = max ROUGE-L
F-measure (stemmed) of the candidate against every earlier turn sharing at least one token, 0.0 if
none; Novelty = 1 − max cosine (all-MiniLM-L6-v2) against existing memories, 1.0 when the set is
empty; Recency = exp(−0.01 × age_hours); type prior = keyword-substring counts per type, argmax,
with priors preference/identity/belief/value 0.9, fact/knowledge/information 0.7,
plan/goal/intention/task 0.5, state 0.3, temporary 0.2, unknown 0.5 (verified: `features/*.py`,
`baselines-code.md`; the paper calls the type prior "POS pattern rules", the code is keyword
counting). Weights and threshold: the README and paper use [0.1, 0.1, 0.1, 0.1, 0.6] over
(U, C, N, R, T) and θ = 0.55; the search actually run is equal weights + 200 random vectors + five
"emphasised" vectors crossed with thresholds {0.30, ..., 0.60}, scored by mean validation-fold F1 in
`KFold(5, seed 42)` with nothing fitted on training folds (verified: `optimize_weights_cv.py`). Four
of the five factors are LLM-free. No graph, degree, overlap or edit-distance term. LLM at write time:
one call (utility), 2580 ms of the 2644 ms per candidate (verified: paper).

Two factors are inert in the authors' own experiments: both experiment scripts compute Novelty
against an empty list, so N = 1.0 for every candidate; Recency is computed against `time.time()`
while the loader gives turns synthetic 2023 timestamps, so R ≈ 0 (verified by reading the code, not
by executing it: `baselines-code.md`). The published gate is thus effectively 0.1 U + 0.1 C + 0.1 +
0.6 T, and a candidate typed preference/identity is admitted regardless of U and C.

Measures: precision, recall and F1 of *admission decisions*. The labels are now documented from the
code: every LoCoMo dialogue turn is a candidate, positive iff its `dia_id` appears in any QA
`evidence` list; 70/15/15 conversation-level split with seed 42 on `locomo10.json` (10
conversations); only the first 150 train+val and the first 100 test candidates are used (verified:
`data_loader.py`, `optimize_weights_cv.py`). The paper says "approximately 1,500 candidate memories
with ground-truth admission labels" without explaining their construction, "LoCoMo test set
(N = 225)" in Table 1, and A-MAC F1 0.583 (val 0.571) vs A-mem 0.541 (precision 0.371, recall
1.000) (verified: arXiv HTML, `related-2026.md`). The earlier reading that the data description
("30 conversations of 15-40 turns") does not match LoCoMo stands against the paper text; the code
loads the 10-conversation file. **No downstream QA is measured**, so the number cannot be compared
to ours. Code released (MIT, one commit) without the data file, the result JSONs, the README's
`WeightOptimizerCV` / `ConversationTurn` / `load_locomo_dataset` APIs, or the paper's Algorithm 1
conflict merge (similarity ≥ 0.85) (verified). The utility model is an unpinned Ollama tag, so the
numbers are not exactly reproducible. Venue: "ICLR 2026 MemAgents workshop" per the arXiv HTML
(verified: `related-2026.md`); the OpenReview page (id mmdqUrEY24) was not fetched **(VERIFY)**.

Relevance: `amac_lite` drops the utility call. `02-design-critique.md` D and `05-critique-response.md`
item 10 record that the project's variant must use A-MAC's feature definitions (ROUGE-L against prior
turns, embedding novelty over live facts) with weights fitted by the same learner as the graph gate;
the recency constant (0.01 per hour) and the type-prior rule above are now known, so a reproduction
can be described as such provided it also states whether the empty-memory novelty and wall-clock
recency quirks are reproduced or corrected.

### MemoryOS (Kang, Ji, Zhao & Bai, 2025; arXiv 2506.06326; EMNLP 2025)
Write path: three tiers (short-, mid-, long-term). Short-term eviction is FIFO; promotion from
mid-term to long-term is an arithmetic rule, Heat = α·N_visit + β·L_interaction + γ·R_recency with
R_recency = exp(−Δt/μ), α = β = γ = 1, μ = 1e7 s, promoted when Heat > τ = 5 and L_interaction then
reset; STM length 7; MTM 200 segments with lowest-heat eviction; pages merge into a segment when
cos + Jaccard ≥ 0.6 (verified: paper PDF and HTML, `related-2026.md`, `baselines-code.md`). The code
disagrees with the paper on constants: the library (`memoryos-pypi/mid_term.py`) uses α = β = 1.0,
γ = 1 with R = exp(−Δhours/24) and LFU eviction by access count, not lowest heat; the `eval/` scripts
that produced the LoCoMo numbers use α = β = 0.8, γ = 0.0001, R = exp(−Δs/3600), STM capacity 1, MTM
2000, retrieval queue 10, thresholds 0.1, answers from gpt-4o-mini at temperature 0.7, and a merge
keyword score 0.5(|overlap|/|A| + |overlap|/|B|) rather than Jaccard (verified: `baselines-code.md`).
LLM at the heat computation: no. LLM elsewhere on the write path: yes. The STM→MTM step calls the
model twice per evicted page (continuity check against the previous page, page meta-information)
plus once per batch (multi-theme summary), so three calls only when a batch holds one page, and the
promotion trigger, when heat ≥ threshold, runs two parallel LLM calls (user-profile analysis and
knowledge extraction) before writing to long-term memory and resetting N_visit and L_interaction to
0 (verified: `updater.py`, `memoryos.py`). N_visit increments only when a retrieval matches a page,
so in a harness that ingests all turns before any query heat reduces to L_interaction plus recency,
i.e. segment size (reader's risk note).

Measures: LoCoMo F1 and BLEU-1 on four categories (single-hop / multi-hop / temporal / open-domain
in the paper's order) with GPT-4o-mini and Qwen2.5-3B backbones; MemoryOS GPT-4o-mini F1/BLEU-1
35.27/25.22, 41.15/30.76, 20.02/16.52, 48.62/42.99; about 3,874 tokens and 4.9 LLM calls per
response; the evaluation script prints set-based F1 by raw category integer, and the paper's
MemGPT and A-Mem rows are A-MEM's category 1-4 numbers relabelled in the paper's prose order
(verified: `data-locomo.md`; see "Scoring conventions on LoCoMo"). Repository: BAI-LAB/MemoryOS,
Apache-2.0, ships `eval/locomo10.json` (verified). Relevance: prior art for "LLM-free arithmetic
promotion on LoCoMo", usage-based not structural; our `heat` baseline is a write-time analogue of
this promotion score (visits of the candidate's non-hub entities, turn length / 50, exp(−Δt/μ) with
μ = 1e7 s). MemoryOS uses heat for promotion, not admission; the paper must say so, and should
expose α, β, γ, the decay constant and the threshold as configuration so the paper, library and
eval variants can all be run (reader's suggestion). `02-design-critique.md` D gives the honest label
for the current implementation ("length + revisit heuristic (MemoryOS-inspired)").

### Memory-R1 (Yan et al., 2025; arXiv 2508.19828 v5; ACL 2026 long paper 2026.acl-long.583)
Write path: a "memory manager" LLM is trained with reinforcement learning (PPO or GRPO) to choose
ADD / UPDATE / DELETE / NOOP for each fact that an LLM extraction step produces (prompt adapted from
MemGPT / Mem0), given retrieved old memories; the reward is exact match of a frozen "answer agent"
on the updated bank, no judge; the answer agent retrieves 60 memories (30 per speaker) and distils
them; training data 152 / 81 / 1,307 QA (1:1:8) from LoCoMo minus adversarial, no gold operation
labels; backbones LLaMA-3.1-8B-Instruct and Qwen-2.5-3B/7B/14B-Instruct; VERL on 4 × H100 (8 for
14B), batch 128, prompt 4096 / response 2048 tokens, PPO actor lr 1e-6, critic 1e-5, GRPO actor
only; the memory bank is built with GPT-4o-mini (the text is inconsistent on 24 vs 50 preceding
turns) (verified: arXiv HTML, `baselines-code.md`). LLM at write time: yes, the manager is itself a
fine-tuned model, so a faithful reproduction is one model call per candidate at inference plus the
extraction call. Measures: LoCoMo F1 / BLEU-1 / J (judge model not named) on the 1,307-question
test split; LLaMA-3.1-8B overall: Mem0 30.41/22.22/45.68, PPO 41.05/32.91/57.54, GRPO
45.02/37.51/62.74; Qwen-2.5-7B: Mem0 30.61/23.55/53.30, GRPO 43.14/36.44/61.51; MSC; LongMemEval
zero-shot after LoCoMo training, GRPO J 55.40 (LLaMA) and 57.80 (Qwen), split and judge unstated
(verified: `baselines-code.md`, `data-longmemeval.md`). Code: the official repository
yansikuan/memory-r1 holds only README, LICENSE (Apache-2.0) and assets, "Code coming soon" as of
2026-10-18; the Anthology page lists no code URL; the one third-party reproduction
(pradyutnair-prosus/memory-r1, Qwen2.5-7B + LoRA r = 64, GRPO on LoCoMo conv-26) reports Memory
Manager validation EM 0.0000 / F1 0.0371 ("reward is sparse") and marks the paper's models "Not
released" (verified). Relevance: the pre-registration's "learned gate, 0 LLM calls at inference" is
therefore not Memory-R1 as published; it can only be a small classifier distilled from gate
decisions, and any "Memory-R1-style" gate is a reimplementation that cannot be called faithful. On a
24 GB GPU the reachable form is LoRA-GRPO on a 3-4B model (`models-cost.md`). Decide and state
which; flagged in `main.tex`.

### LightMem (Fang et al., 2025; arXiv 2510.18866 v4; ICLR 2026)
Write path: lightweight pre-filtering and compression of what enters memory and "sleep-time"
(offline) updates instead of online LLM updates per write. As coded (`src/lightmem/memory/lightmem.py`):
optional LLMLingua-2 compression per message (re-compressed while ≥ 512 tokens; empty output keeps
the original); a sensory buffer counting user-role tokens against 512; coarse topic boundaries from
local maxima of sentence-level attention in LLMLingua-2 layers [8, 9, 10, 11] with the first and
last 3 tokens masked; fine boundaries where adjacent-turn cosine (all-MiniLM-L6-v2) falls below an
adaptive threshold starting at 0.2 and rising by 0.05 to 0.5, kept when within ±3 of a coarse
boundary; a short-term buffer hard-coded to 512 tokens; one JSON extraction call
(`METADATA_GENERATE_PROMPT`) per flushed batch; `offline_update` embeds and inserts into Qdrant;
`online_update` is a stub returning None; the offline queue (`top_k=20`, `keep_top_n=10`) is
resolved by `UPDATE_PROMPT` calls returning update / delete / ignore (similarity threshold 0.8,
score threshold 0.9 by default, 0.8 in the LongMemEval script); the config field
`extract_threshold` is never referenced (verified: `baselines-code.md`). The filter is a token
classifier (LLMLingua-2) plus attention and embedding heuristics, not an LLM; the remaining model
calls are the per-batch extraction call and the offline update calls. LLM at write time: reduced,
not zero (resolved). Claims (v4 abstract): QA accuracy +7.7% (LongMemEval) / +29.3% (LoCoMo), total
tokens down up to 38x / 20.9x, API calls down up to 30x / 55.5x, purely online test-time cost down
106x / 117x tokens and 159x / 310x calls; the v1 abstract claimed "up to 10.9%" accuracy and "up to
117x" fewer tokens, later versions report 2.09-6.40% (GPT) and up to 7.67% (Qwen) (verified: arXiv
abs pages). Measures: LongMemEval-S and LoCoMo accuracy by a gpt-4o-mini judge (LoCoMo overall only,
no per-category numbers; single runs, no variance), plus "Total tokens" (input + output of summary
and update calls), "Calls" (LLM invocations during construction) and "Runtime" (wall-clock
construction including LLMLingua-2), retrieval and answering excluded, embeddings separate. Table 2
LongMemEval-S, GPT-4o-mini: Full-text 56.80, Naive RAG 61.00, LangMem 37.20, A-MEM 62.60, MemoryOS
44.80, Mem0 53.61, LightMem 64.29-68.64; Qwen3-30B-A3B: Mem0 39.51, A-MEM 65.20, MemoryOS 49.60,
LightMem up to 70.20 (a 73.20 cell is read by `baselines-code.md` as Table 9's value for the same
Qwen setting and by `data-longmemeval.md` as a GLM-4.6-backbone row; the readers disagree,
**(VERIFY)** against Table 9); LoCoMo ACC GPT-4o-mini / Qwen:
FullText 71.83/74.87, NaiveRAG 63.64/66.95, LangMem 57.20/60.53, A-MEM 64.16/56.10, MemoryOS
(locomo script) 58.25/61.04, Mem0 61.69/43.31, LightMem best 72.99/72.60 (verified: arXiv HTML,
`data-longmemeval.md`, `data-locomo.md`). Baselines were measured with `memory_toolkits/token_monitor.py`
(monkey-patched LLM calls, litellm tokenizer counts with the gpt-4 tokenizer) while LightMem's own
numbers come from provider `usage`, so cross-system efficiency comparisons mix methodologies
(verified). Code MIT. Relevance: sets the bar for any efficiency claim; the paper reports write-path
calls and tokens per 1,000 candidates next to this accounting, not a headline "fraction of LLM
calls", and must count all systems with one method.

### GAM, "Hierarchical Graph-based Agentic Memory for LLM Agents" (Wu et al., 2026; arXiv 2604.12285)
Write path: an event-progression graph is consolidated into a topic network when a "semantic
shift" is detected. The paper formally defines the consolidation trigger b_t = 1[Δ(G_event,
G_topic) > ε] (Eq. 2) and then states that "explicitly computing the high-dimensional distance Δ is
computationally expensive and sensitive to lexical noise", so the detection is implemented as a
semantic discrimination task driven by an LLM, "a neural proxy for the thresholding function", and
"the instruction strictness of the semantic discrimination prompt acts as a heuristic proxy for ε";
Δ is never given a closed form and ε never a number; Algorithm 2 checks at session end or when the
buffer exceeds T_max = 2048 tokens; consolidation retrieves top-K = 5 topic candidates and adds an
edge when the LLM's relation confidence exceeds an unreported τ; retrieval k = 10, all-MiniLM-L6-v2
embeddings, ms-marco-MiniLM-L-6-v2 reranker (verified: arXiv HTML, `measure-lineage.md`). No
admission gate. LLM at write time: yes. Code: neither reader report that covers GAM records a
repository or its absence **(VERIFY)**. Measures: LoCoMo average F1 40.00 with
Qwen2.5-7B vs Mem0 35.38, 43.14 with GPT-4o-mini (verified: `related-2026.md`); other benchmarks
**(VERIFY)**. Authors: Wu et al., submitted 14 Apr 2026 (verified: `measure-lineage.md`). Relevance:
the strongest published statement of the gap SIVa fills; its "lexical noise" objection is a real risk
for SIVa after imperfect entity resolution and is addressed by the oracle-ER arm. Any "GAM baseline"
is a re-interpretation and must be labelled so.

### Kumiho, "Graph-Native Cognitive Memory for AI Agents: Formal Belief Revision Semantics" (Park, 2026; arXiv 2603.17244)
Write path: AGM / Hansson belief-revision postulates mapped onto a Neo4j + Redis property-graph
memory with immutable revisions and tag pointers; `AnalyzeImpact` is a read-side dependency-edge
BFS; an offline "Dream State" consolidation uses an LLM-assigned relevance_score in [0, 1]. No
pre-ingestion admission score and no magnitude-of-change measure; section 16.2 lists in-degree in
the dependency graph as a future entrenchment ordering. LLM at write time: in consolidation, yes.
Measures: LoCoMo token F1 with Porter stemming "following the official LoCoMo metric" on all 1,986
questions, four-category F1 on n = 1,540 plus binary refusal accuracy on the 446 adversarial
questions; 0.447 four-category F1, 97.5% refusal, 0.565 overall; the paper says there is "no
standardized LoCoMo leaderboard" (verified: arXiv HTML, `data-locomo.md`). The benchmark README
(kumiho-benchmarks) pins the answer model to gpt-4o-2024-08-06 and reports a reproducible 0.424 /
0.361 / 0.457 / 0.248 / 0.955, overall 0.531, calling the earlier 0.565 / 0.564 runs with unpinned
gpt-4o non-reproducible, and claims an LLM judge inflates scores 1.5-2x over token F1 (verified).
Relevance: anticipates but does not implement a degree-based signal; its F1 is the official
stemmed token F1, comparable in kind to our LoCoMo metric but not in pipeline; its drift note is the
argument for pinning model snapshots.

### Nemori, "What Deserves Memory: Adaptive Memory Distillation for LLM Agents" (2025; arXiv 2508.03341)
Write path: advertised as prediction-error (surprise) based; the full text shows two LLM calls per
write, one producing an anticipatory schema and one extracting the deviation, then labelling
new / merge / conflict; there is no numeric surprise score. LLM at write time: yes, two calls.
Measures: LoCoMo on 1,540 questions (adversarial excluded) with a gpt-4o-mini judge whose prompt
wording matches Mem0's, answer models gpt-4o-mini and gpt-4.1-mini; Table 2 averages gpt-4o-mini:
Nemori 73.0, full context 72.3, Mem0 61.3, Zep 58.5, A-MEM 52.5, MemoryOS 54.5, LangMem 51.3;
gpt-4.1-mini Nemori 80.8 vs full context 80.6; the averages reproduce only under the official label
mapping (verified: arXiv HTML, `data-locomo.md`). LongMemEval_S: gpt-4o-mini full-context 55.0% vs
Nemori 64.2%, gpt-4.1-mini 65.6% vs 74.6%, context 101K vs 3.7-4.8K tokens, judge model not stated
(verified: `data-longmemeval.md`). Relevance: an LLM-judged baseline family; also illustrates how
far QA numbers spread across pipelines, which dwarfs plausible admission effects (a
Threats-to-validity point).

### Hindsight (2025; arXiv 2512.12818; vectorize-io/hindsight)
Evidence-backed "observations" whose proof counts are refined as new evidence arrives; the
mechanism is not disclosed in the abstract or README. LLM at write time: **(VERIFY)**, no reader
report reaches the mechanism. Measures: LoCoMo scored by a binary LLM judge with GPT-OSS-120B at
temperature 0; Table 4 overall Hindsight OSS-20B 83.18, OSS-120B 85.67, Gemini-3 89.61; its baseline
rows (Backboard 90.00, Memobase 75.78, Zep 75.14, Mem0-Graph 68.44, Mem0 66.88, LangMem 58.10,
OpenAI 52.90) are copied from Backboard's published results, not reproduced, and its per-category
columns follow the Mem0 label convention (the 89.61 reproduces as a weighted mean only with
n = 282/96/841/321) (verified: arXiv HTML, `data-locomo.md`, the convention by arithmetic).
LongMemEval_S (judge GPT-OSS-120b): Hindsight 83.6 / 89.0 / 91.4 (OSS-20B / OSS-120B / Gemini-3 Pro
answerer), Supermemory 81.6 / 84.6 / 85.2, full-context GPT-4o 60.2, Zep 71.2, where the
full-context and Zep rows reuse Supermemory's GPT-4o-judged numbers; the GitHub README repeats the
table without naming the judge or dataset version (verified: `data-longmemeval.md`). Relevance: a
count-based agreement signal on beliefs, and the current state-of-the-art bar that an admission-only
intervention sits far below; cite only if its mechanism can be verified, and never place its numbers
on the same axis as gpt-4o- or F1-scored systems.

### A-MEM (Xu et al., NeurIPS 2025; arXiv 2502.12110 v11)
Write path: Zettelkasten-style notes; a model generates note attributes (keywords, tags, context)
and links to existing notes, and "memory evolution" updates existing notes when new ones arrive.
LLM at write time: yes **(VERIFY)** the exact number of calls per write and whether linking uses
embeddings before the model; no reader report reaches the write path. Measures: F1 and BLEU-1 (plus
ROUGE, METEOR, SBERT in the appendix) on all five LoCoMo categories including adversarial, table
order Multi Hop / Temporal / Open Domain / Single Hop / Adversarial; GPT-4o-mini F1: A-Mem
27.02/45.85/12.14/44.65/50.03, MemGPT 26.65/25.52/9.15/41.04/43.29, LoCoMo baseline
25.02/18.41/12.04/40.36/69.23; the reproduction repository README states the official mapping
(1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial) and its set-based, unstemmed
F1 utils are the ones Mem0, MemoryOS and Memory-R1 borrowed (verified: arXiv HTML and raw README,
`data-locomo.md`). LightMem's accounting attributes 986.55 construction calls to A-MEM on
LongMemEval-S (verified: LightMem Table 2). Relevance: LLM-gate family, graph-like links without a
structural admission score. Its LoCoMo J under the Mem0 protocol (48.38) is the lowest of the
systems Mem0 compared (verified: Mem0 Table 2); MemoryOS's table relabels its columns (verified), and
Mem0's Table 1 appears to as well (reader's inference, **(VERIFY)**; see "Scoring conventions on
LoCoMo").

### MemoryBank (Zhong et al., AAAI 2024)
Write path: stores conversation summaries and user-portrait updates generated by a model; retention
follows an Ebbinghaus forgetting curve updated on recall. LLM at write time: yes for summaries
**(VERIFY)**. Measures: a custom long-term dialogue set with human and model judgements
**(VERIFY)**. Relevance: the forgetting-curve family; no admission score. Not covered by the kick-off
reads.

### Generative Agents (Park et al., UIST 2023)
Write path: every observation is stored; retrieval scores memories by recency × importance ×
relevance, where importance is a 1-10 rating produced by the model at write time; reflections are
synthesised by the model when accumulated importance crosses a threshold. LLM at write time: yes
(importance, reflection) **(VERIFY)** the exact prompt. Measures: believability study. Relevance:
the first LLM-era importance score, a precedent for a per-write value signal, but model-produced.
Not covered by the kick-off reads.

### MemGPT / Letta (Packer et al., 2023)
Write path: the model itself manages tiers (main context, recall, archival) through function
calls; what to write is the model's decision. LLM at write time: yes, by construction. Measures:
document QA and multi-session chat consistency. Relevance: the "memory as OS" framing; no
structural signal. The Mem0 judge prompt is "adapted from Packer et al." (verified: Mem0 paper).

### EM-LLM (Fountas et al., ICLR 2025)
Write path: token-level episodic segmentation by Bayesian surprise with graph-theoretic boundary
refinement, inside the model's context management; LLM-judge-free. LLM at write time: no separate
call, but it operates on the model's own token stream, not on extracted facts. Measures:
LongBench and InfiniteBench. Relevance: weakens any blanket "nobody uses graph structure for
memory writes" phrasing; the paper's claim is about fact admission in a symbolic memory.

### HippoRAG and HippoRAG 2 (Gutiérrez et al., NeurIPS 2024; ICML 2025)
Write path: an open-information-extraction KG built by the model at indexing time; the one
frequency-based graph weight, *node specificity* (inverse frequency), and Personalized PageRank
act at read time. LLM at write time: yes (OpenIE). Measures: multi-hop QA (HotpotQA, MuSiQue,
2Wiki). Relevance: read-side structural weighting; HippoRAG 2 is the selector that Fei et al. attack
(their Table VIII: HippoRAG2 @ d437bfb, OpenIE, union-find merge, igraph PPR; verified:
`measure-lineage.md`).

### neo4j-labs/agent-memory and the Neo4j Agent Memory Service
Write path: POLE+O long-term graph, extraction via spaCy / GLiNER / LLM, three-stage entity
resolution with a pending SAME_AS review queue, a dedupe primitive, reasoning traces with TOUCHED
edges and a count-level read audit; no admission scoring is documented. Skill distillation gates
(grounding ≥ 0.9, coverage ≥ 0.6, pending human review) exist for procedural memory, not for facts.
LLM at write time: optional for extraction. Measures: none published. Relevance: confirms that a
production LPG memory has no competing structural value function; our graph could be mirrored
there but the pre-registration dropped the mirror.

### MemOS (Li et al., 2025)
"MemCube" units with provenance and versioning as a memory operating system. Write-time admission
mechanism: **(VERIFY)**; the docs record only the framing and no reader report covers it. One data
point: LoCoMo_refined re-scores MemOS at 63.60% under its strict Qwen3-14B judge (verified:
`data-locomo.md`). Relevance: provenance-carrying memory units, comparable to our tiered edges; cite
only for that.

### Systems added from the May-October 2026 sweep (`related-2026.md`, read 18 Oct 2026)
None of these scores a candidate fact by degree, overlap or connectivity of the existing graph
before admission; none measures a write's realised graph delta; none tests a
connected-vs-disconnected prior (verified absence across about 90 screened papers, with full texts
read for GovMem, Epistemics, CPB, What-to-Admit, A-MAC, Fei et al., STRATA, EngramRAG, HeLa-Mem, GAM
and the graph-memory survey; the rest on abstracts). Numbers are the papers' own.

| System | Write-time mechanism | LLM at write | Where structure enters | Measures | Relevance to T2 | Flag |
|---|---|---|---|---|---|---|
| Selective Forgetting (arXiv 2608.28978, Rusu, Khanzadeh, Alalfi, 29 Aug 2026) | writes unchanged; periodic pruning of nodes scoring low on weighted recency, access frequency, degree centrality, age | not stated | forgetting | LongMemEval token F1; graph did not beat a flat vector baseline; ~9.8% nodes removed, no F1 change | degree as a retention signal, the nearest "degree is not new" objection | verified (abstract) |
| EngramRAG (arXiv 2609.32049, Potineni et al., 25 Sep 2026) | dedup by cosine > 0.95 with an access-count increment; Hebbian edge weights 0.4 cos + 0.4 Σ exp(−λΔt) + 0.2 R_type | not stated | decay λ_eff(v) = λ_base(type)·exp(−2.0·Score_topology(v)), Score_topology from centrality, degree, type prior; U-PPR d = 0.85, θ = 0.6; log scaling against hub capture | LoCoMo Recall@5 53.21% vs dense 38.29%, MRR 0.4203 vs 0.2937; SUPERSEDES filtering cut split-brain hallucinations 70% → 0%; no per-write delta, no topology-vs-recency ablation on QA | topology at decay time; its log scaling is reusable against speaker-hub Impact | verified (full text) |
| STRATA (arXiv 2610.19386, Zhu et al., 14 Oct 2026) | similarity band: noop if s > τ_high, add_bullet in [τ_low, τ_high], else create | not stated for the band | GC deletes lowest-Q nodes to cap·(1−0.15) but protects Tarjan cut vertices; Louvain (resolution 1.5) meta-frameworks promoted if conductance < 0.6 and novelty distance ≥ 0.6 | ablation: articulation protection 69.8 → 72.9 success, regret 13.2% → 4.8%, bridges kept 0 → 16; removing the novelty gate raises meta churn 3.1% → 28.6%; no per-write delta | structure protects at GC, scalar similarity admits | verified (full text) |
| HeLa-Mem (arXiv 2604.16839, ACL 2026) | links consecutive turns with small weights | not stated | hubs flagged for Hebbian distillation when D(v) = Σ_j w_ij > δ_hub (value not given) | LoCoMo F1 (GPT-4o-mini) multi 40.14 / temporal 47.29 / open 29.70 / single 51.89 vs its MemoryOS reproduction 38.39/41.58/23.75/45.86; Table 3 "Full" row inconsistent with Table 1 | hub detection for distillation only | verified (full text) |
| Beyond Freshness: Closure-Aware Admission (OpenReview Ulqx7pVvcS, ICLR 2027 submission, anonymous) | task-constraint validation plus transitive dependency closure quarantines inadmissible memories | not stated | reachability at admission | matches full transitive-closure decisions on SQLite and Mem0 workloads; no numbers in the abstract | the only admission-time structural rule found; reachability for invalidation, not novelty or impact; under review | verified (abstract) |
| SAGE: A Novelty Gate (arXiv 2605.30711, Wang, Brahma, Henao, v2 18 Jun 2026) | von Mises-Fisher density over memory embeddings with an adaptive threshold tracking store geometry; routes ADD / skip / LLM-merge | only on merge | none (embedding space) | gains over Mem0 on LoCoMo with GPT-4o-mini | an LLM-free novelty gate to compare against; embedding, not graph | verified (abstract) |
| ConsistencyGate (arXiv 2607.22962, Zhang & Li, 25 Jul 2026) | K LLM queries give a soft support score; admit when the mean exceeds a threshold; log-prob single-pass variant | yes, K calls | none | LoCoMo-Contam, MSC-Contam, MemContam | LLM-vote agreement as admission | verified (abstract) |
| TokenGate (OpenReview 2mNNRc1MNo, ICLR 2027 submission) | learned admission model over "twelve structured views" of each candidate, trained on 5,000 teacher-labelled LongMemEval admission examples | not stated in the abstract (a learned model trained on teacher labels; whether inference is model-free is not said) **(VERIFY)** | none | AUROC .823 vs .811 repetition control; 41.7% fewer stored memory-body tokens; LoCoMo on two test conversations vs A-MEM, Letta, Mem0 | the closest learned text-only gate; its labels would give LongMemEval admission P/R if released | verified (abstract) |
| SHM-GATE (OpenReview B4JBE7ZfwQ, ACL ARR May 2026, submission 4658) | atomic-claim structuring; gate "rejects speculation and shadows contradicted facts"; three tiers (private episodic, shared semantic graph, frozen invariants) | not stated | tiers, not scores | MA-MEMCONFLICT (250 samples); +14 Memory Consistency Score over Flat-RAG | tiered memory with a non-structural gate | verified (abstract) |
| Multi-Factor Value Model (arXiv 2606.12945, v2 20 Jun 2026) | V(m) = Σ w_i f_i(m) over emotional intensity, goal relevance, value alignment, self/user relevance, task utility, reliability, usage history | none ("single CPU, no API calls") | none | LongMemEval gold-evidence retention 0.770 ± 0.011 vs uniform 0.657, best single factor 0.518, recency 0.368 | an LLM-free linear value model with no graph factor; a baseline to reimplement | verified (abstract) |
| LRE, Learning What Not to Forget (arXiv 2606.20954, v2 3 Sep 2026) | "kilobyte-scale, CPU-only, language-model-free scorer" | none | none | best budgeted answer quality on LoCoMo reading 68% fewer tokens | LLM-free scorer baseline | verified (abstract) |
| MemReader (arXiv 2604.07877, v2 10 Apr 2026) | GRPO-trained 4B policy choosing write / defer / retrieve / discard after "explicitly evaluat[ing] information value, reference ambiguity, and completeness" | yes (the policy) | none | not recorded | "information value" as an LLM judgement, the opposite of a computed one | verified (abstract) |
| DeltaMem (arXiv 2604.01560, 2 Apr 2026) | RL-trained manager rewarded by "a novel Memory-based Levenshtein Distance" over textual memory updates | yes | none (text edit distance) | LoCoMo, HaluMem, PersonaMem | the nearest "write delta as a quantity", textual not graph | verified (abstract) |
| Experience Memory Graph (arXiv 2607.13884, Wang et al., v2 10 Aug 2026) | failure recovery as graph matching; stores graph edit paths derived offline between failed and expert trajectory graphs | not stated | graph edit paths, offline | ALFWorld, ScienceWorld | the only agent-memory use of graph edit distance found; not for admission or write deltas | verified (abstract) |
| GovMem (arXiv 2607.02579, Qi, Xu, Li, 30 Jun 2026) | promote / reject / needs-review from provenance metadata (source, prompt family, parent event, environment, trust), categorical dependency labels (independent, copied source, shared prompt/tool, same-agent echo, low trust, unknown), counterevidence, scope | not stated | none | false promotion 0.597 (raw frequency) vs 0.040 with recall 0.960 and review burden 0.151 on GovMem-Bench; 0.111 / 0.692 on 120 human-gold real-trace candidates; 0 of 133 external coding memories safe to auto-promote (114 no, 19 review) | "textual agreement alone is not a calibrated confidence signal": the direct challenge to overlap Trust | verified (full text) |
| What to Admit and How to Present (arXiv 2610.11188, Liu & Ding, 8 Oct 2026) | admission at recall time (after retrieval, before generation): scope, subject match, role, epistemic support, temporal status, utility; permission-semantic typed units | judge only | none | MemSyco-Bench, GLM-5.2 judge; cross-domain failure 42.5% → 25.0% on a DeepSeek-V4-Pro dev set (n = 200, McNemar p = 1.6e-4) | admission vocabulary at read time, not write time | verified (full text) |
| Memory Type Varies (arXiv 2610.11573, 8 Oct 2026) | fine-tuned type router plus LLM-judged pruning | yes (pruning) | none | LongMemEval-S Recall@1 59.15% vs LightMem 56.99%; LoCoMo 31.07 vs A-Mem 28.80 | uses MemoryOS only as a baseline, does not describe its heat | verified (full text) |
| TierMem (arXiv 2602.17913, 20 Feb 2026) | two tiers (summary index, immutable raw log) with a sufficiency router | router | none | LoCoMo accuracy 0.851 vs 0.873 raw-only with 54.1% fewer input tokens, 60.7% lower latency | tiered provenance without a value score; the "TierMem" GovMem cites | verified (abstract) |
| Useful Memories Become Faulty (arXiv 2605.12978, v2 29 Aug 2026) | continuous LLM consolidation | yes | none | utility first rises then falls below the no-memory baseline; GPT-5.4 fails on 54% of ARC-AGI problems it had solved without memory; retaining raw episodes doubles accuracy | evidence that consolidation itself destroys value; a Threats-to-validity citation for the consolidation operator | verified (abstract) |
| Retain or Consolidate? (arXiv 2607.17545, v2 21 Jul 2026) | budget-driven choice via an Offline Abstraction-Safety learner on pre-generation features | yes (consolidation) | none | consolidation "improves absolute accuracy by up to 48% under tight budgets" on LongMemEval; retention wins under loose budgets | why gating matters only under a token budget | verified (abstract) |
| Epistemics of Agent Memory / ConsolidationBench (arXiv 2609.33013, Annapureddy & Thamatani, 26 Sep 2026) | scores Retention, Compression, Reuse against a computable oracle; governance G1-G4 with poisoning handled by an instability signal | varies by policy | none | Consolidation Quality Score (geometric mean of retention and reuse, ε = 0.05) does not predict LoCoMo multi-hop / 2WikiMultiHopQA transfer: pooled Spearman ρ = −0.24, CI spanning zero, 2,532 answer cells, twelve policies; "lambda-prevalence coupling" (~0.61 vs ~30 forgets per seed) | a quality score that fails to predict QA deltas: the precedent for a null H2, and for pre-registering it | verified (full text) |
| Graph-Based Personalized Memory survey (arXiv 2609.08599, 8 Sep 2026) | survey of write / merge / forget signals: user feedback, functional role, semantic matching, duplicate detection, conflict detection, recency, buffering/gating | n/a | names no system using degree, centrality or connectivity for writes; "structural novelty", "graph edit distance", "information value" and al-Saffar absent | notes benchmarks "mostly observe whether the final answer is correct rather than whether the memory graph itself is correct" | documentary support for the gap | verified (HTML) |
| Self-Evolving Agents as Dynamic Graph Transformation survey (arXiv 2608.18104; `related-2026.md` dates it 10 Jun 2026, which an August-2026 identifier cannot carry, date **(VERIFY)**) | memories as typed nodes "updated through schema-constrained rewrites" | n/a | "the coupling between evolving agent state and dynamic graph topology remains underexplored" | n/a | names no method scoring candidates by structure before writing | verified (abstract) |

---

## D. Selection integrity and poisoning

### Fei et al., "Selection Integrity for LLM Graph Memory: An Accumulability Criterion for Information-Flow-Blind Retrieval" (arXiv 2606.12290, 10 Jun 2026)
What the paper shows (verified: arXiv HTML, `related-2026.md`, `measure-lineage.md`): selectors
s(i) = g(c(i, Q), σ(i; G, Q)) with a content term and a structural term; structural reliance
S_dom = 1 − Jaccard(top-k(G), top-k(G⁻)); Criterion 1 (adversarial accumulability) says the
divergence set is non-empty only if (a) "a no-source structural write can divert an Ω(1) share of
what determines top-k membership" onto nodes outside every returned item's derivation closure and
(b) that share exceeds the selected target's anchoring margin; "Reliance is not the criterion;
accumulability is." Personalized PageRank satisfies (a) always; per-item selectors, content-fixed
rerankers, RRF, vector selection and node-distance reranking fail (a) (Graphiti node distance:
S_dom 0.63 but 0/28 evictions; RRF and Mem0 S_dom 0). Sourceless structural writes to a HippoRAG 2
selector silently misdirected 28 of 499 clean-correct sandboxed ledger transfers (harm 0.0561,
37,066-node KG); Appendix A1: "six no-source RELATES_TO edges (no co-retrievable passage) demote the
gold passage to rank 1". Table IV harm on 615 rows: no defence .0537, B-SLM "Bayesian trust (post)"
.0537, B-AMemGuard "consensus self-check (post)" .0537, B-Rerank .078, AuthSelect .000; the
post-selection baselines are the authors' "own faithful post-selection re-implementations" and fail
"because every retrieved record is authentic, so they pass it". The remedy, AuthSelect, recomputes
selection on the authenticated (write-provenance-verified) subgraph and accepts the answer only if
it matches: binary, not a score, zero over-block, 1.9-3.2% latency; a four-level provenance lattice
⊥ ≤ U ≤ A ≤ T is defined. The paper does not gate candidate facts at write time ("does not detect the
attack"). Code "will be released" (anonymised repository); no URL or licence. Evaluated on LoCoMo
(about 585 clean-correct targets from 10 conversations) and, per the earlier verdicts, HotpotQA,
MuSiQue and 2Wiki. No LLM in the attack.

Correction to the earlier reading (verified: `related-2026.md`; adopted in the pre-registration's
deviation log item 14): Fei et al. never discuss overlap or agreement-based trust |E_t ∩ E_c| / |E_t|
and do not show that agreement trust is reallocatable; their result is about what determines top-k
retrieval membership under PPR. The statement that SIV's Trust, being an overlap with a writable
context graph, can be inflated by sourceless structural writes to G_c is this paper's own inference,
and H4's sourceless poisoning condition is what tests it. B-SLM re-implements SuperLocalMemory (arXiv
2603.02240, Bhardwaj, 17 Feb 2026 (date (VERIFY): a March-2026 identifier with a February date); code varun369/SuperLocalMemoryV2, AGPL-3.0-or-later), whose
published rule is per-agent and additive, t_{i+1} = t_i + δ/(1 + n·η), η = 0.01, t_0 = 1.0 in
Sec. 4.1 but Beta(2,1) = 0.667 in the evaluation, signals +0.015 / +0.01 / +0.02 and −0.02 / −0.03 /
−0.025, writes and deletes blocked below 0.3; it is not a Beta posterior per memory, and the trust
does not yet feed the re-ranker (verified: `measure-lineage.md`). Fei et al.'s own code and B-SLM
details are unpublished, so neither can be replicated exactly today. Position the result as
complementary, not competing.

### Poisoning, trust and provenance papers from the 2026 sweep (verified: `related-2026.md`)
* **CPB, Correlated Promotion Benchmark** (arXiv 2609.30813, Li et al., 25 Sep 2026): admission
  policies private-only, share-all, majority vote, LLM judge, governance (source-type) gate,
  independence vote, Mem0 store/gated, A-MemGuard, plus lineage-oracle and semantic-collapse arms,
  across Qwen3.5-9B, Gemma-3-12B, Llama-3.1-8B and Claude Opus 5; finding F6 "admission follows the
  form of the second source more than the truth of the claim"; governance-gate false adoption
  0.09/0.06/0.07/0.06 vs share-all 0.45/0.32/0.47/0.44; uncontested false beliefs asserted in
  0.98/0.99/0.99/0.97 of probes; source types web_text, register_document, "authoritative" with no
  formal tier ordering; logistic and GBT policies "need a whole episode's evidence graph, so they run
  on Static only"; code github.com/lxy1134/iclr_2027, no licence stated; no policy uses degree,
  connectivity or overlap (verified: full text). Its gold admission actions and lineage logs are
  reusable to test agreement-trust inflation under copies and paraphrases, with an SIV policy as a
  further arm.
* **Louck, Securing LLM-Agent Long-Term Memory Against Poisoning** (arXiv 2606.24322, 23 Jun 2026):
  three laundering channels (agent summarisation, trusted-tool echo, manufactured corroboration);
  T1 content/lineage defences cannot be sound under laundering, T2 write-time origin binding is
  necessary, T3 non-malleable origin-bound authority with Sybil-resistant corroboration-gated
  elevation is sufficient; TMA-NM 0% attack success vs up to 68% for existing defences across eight
  frontier models (verified: abstract). The reader's summary also credits Louck with an explicit
  provenance lattice, which its fact list attributes to Fei et al. **(VERIFY)** which paper defines
  the lattice. Any agreement-based arm in H4 should be combined with origin binding in at least one
  condition (reader's risk note).
* **FARMA / SENTINEL** (arXiv 2607.05029, 6 Jul 2026): forged reasoning entries with self-referential
  reinforcement defeat consensus-based defences including A-MemGuard, up to 100% ASR; SENTINEL's
  Reasoning Guard "structurally analyzes candidate entries for forgery using five weighted signals",
  ASR to 0% with no false positives on 326 benign traces (verified: abstract).
* **Manufactured Confidence** (arXiv 2606.29279, Kwon, 28 Jun 2026): consolidation turns hedged
  remarks into confident facts; source attribution does not matter but phrasing confidence does; "a
  single load-bearing memory is the hazard, and one redundant source restores correct decisions";
  code github.com/collapseindex/manufactured-confidence (verified: abstract). Its probes can test
  whether overlap Trust is insensitive to phrasing confidence.
* **SMSR** (arXiv 2606.12703, 10 Jun 2026): HMAC-SHA256 write-time provenance signing, query-time
  randomised memory ablation and verdict-based majority voting; a "Consistent Minority Effect" where
  a consistent adversarial answer wins string-based voting; "no provenance-free retrieval-time filter
  can certify against adaptive injection" (verified: abstract).
* **When Does Belief-Based Agent Memory Help? / Nous** (arXiv 2606.22030, Singh, v2 16 Jul 2026):
  Bayesian updating gives little benefit over last-write-wins on LoCoMo; proposes provenance-capped
  updating where "trust is bounded by source provenance rather than textual confidence"; reports a
  27.5-point gap between token-F1 and LLM-judge scores (verified: abstract).
* **ShadowMerge** (arXiv 2605.09033, v3 15 May 2026): poisons graph memory by injecting a relation
  that shares the query-activated anchor and canonicalised relation channel of benign evidence;
  93.8% average ASR on Mem0 (50.3 points above the best baseline) on PubMedQA, WebShop, ToolEmu; no
  structural-scoring defence proposed (verified: abstract). An anchoring-based score would reward
  exactly this attack; a point for H4's design.
* **MemLineage** (arXiv 2605.14421, 14 May 2026): RFC-6962 Merkle log over Ed25519-signed entries, a
  weighted derivation DAG with max-of-strong-edges propagation, a sensitive-action gate refusing
  dispatches descending from an external ancestor; zero ASR in its harness and on AgentDojo banking
  pairs (verified: abstract).
* **Stored Is Not Supported** (arXiv 2609.02127, He & Yu, 2 Sep 2026): typed provenance graph with
  five dimensions (origin, dependency lineage, epistemic role, validity, disclosure scope) and
  release-time mediation; on 24 conformance cases the flat rule released 19/19 unsafe candidates, the
  source-tag rule 18/19, typed mediation 0/19; no discrete trust tiers, no write-time admission rule
  (verified: abstract).
* **MemAudit** (arXiv 2605.23723, 22 May 2026): post hoc counterfactual memory influence score plus
  "a memory consistency graph that identifies structurally anomalous memories within the broader
  memory store"; MINJA attack success 70% → 0% (QA) and 83.3% → 0% (RAP); graph measures not
  specified in the abstract (verified: abstract).
* **A-MemGuard** (arXiv 2510.02373; ICML 2026 poster): consensus validation of reasoning paths with a
  separate lesson store; > 95% attack-success reduction; FARMA reports it fails under flooding of
  consistent forged entries. From search summaries only **(VERIFY)**.

What this means for H4 (reader's risk notes): GovMem, CPB and Manufactured Confidence all show that
repetition and agreement are poor trust signals, and CPB shows source-type gating is the strongest
simple policy; SIV's Trust = |E_t ∩ E_c| / |E_t| is an agreement measure and will be attacked as
poisonable unless at least one arm combines it with origin binding.

---

## E. Benchmarks

### LoCoMo (Maharana et al., ACL 2024; Proc. ACL Vol. 1, pp. 13851-13870, DOI 10.18653/v1/2024.acl-long.747)
Ten very long dyadic conversations with session timestamps, per-session *observations* (fact
sentences with the dialogue ids they came from), human-annotated event summaries, and 1,986
questions in five categories with gold evidence dialogue ids. Verified from the downloaded file
`locomo10.json` (https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json,
2,805,274 bytes, sha256 79fa87e9...98ff4) and the official repository (`data-locomo.md`): 10
conversations (conv-26, -30, -41, -42, -43, -44, -47, -48, -49, -50), 272 sessions (19-32 per
conversation), 5,882 turns (10-47 per session, mean 21.6), 1,986 QA, 2,541 observations, 669 event
entries; date strings "1:56 pm on 8 May, 2023" parse with `%I:%M %p on %d %B, %Y`; conv-26 carries
orphan `session_20..35_date_time` keys with no sessions, so iterate over `session_N` keys; `dia_id`
is "D{session}:{turn}", 1-indexed; 1,226 turns carry a BLIP caption, 910 an `img_url`, images are not
released; speaker names repeat across conversations ("John" in three), so key entities by
(sample_id, name); about 200K cl100k tokens of "speaker: text [+caption]" (ACL camera-ready: 16,618.1
tokens per conversation with the paper's tokenizer, placed in Table 1 by the reader's fact list and in
Table 5 by its summary, so the table number is **(VERIFY)**; Mem0 says about 26,000). QA records: `answer` is
a string except 6 ints; category 5 (446 records) carries `adversarial_answer` and no `answer` in
444/446; evidence lists contain 4 empty lists, 6 malformed strings and 3 dangling ids. Category
integer mapping in the data and in the official code: 1 = multi-hop (282), 2 = temporal (321),
3 = open-domain (96), 4 = single-hop (841), 5 = adversarial (446); the paper's prose numbers the
categories (1) single-hop ... (5) adversarial, which does not match the file (verified: ACL Table 5
counts, `task_eval/evaluation.py` branching, `evaluation_stats.py` print order, the A-MEM README,
content heuristics). The flag on the category mapping is resolved; the loader fix and the
pre-registration erratum are recorded in `00-preregistration.md` §6.1 and log item 10.

Scoring: the official `task_eval/evaluation.py` normalises (drop commas, lowercase, strip
punctuation, remove a/an/the/and), Porter-stems, and computes multiset-overlap F1; category 1 splits
prediction and gold on commas and averages over gold sub-answers the max F1 over predicted
sub-answers; categories 2-4 use plain F1 with the category-3 gold truncated before the first ";";
category 5 scores 1 iff "no information available" or "not mentioned" appears in the lowercased
output, tied to a two-option prompt "(a) Not mentioned in the conversation (b) {trap answer}";
overall = mean over all 1,986 including category 5 (verified: raw scripts). Details and the
competing conventions are in the final section of this file. The project's scorer now vendors these
rules (`05-critique-response.md` item 7).

Provenance of the observations: generated by **gpt-3.5-turbo** (`scripts/generate_observations.sh`
→ `task_eval/get_facts.py` → `generative_agents.memory_utils.get_session_facts`) with
CONVERSATION2FACTS_PROMPT and in-context examples, output `{speaker: [[observation, dia_id], ...]}`;
session summaries also by gpt-3.5-turbo (temperature 1.0, 256 tokens); event summaries are
human-annotated (verified: raw scripts, `data-locomo.md`). Note the discrepancy: `00-preregistration.md`
§6.1 and `02-design-critique.md` B(i) call the observations "GPT-4 session observations"; the official
scripts name gpt-3.5-turbo. The substantive point stands either way: the observations were produced
with the whole session visible, so the oracle arm is "zero LLM calls at the gate", not "LLM-free
extraction". Licence: the repository's LICENSE.txt is Creative Commons Attribution-NonCommercial 4.0
and the paper's App. B.2 says CC BY-NC 4.0 (verified); `00-preregistration.md` §13 currently says
"MIT-licensed per their repositories; re-check", which is wrong for LoCoMo and must be corrected
there. The arXiv version (2402.17753, single version v1, 27 Feb 2024) describes a 50-conversation,
7,512-question dataset that was never released; cite the ACL camera-ready and the GitHub file for
every statistic (verified). Published F1 (single/multi/temporal/open/adversarial/overall): ACL
Table 2, human 95.1/85.8/92.6/75.4/89.4/87.9; gpt-4-turbo 72.3/51.5/51.4/38.5/15.7/51.6; ACL Table 3,
RAG gpt-3.5-turbo + DRAGON dialog top-25 59.9/38.7/37.2/25.0/12.8/41.0 (verified). Known gold-label
noise: wrong speaker attribution, ambiguous questions, image-dependent questions whose captions lack
the answer (1,064 of 2,815 evidence references point at caption-bearing turns) (verified: Zep blog,
LoCoMo_refined, data). Relevance: the oracle extraction arm uses the observations; the evidence ids
give admission labels (repair or exclude the 6 malformed and 3 dangling ids first).

### LoCoMo variants (verified: `data-locomo.md`)
* **LoCoMo_refined** (mem-eval-suite): 1,382 questions after revising 337 samples with five human
  annotators; a stricter Qwen3-14B judge (temperature 0) requiring all gold information, no
  contradiction, no unsupported detail and correct temporal granularity; the original lenient judge
  agreed with humans 43.67% vs 86.33% for the refined judge on 300 samples; re-scored Mem0 48.91%
  (−15.56), EverMemOS 58.25% (−22.07), MemOS 63.60%. An optional secondary set and the citation for
  "the lenient judge is unreliable".
* **LoCoMo-Plus** (arXiv 2602.10715): criticises LoCoMo for equating memory with explicit factual
  recall; shows EM/F1/BLEU/ROUGE peak near reference length; finds task disclosure inflates temporal
  and adversarial scores; judge gemini-2.5-flash with three-level labels for factual/commonsense and
  strict binary labels for temporal and adversarial.
* **Kumiho benchmark runner** (kumiho-benchmarks): official Porter-stemmed F1, no judge, pinned
  gpt-4o-2024-08-06, per-category counts 841/282/321/96/446 (see Kumiho in section C).

### LongMemEval (Wu et al., ICLR 2025; arXiv 2410.10813, v1 14 Oct 2024, v2 4 Mar 2025)
500 questions, each with its own haystack of user-assistant sessions; the S variant is the smaller
haystack, M about 500 sessions and about 1.5M tokens, oracle only the evidence sessions. Verified
from the downloaded cleaned files and the repository (`data-longmemeval.md`): the official current
data is HF `xiaowu0162/longmemeval-cleaned` (MIT, created 2025-09-19; `longmemeval_oracle.json`
15,388,478 B, `longmemeval_s_cleaned.json` 277,383,467 B, `longmemeval_m_cleaned.json` 2,737,100,077 B);
the September 2025 cleaning removed 17 filler sessions from S/M (change log: 21 rows, issues "answer
interference", "info leak (no conflict)", "ambiguous reference"); the oracle file is byte-identical
to the original. Each file is a JSON list of 500 instances with keys question_id, question_type,
question, answer, question_date, haystack_dates, haystack_session_ids, haystack_sessions,
answer_session_ids. Six question types: temporal-reasoning 133, multi-session 133, knowledge-update
78, single-session-user 70, single-session-assistant 56, single-session-preference 30; the 30
abstention questions carry the `_abs` id suffix and sit inside those types (multi-session 12,
temporal 6, knowledge-update 6, single-session-user 6), so "abstention" is not a question_type value.
Dates are "%Y/%m/%d (%a) %H:%M". Turns are {role, content} with `has_answer` present only on turns of
evidence sessions (S: 10,960 turns with the key, 896 True; 235,790 without), so use
`turn.get('has_answer', False)`. S cleaned: 47.7 sessions per haystack (38-62), 493.5 turns, 10.3
turns per session; o200k_base content tokens mean 101.6k (README: about 115k Llama-3 tokens).

Gold evidence for admission P/R: session level `answer_session_ids` (890 non-abstention evidence
sessions); turn level `has_answer == True` (896 turns: 842 user, 54 assistant; single-session-
assistant evidence is on assistant turns, 51 of 56 such questions have no user-side flag); 45
non-abstention evidence sessions have no flagged turn, so turn-level recall denominators must use
flagged turns only or fall back to session level; abstention questions still carry
`answer_session_ids` and 10 True flags and must be excluded, as the official retrieval eval does;
44 questions have an evidence session dated after `question_date`, so never time-filter candidates
by question date; 13 S haystacks contain a duplicated filler session id, so align sessions, ids and
dates by position, never by id (verified). The official retrieval eval indexes user turns only with
ids `sess_id + '_' + (1-based turn index)` and reports recall_any / recall_all / NDCG@{5,10,50}.

Scorer (`src/evaluation/evaluate_qa.py`): `gpt-4o` → `gpt-4o-2024-08-06` (also gpt-4o-mini-2024-07-18
and a local llama-3.1-70b-instruct), temperature 0, max_tokens 10, label = `'yes' in
response.strip().lower()`, prompt chosen by question_type with the abstention prompt iff `_abs` in
the id; the temporal prompt does not penalise off-by-one day counts, the knowledge-update prompt
accepts previous information alongside the updated answer, the preference prompt treats the answer
as a rubric; `print_qa_metrics.py` asserts the gpt-4o-2024-08-06 label model and prints per-type,
"Task-averaged Accuracy" (mean of the six type means), "Overall Accuracy" (mean over 500) and
"Abstention Accuracy" over the 30 `_abs` items, which also stay in their type bins; judge-human
agreement (paper Table 6) 0.98 average, preference 0.90, abstention 0.97, others 1.00 (verified). The
README and scripts disagree on the log filename and the argument count. Reported LongMemEval_S
numbers are not like-for-like: Zep 63.8 / 71.2 (GPT-4o judge, original haystacks), Nemori 64.2 / 74.6
(judge unstated), LightMem 56.80-70.20 (gpt-4o-mini judge), Hindsight 83.6-91.4 (GPT-OSS-120b
judge), Memory-R1 J 55.40 / 57.80 (split and judge unstated); pre-September-2025 numbers were
measured on the original haystacks (verified). LongMemEval-V2 (451 questions over web-agent
trajectories, HF `xiaowu0162/longmemeval-v2`, May 2026) exists; v1 is the dialogue-memory setting all
compared systems report (verified). Licence MIT (repository and dataset; sources ShareGPT
Apache-2.0, UltraChat MIT).

Relevance: the primary dataset for H1 and H3; `has_answer` turns give the admission labels. The
earlier note that `paper2/data.py` defaulted to `longmemeval_oracle.json` is recorded as fixed in
`05-critique-response.md` item 8 (LongMemEval-S default); the pre-registered runs must use the S
split and should state the dataset version (cleaned), the judge (gpt-4o-2024-08-06) and which of the
two aggregations is reported.

---

## F. Surveys consulted
"Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions" (2025) lists KV-cache
eviction scores, MemoryBank forgetting and graph stores (HippoRAG, Zep, Graphiti, G-Memory) and a
proposed meta-learned memory value function; impact-of-update measurement exists only in the
model-editing literature (ripple effects, locality). "Memory in the Age of AI Agents" (2025) is
the second survey the literature map lists. Neither records a structural write policy. Author
lists of these two **(VERIFY)**; the kick-off reads did not cover them. Two 2026 surveys were read
and agree: "Graph-Based Personalized Memory for LLM Agents" (arXiv 2609.08599, 8 Sep 2026) names no
system using degree, centrality or connectivity for write decisions and does not mention al-Saffar,
"structural novelty", "graph edit distance" or "information value"; "Self-Evolving Agents as Dynamic
Graph Transformation" (arXiv 2608.18104) frames updates as schema-constrained rewrites and calls
the coupling of agent state and graph topology "underexplored" (verified: `related-2026.md`).

---

## G. Methodological assets found by the readers (for the H2 design and the appendix)
* Per-fact attribution without exhaustive leave-one-fact-out, which `models-cost.md` estimates at
  about 1.9M re-answers on LongMemEval-S: MeClear (arXiv 2609.09115; LOO screening plus sampled
  Shapley; target recall 85.9%, task recovery 82.3%), Causal Memory Policy (arXiv 2610.02070;
  reserved random context slots with self-normalised IPW; identification fails for 54% of required
  memories on LongMemEval and 67% on LoCoMo; "no aggregation available to a retention policy predicts
  a memory's value on unseen queries"), MemAudit's counterfactual influence score (verified:
  `related-2026.md`).
* Gold admission data: LoCoMo `evidence` ids and LongMemEval `has_answer` turns (verified: data);
  A-MAC's about 1,500 LoCoMo labels, whose rule is the evidence-id test (verified: code); TokenGate's
  5,000 teacher-labelled LongMemEval admission examples (release unknown); MEMAUDIT's exact
  package-oracle protocol (arXiv 2605.02199; MILP-certified optimal write set over exported Mem0,
  A-Mem and Letta stores); MemOps' gold operation traces (arXiv 2607.12893) (verified: abstracts).
  Prefer gold-evidence-derived labels over A-MAC's (reader's risk note).
* Reference scorers and harnesses: the official LoCoMo `task_eval/evaluation.py` (vendor
  `normalize_answer`, `f1_score`, `f1` and the per-category branching); the official LongMemEval
  `evaluate_qa.py` prompts and `eval_utils.py` retrieval metrics (MIT); Mem0's harness at commit
  aae5989 (Apache-2.0) for a Mem0-compatible J; the Memobase fork with its explicit category list;
  Zep's per-question grades file (1,540 items) for a per-question correctness vector of a strong
  commercial baseline; the Kumiho runner for a judge-free protocol over all 1,986 questions; the
  LoCoMo observation-generation prompt and examples if candidate facts must be re-extracted with a
  different model (verified: `data-locomo.md`, `data-longmemeval.md`).
* Baseline code to vendor: A-MAC's five extractors, loader and CV search (MIT); MemoryOS's
  `compute_segment_heat` / `compute_time_decay` (Apache-2.0); Mem0's prompts and the v1.0.2
  `_add_to_vector_store` (Apache-2.0); LightMem's `token_stats` schema and `token_monitor.py` (MIT)
  for a single accounting method (verified: `baselines-code.md`).
* Anti-hub device: EngramRAG's log scaling of topology scores (verified: `related-2026.md`).
* Adapter pattern: ForgetEval's six-method Adapter Protocol under MIT (arXiv 2606.15903) for
  plugging a gate into existing stores (verified: abstract).

---

## Scoring conventions on LoCoMo

Everything in this section is from `data-locomo.md`, whose reader downloaded `locomo10.json`, the
official repository scripts, Mem0's evaluation harness at commit aae5989, the Memobase fork and
Zep's `locomo_eval` code and grades file, and read the ACL camera-ready PDF; per-paper numbers were
read from arXiv HTML. The paper must state its own mapping explicitly, cite `task_eval/evaluation.py`
and ACL Table 5, and never place a per-category number from another paper in a table without
re-deriving its convention.

### The five category-label conventions
The released file stores `category` as an integer. ACL Table 5 gives the counts single-hop 841
(42.3%), multi-hop 282 (14.2%), temporal 321 (16.1%), open-domain 96 (4.8%), adversarial 446
(22.4%); the data has 1→282, 2→321, 3→96, 4→841, 5→446. The paper's prose numbers the categories
(1) single-hop, (2) multi-hop, (3) temporal, (4) open-domain, (5) adversarial, which is not the
file's order, and this is where the confusion starts (verified).

| Convention | 1 | 2 | 3 | 4 | Used by | Evidence |
|---|---|---|---|---|---|---|
| A, official | multi-hop | temporal | open-domain | single-hop | the released data and `task_eval/evaluation.py`; A-MEM (repo README states it); Kumiho; Nemori (its averages reproduce only with n = 321/96/282/841) | ACL Table 5 counts; `evaluation.py` scores category 1 with the comma-split "multi-hop" F1 and 2/3/4 with plain F1 ("single-hop, temporal, open-domain"); `evaluation_stats.py` prints in key order [4, 1, 2, 3, 5] = the paper's Single/Multi/Temporal/Open/Adversarial columns; 83% of category-2 questions start "When/How long", 98% of category-1 have ≥ 2 evidence ids, 44% of category-3 contain would/likely (verified) |
| B, Mem0 harness | single-hop | temporal | multi-hop | open-domain | Mem0; Memobase; Backboard/Hindsight; Memory-R1 (approximately, given its 1,307-question test split) | the Memobase fork adds `categories = ['single_hop','temporal','multi_hop','open_domain']` indexed by category − 1 (verified: raw `generate_scores.py`); Mem0's own `generate_scores.py` prints by integer only, and its overall 66.88 reproduces exactly as (67.13·282 + 55.51·321 + 51.15·96 + 72.93·841)/1540 only under this mapping (under A it would be 62.1); Mem0's Table 1 places A-MEM's published "Multi Hop 27.02 / Temporal 45.85 / Open Domain 12.14 / Single Hop 44.65" into its "Single-Hop 27.02 / Temporal 45.85 / Multi-Hop 12.14 / Open-Domain 44.65" columns (the Mem0 attribution is the reader's inference from three signals, marked unverified in the report, **(VERIFY)** against Mem0's authors); Hindsight's 89.61 reproduces as a weighted mean only with n = 282/96/841/321 (verified by arithmetic) |
| C, MemoryOS | single-hop | multi-hop | temporal | open-domain | MemoryOS | its table relabels A-MEM's and MemGPT's category 1-4 numbers in the paper's prose order; its `evalution_loco.py` prints set-based F1 by raw integer (verified) |
| D, Zep README | multi-hop | open-domain | temporal | single-hop | the getzep/zep-papers `locomo_eval` README | recomputing Zep's released grades file by integer gives cat 1 0.7411 (n = 282), cat 2 0.6604 (321), cat 3 0.6771 (96), cat 4 0.7979 (841), overall 0.7513; the README reports Multi-Hop 74.11, Open Domain 66.04, Temporal 67.71, Single-Hop 79.79, so it labels category 2 (temporal) "Open Domain" and category 3 (open-domain) "Temporal" (verified; whether this was intentional or a typo is unknown) |
| E, Memobase "Zep*" row, copied into Hindsight | single-hop | multi-hop | open-domain | temporal | the Memobase README's Zep* row and Hindsight's Zep row | 74.11 / 66.04 / 67.71 / 79.79 presented as Single/Multi/Open/Temporal, i.e. yet another assignment of Zep's four integers (verified) |

Only overall scores are label-invariant. A per-category comparison across papers must be re-derived
from the counts (841/282/321/96/446) or recomputed from released per-question files; otherwise it
compares different question sets under the same name.

### The official scorer
`task_eval/evaluation.py` (verified from the raw file): `normalize_answer` removes commas,
lowercases, strips punctuation and removes a/an/the/and; tokens are Porter-stemmed; `f1_score` is
multiset-overlap F1. Category 1 uses `f1()`, which splits prediction and gold on "," and averages,
over gold sub-answers, the max F1 over predicted sub-answers (code comment "multi-hop eval").
Categories 2, 3 and 4 use plain `f1_score` (comment "single-hop, temporal, open-domain"), with the
category-3 gold truncated before the first ";" (11 category-3 golds carry alternatives). Category 5
scores 1 iff "no information available" or "not mentioned" appears in the lowercased output; the
official prompt poses category 5 as a two-option choice "(a) Not mentioned in the conversation (b)
{trap answer}" in random order, and the code reads `qa['answer']` where the released file stores the
trap under `adversarial_answer`. The QA prompt (`task_eval/gpt_utils.py`) is "Based on the above
context, write an answer in the form of a short phrase for the following question. Answer with exact
words from the context whenever possible.", and category-2 questions get the suffix " Use DATE of
CONVERSATION to answer with an approximate date." `evaluation_stats.py` computes "Overall accuracy"
as the mean over all 1,986 questions including category 5. The paper's metric text: "For each
category, we calculate the F1 score for exact matches, following the normalization of both the
predicted and the actual ground truth answers" (verified). Refusal accuracy therefore depends
heavily on the answer prompt; most memory papers drop category 5 (1,540 questions), and the paper
must decide and state this up front (reader's risk note). The project's `evaluate.py` vendors these
rules (`05-critique-response.md` item 7); `02-design-critique.md` A2 records what the earlier
implementation got wrong.

### The Mem0 "J" protocol
Code at mem0ai/mem0 commit aae5989 (`evaluation/`, since removed from main; copied verbatim by
Memobase and Zep) (verified): judge gpt-4o-mini, temperature 0.0, `response_format json_object`,
prompt "Your task is to label an answer to a question as CORRECT or WRONG ... you should be generous
with your grading - as long as it touches on the same topic as the gold answer, it should be counted
as CORRECT ... For time related questions ... same date or time period ... CORRECT", adapted from
Packer et al. (MemGPT); category 5 skipped ("Skip category 5"), so 1,540 questions; answer model
gpt-4o-mini; the paper reports mean ± sd over 10 judge runs; "LLM Score" is a binary 0/1 per
question. Its F1 and BLEU-1 come from `evaluation/metrics/utils.py`, "Borrowed from
https://github.com/WujiangXu/AgenticMemory/blob/main/utils.py": set-based token F1 after
lowercasing and replacing . , ! ? with spaces, no stemming, no article removal, no comma-splitting;
BLEU-1 via nltk `sentence_bleu` with `SmoothingFunction().method1`. This F1 is not the official
metric. Zep's `locomo_eval` reuses the prompt with gpt-4o-mini structured output (`is_correct`),
gpt-4o-mini as answer model, `category != 5`, and ingests each turn as "speaker: text" with
`created_at` from `session_N_date_time` + " UTC" (verified). Nemori's unified judge prompt matches
Mem0's wording (verified).

### Metric heterogeneity and why only overall scores are comparable
| Metric | Systems | Notes |
|---|---|---|
| official Porter-stemmed multiset F1 with comma/semicolon handling, category 5 by the two-substring check, all 1,986 | LoCoMo paper; Kumiho | Kumiho pins gpt-4o-2024-08-06 and shows unpinned gpt-4o drifted 0.565 → 0.531 |
| A-MEM set-based unstemmed F1 and BLEU-1 | A-MEM (five categories incl. adversarial); MemoryOS (four); Memory-R1 (four, 1,307-question split); Mem0 (as F1/B1 beside J) | numerically different from the official F1 on the same outputs |
| lenient gpt-4o-mini judge, category 5 dropped | Mem0, Zep, Nemori, LightMem (overall only), Memobase | LoCoMo_refined: the lenient judge agrees with humans 43.67% of the time vs 86.33% for its strict judge on 300 samples; Kumiho's README claims judge accuracy inflates scores 1.5-2x over token F1; Nous reports a 27.5-point F1-vs-judge gap on LoCoMo |
| GPT-OSS-120B binary judge, temperature 0 | Hindsight | baseline rows copied from Backboard, not reproduced |
| Qwen3-14B strict judge on 1,382 revised questions | LoCoMo_refined (Mem0 48.91, EverMemOS 58.25, MemOS 63.60) | a different question set |
| gemini-2.5-flash, three-level and strict binary labels | LoCoMo-Plus | a different protocol and critique |

Four further reasons numbers do not transfer (verified unless flagged): published baseline rows are
often copied, not reproduced (Mem0's Table 1 reproduces A-MEM's published numbers with relabelled
columns, the reader's inference **(VERIFY)**; Hindsight copies Backboard; Memobase copies Mem0), and
Backboard's 90.00 is described in the reader's risk notes as not independently reproducible, with no
source attributed **(VERIFY)**; Mem0's own Zep run
(65.99) was disputed by Zep (75.14 ± 0.17) over role assignment, timestamp handling and sequential
search; answer models drift when unpinned (Kumiho); and gold-label noise (wrong speaker attribution,
ambiguous and image-dependent questions, 6 malformed and 3 dangling evidence ids, 4 empty evidence
lists) biases any metric computed against evidence. With 10 conversations, 96 open-domain questions
and 18 distinct speaker names, per-category intervals are wide and per-conversation clustering must
be respected (reader's risk note; the statistical plan in `02-design-critique.md` C and the
pre-registration log item 12 already treat the conversation as the LoCoMo inference unit).

Consequences for the paper: report the official stemmed F1 on all 1,986 questions with category 5
by the official substring rule as the primary LoCoMo metric, a Mem0-compatible J (gpt-4o-mini,
temperature 0, the aae5989 prompt, category 5 skipped) only as a secondary column for comparability
with Mem0-convention tables, pin the answer model snapshot and the judge version, and label every
per-category column with the data integer and n as well as the name.
