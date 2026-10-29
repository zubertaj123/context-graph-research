# Reader: related:2026

SCOPE AND METHOD. Fresh search for May–Oct 2026 work that could close the gap of "al-Saffar semantic information value (SIV = Trust × Impact, graph-edit delta) as an LLM-free admission/consolidation signal for agent memory graphs". Sources actually read: arXiv abstract pages and HTML full texts (via WebFetch), the arXiv export API (3 queries), OpenReview api2 search (5 queries), Google Scholar (1 query), two curated lists (DEEP-PolyU Awesome-GraphMemory; AgentMemoryWorld Awesome-Agent-Memory papers/2026.md, which covers only to 2026-07-21), and the al-Saffar iiWAS'10 and MemoryOS PDFs extracted with pdftotext. Semantic Scholar API returned HTTP 429 on all three attempts, so it contributed nothing. About 90 candidate papers were screened against four criteria: (a) graph-structural scoring of candidate facts before admission; (b) a write's realized change measured as a graph delta; (c) connected-vs-disconnected priors; (d) agreement-trust poisoning with provenance tiers.

HEADLINE. As of 2026-10-18, no paper found satisfies (a), (b) or (c). Criterion (d) is partially covered by several security/benchmark papers, none of which uses a graph-structural admission score. The paper's gap is still open, but its framing of two objections must be corrected (below).

NEAREST NEIGHBOURS BY CRITERION.
(a) Structure used for retention/consolidation, never for admission: Selective Forgetting (2608.28978) prunes nodes on "recency, access frequency, degree centrality, and age" and finds the KG does not beat a flat vector baseline on LongMemEval; EngramRAG (2609.32049) sets decay λ_eff(v)=λ_base·exp(−2.0·Score_topology(v)) where Score_topology combines centrality, degree and a type prior, but the write path is a cosine>0.95 dedup; STRATA (2610.19386) protects Tarjan articulation points during garbage collection and admits via a scalar similarity band (noop/add_bullet/create), with an ablation showing articulation protection lifts success 69.8→72.9; HeLa-Mem (2604.16839) flags hubs by summed edge weight D(v)=Σw_ij for distillation only. The only admission-time structural rule is Beyond Freshness (ICLR'27 submission Ulqx7pVvcS), which uses transitive dependency closure to quarantine invalidated memories — reachability, not novelty/impact. Fei et al. (2606.12290) use structure (PPR) at retrieval, not writing. GAM (2604.12285) explicitly rejects a numeric graph-distance threshold: "the LLM serves as a neural proxy for the thresholding function."
(b) No paper measures a write's realized graph delta and correlates it with per-question accuracy. DeltaMem (2604.01560) uses a "Memory-based Levenshtein Distance" as an RL reward over textual memory updates; Experience Memory Graph (2607.13884) derives graph edit paths offline between failed/expert trajectory graphs; the dynamic-graph-transformation survey (2608.18104) frames memory updates as "schema-constrained rewrites" without measuring them. STRATA, EngramRAG and HeLa-Mem report only aggregate graph counts.
(c) Nothing. The only statement of a disconnected-component prior remains al-Saffar's own cost-1 rule.
(d) Partial. CPB (2609.30813) shows copies/paraphrases inflate apparent support ("admission follows the form of the second source more than the truth of the claim"); its source-type gate cuts false adoption to 0.06–0.09 but uses labels (web_text, register_document, authoritative), not formal tiers, and no store-level graph metric. GovMem (2607.02579) uses categorical dependency labels, says "textual agreement alone is not a calibrated confidence signal", and found 0 of 133 external coding memories safe to auto-promote. Louck (2606.24322) is the only paper with an explicit provenance lattice ⊥≤U≤A≤T and a "manufactured corroboration" laundering channel with Sybil-resistant corroboration-gated elevation. FARMA (2607.05029) defeats consensus-based A-MemGuard; Nous (2606.22030) proposes provenance-capped updating; SMSR (2606.12703) names a "Consistent Minority Effect" in voting.

RE-CHECKS. GovMem, Epistemics/ConsolidationBench (2609.33013), CPB and What to Admit (2610.11188) contain no graph-structural admission feature (full HTML read). ConsolidationBench scores Retention/Compression/Reuse against an oracle; its CQS does not predict LoCoMo/2Wiki transfer (Spearman −0.24, CI spans 0). What-to-Admit's FC factors are scope, subject match, role, epistemic support, temporal status, utility, applied at recall time.

CORRECTIONS TO THE PAPER'S OBJECTION LIST.
1. Fei et al. 2026 do NOT show agreement-trust is reallocatable. Their criterion is that a "no-source structural write can divert an Ω(1) share of what determines top-k membership"; PPR is exposed, Graphiti node-distance immune ("Reallocatability, not reliance"). They never discuss overlap trust |Et∩Ec|/|Et|. The paper should restate: SIV's Trust is an overlap with the writable context graph, so sourceless structural writes to Gc can inflate later candidates' Trust — an inference the paper must test itself.
2. A-MAC's 0.583 is confirmed as admission-decision F1 on "LoCoMo test set (N=225)" with ground-truth admission labels (A-mem baseline 0.541 = precision 0.371/recall 1.000); label provenance is undocumented.
3. MemoryOS heat (for the baseline) is Heat=α·Nvisit+β·Linteraction+γ·Rrecency with α=β=γ=1, R=exp(−Δt/μ), μ=1e7 s, promotion threshold τ=5.
4. al-Saffar exact definitions: SIVa=(|Ec'|−|Ec|)/|Ec'|; SIVp edge weights w=t·degree(v) for edges touching overlap vertices else 1; Trust=|Et∩Ec|/|Et|; Impact=Σdeg(overlap vertices)/Σdeg(all Gc vertices); SIVp=Trust×Impact ("We use multiplication"); Gc'=Gt∪Gc. The single-fact Trust degeneracy and speaker-hub dominance of Impact are therefore intrinsic to the published formulas.

METHODOLOGICAL ASSETS FOUND. Per-fact attribution without full leave-one-fact-out: MeClear (2609.09115, LOO screening + sampled Shapley), Causal Memory Policy (2610.02070, reserved random context slots with IPW; identification fails for 67% of required memories on LoCoMo), MemAudit counterfactual influence score. Gold admission data: TokenGate's 5,000 teacher-labelled LongMemEval admission examples (OpenReview 2mNNRc1MNo); A-MAC's ~1,500 LoCoMo labels; MEMAUDIT exact package oracle (2605.02199); MemOps gold operation traces (2607.12893). Learned-gate baselines to reimplement: SAGE vMF novelty gate (2605.30711), LRE LLM-free scorer (2606.20954), Multi-Factor Value Model (2606.12945, seven non-graph factors, CPU-only).

WHAT REMAINS OPEN (2026-10-18). (1) Any admission or consolidation gate that scores a candidate triple by structural novelty/impact on the existing memory graph; (2) any measurement of realized per-write graph change (edges added, components merged, degree shift) correlated with per-question QA deltas on LoCoMo/LongMemEval; (3) any test of connected-vs-disconnected candidate priors; (4) an evaluation of overlap-based (agreement) trust under sourceless structural writes with tiered provenance; (5) an LLM-free admission policy compared at matched memory size against admit-all, random, MemoryOS heat, A-MAC, Mem0 ADD/UPDATE/DELETE/NOOP and Memory-R1 gates. The field's closest work treats structure as a retention/decay or retrieval signal and treats trust as source-typing; the write-time structural-value question is unoccupied.

## Facts (each with source and verification flag)

- [verified] al-Saffar & Heileman (iiWAS 2010) define Trust(Gt,Gc)=|Et∩Ec|/|Et| (eq. 8), Impact(Gt,Gc)=Σ_i d_i / Σ_i d'_i with d_i=degree of overlap vertices in Gc and d'_i=degree of all Gc vertices (eq. 9), SIVp=f(Impact,Trust) with f=multiplication (eq. 10), Gc'=Gt∪Gc (eq. 6), and SIVa=(|Ec'|−|Ec|)/|Ec'| (eq. 5); SIVp weights edges touching overlap vertices by t×degree and gives cost 1 to edges of disconnected components.  
  source: https://hpc.pnl.gov/people/sinan/publications/al-Saffar_InfValue-iiWAS10.pdf
- [verified] al-Saffar's paper states the intuition 'learning of new unknown (disconnected) concepts is not as important as learning related to already known concepts which in turn is less important than learning concepts that are centrally connected' and computes SIVa only when inference rules l are known (Gc'=l(Gc,Gt)); otherwise SIVp is a prediction using union plus structural weights.  
  source: https://hpc.pnl.gov/people/sinan/publications/al-Saffar_InfValue-iiWAS10.pdf
- [verified] GovMem (arXiv 2607.02579, Qi, Xu, Li, submitted 30 Jun 2026) outputs promote/reject/needs-review from provenance metadata (source, prompt family, parent event, environment, trust), dependency-aware support with categorical labels (independent, copied source, shared prompt/tool, same-agent echo, low trust, unknown), counterevidence and scope; it defines no graph or graph-structural quantity and does not measure writes as graph deltas.  
  source: https://arxiv.org/html/2607.02579
- [verified] GovMem reports false promotion 0.597 (raw frequency) vs 0.040 (GovMem-verified) with recall 0.960 and review burden 0.151 on synthetic GovMem-Bench; on 120 human-gold real-trace candidates held-out false promotion 0.111 with review burden 0.692; on a 133-row external coding-agent packet human adjudication found no safe automatic promotions (114 no, 19 review). It states 'textual agreement alone is not a calibrated confidence signal'.  
  source: https://arxiv.org/html/2607.02579
- [verified] The Epistemics of Agent Memory (arXiv 2609.33013, Annapureddy & Thamatani, 26 Sep 2026) introduces ConsolidationBench scoring Retention, Compression and Reuse against a computable oracle; its Consolidation Quality Score (geometric mean of retention and reuse with ε=0.05 floor) does not predict real transfer accuracy on LoCoMo multi-hop and 2WikiMultiHopQA (pooled Spearman ρ=−0.24, CI spanning zero, 2,532 answer cells, twelve policies); no graph-structural measure is used anywhere.  
  source: https://arxiv.org/html/2609.33013
- [verified] ConsolidationBench Phase 4 governance covers G1 poison-resistance, G2 reversibility, G3 auditability and G4 quality gate; poisoning is handled by an instability signal, not provenance tiers or agreement voting; 'lambda-prevalence coupling' is a distribution-shift failure where a hindsight-regret hyperparameter tuned at one junk prevalence collapses the forgetting gate at another (~0.61 forgets per seed vs ~30).  
  source: https://arxiv.org/html/2609.33013
- [verified] CPB — A Benchmark and Diagnostic Study of Epistemic Admission in Shared Agent Memory (arXiv 2609.30813, Li et al., 25 Sep 2026) — evaluates admission policies (private only, share all, majority vote, LLM judge, governance/source-type gate, independence vote, Mem0 store/gated, A-MemGuard, plus lineage-oracle and semantic-collapse diagnostic arms) across Qwen3.5-9B, Gemma-3-12B, Llama-3.1-8B and Claude Opus 5; no policy uses degree, connectivity or overlap over the shared store, and no graph delta is measured.  
  source: https://arxiv.org/html/2609.30813
- [verified] CPB finding F6: 'admission follows the form of the second source more than the truth of the claim'; governance (source-type) gate false adoption 0.09/0.06/0.07/0.06 vs share-all 0.45/0.32/0.47/0.44; uncontested false beliefs asserted in 0.98/0.99/0.99/0.97 of probes; source types shown are web_text, register_document and 'authoritative' typing with no formal tier ordering; code at github.com/lxy1134/iclr_2027 with no license stated. Logistic regression and GBT policies 'need a whole episode's evidence graph, so they run on Static only'.  
  source: https://arxiv.org/html/2609.30813
- [verified] What to Admit and How to Present (arXiv 2610.11188, Liu & Ding, 8 Oct 2026) applies admission after retrieval and before generation (recall time, not write time); factor-compiled admission uses scope, subject match, role, epistemic support, temporal status and utility; permission-semantic admission assigns kind, subject, epistemic status, temporality to typed units; evaluated on MemSyco-Bench with GLM-5.2 judge; tightening admission cut cross-domain failure 42.5%→25.0% on a DeepSeek-V4-Pro dev set (n=200, McNemar p=1.6e-4); no graph measure or provenance tier.  
  source: https://arxiv.org/html/2610.11188v1
- [verified] A-MAC (arXiv 2603.04549, Zhang et al., 4 Mar 2026; ICLR 2026 MemAgents workshop) scores S(m)=Σ w_i·feature over Utility (one LLM call), Confidence (max ROUGE-L over supporting turns), Novelty (1−max cosine, Sentence-BERT), Recency (exp(−0.01·hours)) and Type Prior (POS pattern rules); weights/threshold learned by maximizing F1 via 5-fold CV on ground-truth admission labels; Table 1 'Performance comparison on LoCoMo test set (N=225)' reports A-MAC F1 0.583 vs A-mem 0.541 (precision 0.371, recall 1.000, latency 3831 ms); the F1 is over admission decisions, not QA; LoCoMo provides 'approximately 1,500 candidate memories with ground-truth admission labels' whose construction is not explained.  
  source: https://arxiv.org/html/2603.04549v1
- [verified] Fei et al., Selection Integrity for LLM Graph Memory (arXiv 2606.12290, Zeming Fei et al., 10 Jun 2026): exposure criterion requires that 'a no-source structural write can divert an Ω(1) share of what determines top-k membership' past a selected fact's margin; PPR is exposed, Graphiti node-distance immune (S_dom 0.63 vs 0.24; 0/28 evictions), RRF and Mem0 immune; 28 misdirected sandboxed ledger transfers over 499 clean-correct rows (harm 0.0561); AuthSelect recomputes selection on the authenticated subgraph with zero over-block and 1.9–3.2% latency; a four-level provenance lattice ⊥≤U≤A≤T is defined; the paper never discusses overlap/agreement-based trust and does not gate candidate facts at write time ('does not detect the attack').  
  source: https://arxiv.org/html/2606.12290
- [verified] Fei et al. Appendix A1 reports that 'six no-source RELATES_TO edges (no co-retrievable passage) demote the gold passage to rank 1'; related work mentions A-MemGuard 'uses consensus validation and a dual-memory design' and SuperLocalMemory 'combines per-agent provenance, Bayesian trust, and a Leiden KG layer'; code 'will be released' (anonymized repo for review), no URL or license given.  
  source: https://arxiv.org/html/2606.12290
- [verified] Securing LLM-Agent Long-Term Memory Against Poisoning (arXiv 2606.24322, Louck, 23 Jun 2026) identifies three laundering channels (agent summarization, trusted-tool echo, manufactured corroboration), proves content/lineage defenses cannot be sound under laundering (T1), write-time origin binding is necessary (T2), and non-malleable origin-bound authority with Sybil-resistant corroboration-gated elevation is sufficient (T3); TMA-NM reports 0% attack success vs up to 68% for existing defenses across eight frontier models.  
  source: https://arxiv.org/abs/2606.24322
- [verified] Selective Forgetting: A Graph-Based Memory Framework (arXiv 2608.28978, Rusu, Khanzadeh, Alalfi, 29 Aug 2026) 'periodically prunes nodes that score low on a weighted combination of recency, access frequency, degree centrality, and age'; the graph did not beat a flat vector baseline on LongMemEval; forgetting removed ~9.8% of nodes with no change in token F1; degree is used for forgetting, not admission.  
  source: https://arxiv.org/abs/2608.28978
- [verified] EngramRAG (arXiv 2609.32049, Potineni et al., 25 Sep 2026): write path dedups by cosine similarity >0.95 (increments access count); Hebbian edge weights W_ij=0.4·cos+0.4·Σexp(−λ_access·Δt)+0.2·R_type; U-PPR with d=0.85 and teleport mixing θ=0.6; decay λ_eff(v)=λ_base(type)·exp(−2.0·Score_topology(v)) where Score_topology combines centrality, degree and type prior; LoCoMo Recall@5 53.21% vs dense 38.29%, MRR 0.4203 vs 0.2937; SUPERSEDES filtering cut split-brain hallucinations 70%→0%; no per-write structural change is reported and no ablation isolates topology decay vs recency on LoCoMo QA; log scaling used to avoid 'rich-get-richer' hub capture.  
  source: https://arxiv.org/html/2609.32049
- [verified] STRATA (arXiv 2610.19386, Zhu et al., 14 Oct 2026): admission uses a scalar similarity band ('noop if s>τ_high', add_bullet in [τ_low,τ_high], else create), not graph structure; GC deletes lowest-Q nodes down to cap·(1−0.15) excluding Tarjan cut vertices; Louvain at resolution 1.5 induces meta-frameworks promoted if conductance <0.6 and novelty distance ≥0.6; ablation: articulation protection 69.8→72.9 success, regret 13.2%→4.8%, bridges kept 0→16; removing the novelty gate raises meta churn 3.1%→28.6%; no per-write graph delta reported.  
  source: https://arxiv.org/html/2610.19386
- [verified] HeLa-Mem (arXiv 2604.16839, ACL 2026): hubs flagged for Hebbian distillation when associative strength D(v_i)=Σ_j w_ij exceeds δ_hub (value not given); write path only links consecutive turns with small weights; LoCoMo F1 (GPT-4o-mini) multi-hop 40.14, temporal 47.29, open-domain 29.70, single-hop 51.89 vs MemoryOS reproduced 38.39/41.58/23.75/45.86; Table 3 'Full' row (36.04/46.23) is inconsistent with Table 1.  
  source: https://arxiv.org/html/2604.16839
- [verified] GAM (arXiv 2604.12285v1) states 'Since explicitly computing the high-dimensional distance Δ is computationally expensive and sensitive to lexical noise' and 'the LLM serves as a neural proxy for the thresholding function'; consolidation triggers are session end, 2048-token buffer overflow and natural pauses; linking uses top-5 vector candidates then LLM relation confidence >τ; LoCoMo avg F1 40.00 (Qwen2.5-7B) vs Mem0 35.38; 43.14 with GPT-4o-mini.  
  source: https://arxiv.org/html/2604.12285v1
- [verified] Beyond Freshness: Closure-Aware Admission for Reliable Agent Memory (OpenReview forum Ulqx7pVvcS, ICLR 2027 submission, anonymous, cdate ≈ Sep 2026) 'combines task-constraint validation with transitive dependency closure to quarantine inadmissible memories', notes direct version checks 'can miss transitive invalidation', and matches full transitive-closure decisions on SQLite and Mem0 workloads; no numeric results in the abstract.  
  source: https://api2.openreview.net/notes/search?term=Closure-Aware+Admission+Reliable+Agent+Memory&source=all&limit=5
- [verified] TokenGate: Write Admission and Budgeted Retrieval (OpenReview forum 2mNNRc1MNo, ICLR 2027 submission) trains an admission model on 'twelve structured views of each source candidate' using 5,000 teacher-labelled admission examples from LongMemEval; AUROC .823 vs .811 repetition control; 41.7% reduction in stored memory-body tokens; evaluated on LoCoMo (two test conversations) with A-MEM, Letta, Mem0; no graph-structural feature.  
  source: https://api2.openreview.net/notes/search?term=TokenGate+Write+Admission+Budgeted+Retrieval&source=all&limit=5
- [verified] SHM-GATE (OpenReview forum B4JBE7ZfwQ, ACL ARR May 2026 submission 4658) uses atomic-claim structuring, an admission gate that 'rejects speculation and shadows contradicted facts', and three tiers (private episodic, shared semantic graph, frozen invariants); MA-MEMCONFLICT 250 samples; +14 Memory Consistency Score over Flat-RAG; mechanism of the gate is not graph-structural in the abstract.  
  source: https://api2.openreview.net/notes/search?term=SHM-GATE+Structured+Hierarchical+Memory+Gated+Admission&source=all&limit=5
- [verified] SAGE: A Novelty Gate (arXiv 2605.30711, Wang, Brahma, Henao, v2 18 Jun 2026) 'scores candidate facts with a von Mises-Fisher-based density estimator over memory embeddings' and 'routes them with an adaptive threshold that tracks memory-store geometry' into ADD/skip/LLM-merge; gains over Mem0 on LoCoMo with GPT-4o-mini; embedding-based, not graph-based.  
  source: https://arxiv.org/abs/2605.30711
- [verified] ConsistencyGate (arXiv 2607.22962, Zhang & Li, 25 Jul 2026) 'queries the LLM K times for a soft support score and admits m only when the average exceeds a threshold', with a single-forward-pass log-prob variant; benchmarks LoCoMo-Contam, MSC-Contam, MemContam; requires an LLM for the gate.  
  source: https://arxiv.org/abs/2607.22962
- [verified] DeltaMem (arXiv 2604.01560, 2 Apr 2026) introduces 'a novel Memory-based Levenshtein Distance to formalize the memory updating reward' for RL training of a memory manager on LoCoMo, HaluMem, PersonaMem; the distance is over textual memory updates, no graph is mentioned.  
  source: https://arxiv.org/abs/2604.01560
- [verified] Experience Memory Graph (arXiv 2607.13884, Wang et al., v2 10 Aug 2026) 'reformulates agent failure recovery as a graph matching problem' and stores 'graph edit paths that explicitly indicate how to correct failures' derived offline between failed and expert trajectory graphs (ALFWorld, ScienceWorld); edit distance is not used for admission or to measure write deltas.  
  source: https://arxiv.org/abs/2607.13884
- [verified] Self-Evolving Agents as Dynamic Graph Transformation survey (arXiv 2608.18104, 10 Jun 2026) models memories as typed nodes 'updated through schema-constrained rewrites' and says 'the coupling between evolving agent state and dynamic graph topology remains underexplored'; it names no method scoring candidate memories by graph structure before writing.  
  source: https://arxiv.org/abs/2608.18104
- [verified] Graph-Based Personalized Memory for LLM Agents survey (arXiv 2609.08599, 8 Sep 2026) lists write/merge/forget signals (user feedback, functional role, semantic matching, duplicate detection, conflict detection, recency, buffering/gating) and names no system using degree, centrality or connectivity for write decisions; 'structural novelty', 'graph edit distance', 'information value' and al-Saffar do not appear; it notes benchmarks 'mostly observe whether the final answer is correct rather than whether the memory graph itself is correct'.  
  source: https://arxiv.org/html/2609.08599
- [verified] MemoryOS (arXiv 2506.06326, Kang, Ji, Zhao, Bai): Heat = α·N_visit + β·L_interaction + γ·R_recency (Eq. 4) with R_recency = exp(−Δt/μ), μ=1e7 seconds; coefficients in Eq. 4 'are equality set to 1'; segments with heat > τ=5 are transferred to long-term persona memory; lowest-heat segments evicted when MTM exceeds 200 segments; STM queue length 7; User KB and Agent Traits FIFO capacity 100; L_interaction reset to zero after transition.  
  source: https://arxiv.org/pdf/2506.06326
- [verified] Memory-R1 (arXiv 2508.19828, Yan et al., v5 14 Jan 2026) trains 'a Memory Manager that learns structured operations, including ADD, UPDATE, DELETE, and NOOP' and an Answer Agent with PPO and GRPO using 'only 152 training QA pairs', evaluated on LoCoMo, MSC and LongMemEval; no graph-structural feature.  
  source: https://arxiv.org/abs/2508.19828
- [verified] MeClear (arXiv 2609.09115, Yang et al., 8 Sep 2026) 'combines Leave One Out screening with sampled cooperative Shapley attribution' to find memories with negative downstream utility; target recall 85.9%, task recovery 82.3% (+25.5 pp over LOO) over ten long dialogue memory pools; no graph measure.  
  source: https://arxiv.org/abs/2609.09115
- [verified] Causal Memory Policy (arXiv 2610.02070, Behnam & Wang, v3 8 Oct 2026) reserves context slots for memories sampled with known propensities and estimates utility by self-normalized IPW; identification fails for 54% of required memories on LongMemEval and 67% on LoCoMo; required-vs-non-required AUC 0.54→0.66; 'no aggregation available to a retention policy predicts a memory's value on unseen queries'.  
  source: https://arxiv.org/abs/2610.02070
- [verified] MemAudit (arXiv 2605.23723, 22 May 2026) is post-hoc: a counterfactual memory influence score plus 'a memory consistency graph that identifies structurally anomalous memories within the broader memory store'; MINJA attack success 70%→0% (QA) and 83.3%→0% (RAP); graph measures not specified in the abstract.  
  source: https://arxiv.org/abs/2605.23723
- [verified] When Does Belief-Based Agent Memory Help? (arXiv 2606.22030, Singh, v2 16 Jul 2026): Bayesian updating gives little benefit over last-write-wins on LoCoMo; proposes provenance-capped updating where 'trust is bounded by source provenance rather than textual confidence'; reports a 27.5-point gap between token-F1 and LLM-judge scores.  
  source: https://arxiv.org/abs/2606.22030
- [verified] Forged Reasoning Attacks (arXiv 2607.05029, 6 Jul 2026): FARMA uses 'self-referential reinforcement that defeats consensus-based defenses' including A-MemGuard, up to 100% ASR; SENTINEL's Reasoning Guard 'structurally analyzes candidate entries for forgery using five weighted signals', reducing ASR to 0% with no false positives on 326 benign traces.  
  source: https://arxiv.org/abs/2607.05029
- [verified] Manufactured Confidence (arXiv 2606.29279, Kwon, 28 Jun 2026) finds consolidation turns hedged remarks into confident facts; source attribution does not matter but phrasing confidence does; 'a single load-bearing memory is the hazard, and one redundant source restores correct decisions'; code at github.com/collapseindex/manufactured-confidence.  
  source: https://arxiv.org/abs/2606.29279
- [verified] SMSR (arXiv 2606.12703, 10 Jun 2026) combines HMAC-SHA256 write-time provenance signing with query-time randomized memory ablation and verdict-based majority voting; describes a 'Consistent Minority Effect' where a consistent adversarial answer wins string-based voting; 'no provenance-free retrieval-time filter can certify against adaptive injection'.  
  source: https://arxiv.org/abs/2606.12703
- [verified] Learning What to Remember: Multi-Factor Value Model (arXiv 2606.12945, v2 20 Jun 2026) scores V(m)=Σ w_i f_i(m) over emotional intensity, goal relevance, value alignment, self/user relevance, task utility, reliability, usage history (no graph factors); blind-regime gold-evidence retention on LongMemEval 0.770±0.011 vs uniform 0.657, best single factor 0.518, recency 0.368; 'all experiments run on a single CPU with no API calls'.  
  source: https://arxiv.org/abs/2606.12945
- [verified] Learning What Not to Forget (arXiv 2606.20954, v2 3 Sep 2026) LRE is 'a kilobyte-scale, CPU-only, language-model-free scorer'; on LoCoMo gives best budgeted answer quality reading 68% fewer tokens; no graph measure.  
  source: https://arxiv.org/abs/2606.20954
- [verified] MEMAUDIT package-oracle protocol (arXiv 2605.02199, 4 May 2026) turns 'write-time memory selection into a finite auditable optimization problem with a certified denominator' (concave-over-modular coverage, branch-and-bound with MILP certification) over exported Mem0, A-Mem and Letta stores; LoCoMo/LongMemEval not mentioned; no graph measure.  
  source: https://arxiv.org/abs/2605.02199
- [verified] MemOps (arXiv 2607.12893, 14 Jul 2026) benchmarks remembering, forgetting, updating, reflecting with gold operation traces (trigger, target, scope, state transition, evidence); does not use ADD/UPDATE/DELETE/NOOP terms and does not mention LoCoMo/LongMemEval.  
  source: https://arxiv.org/abs/2607.12893
- [verified] MemReader (arXiv 2604.07877, v2 10 Apr 2026) MemReader-4B 'explicitly evaluates information value, reference ambiguity, and completeness before acting' via a GRPO-trained LLM policy (write/defer/retrieve/discard); 'information value' is LLM-judged, no graph.  
  source: https://arxiv.org/abs/2604.07877
- [verified] Useful Memories Become Faulty When Continuously Updated by LLMs (arXiv 2605.12978, v2 29 Aug 2026): 'As consolidation proceeds, memory utility first rises, then degrades, and can fall below the no-memory baseline'; GPT-5.4 'fails on 54% of a set of ARC-AGI problems it had previously solved without memory'; retaining raw episodes doubles accuracy vs forced consolidation.  
  source: https://arxiv.org/abs/2605.12978
- [verified] Retain or Consolidate? (arXiv 2607.17545, v2 21 Jul 2026): consolidation 'improves absolute accuracy by up to 48% under tight budgets' on LongMemEval while retention wins under loose budgets; decision driven by budget pressure via Offline Abstraction-Safety learner on pre-generation features; no graph measure.  
  source: https://arxiv.org/abs/2607.17545
- [verified] ShadowMerge (arXiv 2605.09033, v3 15 May 2026) poisons graph memory by injecting a relation sharing the same query-activated anchor and canonicalized relation channel as benign evidence; 93.8% average ASR on Mem0 (50.3 points above best baseline) on PubMedQA, WebShop, ToolEmu; no structural-scoring defense proposed.  
  source: https://arxiv.org/abs/2605.09033
- [verified] MemLineage (arXiv 2605.14421, 14 May 2026) attaches an RFC-6962 Merkle log over Ed25519-signed entries and a weighted derivation DAG with max-of-strong-edges propagation; a sensitive-action gate refuses dispatches descending from an external ancestor; zero ASR in its harness and on AgentDojo banking pairs.  
  source: https://arxiv.org/abs/2605.14421
- [verified] Control-Plane Placement Shapes Forgetting (arXiv 2606.15903, v2 16 Jun 2026) releases ForgetEval (1,000 templated + 385 adversarial cases) and a six-method Adapter Protocol under MIT; mutation-time LLM hook reaches 91.7–93.2% overall; no graph measure.  
  source: https://arxiv.org/abs/2606.15903
- [verified] TierMem — From Lossy to Verified (arXiv 2602.17913, 20 Feb 2026) uses two tiers (summary index, immutable raw-log store) with a sufficiency router; LoCoMo accuracy 0.851 vs 0.873 raw-only with 54.1% fewer input tokens and 60.7% lower latency; no graph measure; it is the 'TierMem' GovMem cites for tiered provenance.  
  source: https://arxiv.org/abs/2602.17913
- [verified] Stored Is Not Supported (arXiv 2609.02127, He & Yu, 2 Sep 2026) defines a typed provenance graph with five dimensions (origin, dependency lineage, epistemic role, validity, disclosure scope) and release-time mediation; on 24 conformance cases the flat rule released 19/19 unsafe candidates, source-tag rule 18/19, typed mediation 0/19 unqualified; no discrete trust tiers and no write-time admission rule.  
  source: https://arxiv.org/abs/2609.02127
- [verified] Google Scholar query '"graph edit distance" "agent memory" admission 2026' (since 2026) returned five results, of which only Experience Memory Graph (Wang et al. 2026) uses graph edit distance in an agent-memory setting; none uses it for admission.  
  source: https://scholar.google.com/scholar?q=%22graph+edit+distance%22+%22agent+memory%22+admission+2026&as_ylo=2026
- [verified] arXiv export API query all:"graph edit distance" AND all:memory AND all:agent returned opensearch:totalResults = 0.  
  source: http://export.arxiv.org/api/query?search_query=all:%22graph%20edit%20distance%22%20AND%20all:memory%20AND%20all:agent&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending
- [verified] Memory Type Varies (arXiv 2610.11573, 8 Oct 2026) does not describe the MemoryOS heat score (MemoryOS appears only as a baseline) and uses a fine-tuned type router plus LLM-judged pruning, no graph-structural write signal; LongMemEval-S Recall@1 59.15% vs LightMem 56.99%, LoCoMo 31.07 vs A-Mem 28.80.  
  source: https://arxiv.org/html/2610.11573
- [unverified] A-MemGuard (arXiv 2510.02373; ICML 2026 poster) pairs consensus validation of reasoning paths with a separate lesson store and reports >95% attack-success reduction; a later paper (arXiv 2607.05029) says it fails against flooding of consistent forged entries. (From search summaries, not a direct fetch.)  
  source: https://arxiv.org/html/2510.02373
- [unverified] The MemoryOS GitHub repository (BAI-LAB/MemoryOS) is reported as Apache-2.0 licensed and EMNLP 2025 Oral. (From search summary, not a direct fetch.)  
  source: https://github.com/BAI-LAB/MemoryOS
- [unverified] Ori mnemos v0.4 is cited in a secondary list as using ACT-R vitality and Tarjan structural protection for retention, not admission. (From search summary only.)  
  source: https://github.com/tfatykhov/awesome-agent-memory

## reusable_assets

[
  {
    "what": "al-Saffar & Heileman, 'Computing Information Value from RDF Graph Properties' (iiWAS 2010) \u2014 primary source with eqs. 5\u201310 for SIVa, Trust, Impact, SIVp",
    "url": "https://hpc.pnl.gov/people/sinan/publications/al-Saffar_InfValue-iiWAS10.pdf",
    "how_to_use": "Cite as the definitional source; implement Trust=|Et\u2229Ec|/|Et|, Impact=\u03a3deg(overlap)/\u03a3deg(Gc), SIVp=Trust\u00d7Impact, SIVa=(|Ec'|\u2212|Ec|)/|Ec'| exactly; quote the disconnected-component cost-1 rule for the connected-vs-disconnected prior test. Extracted text is at /tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/alsaffar_pdf/alsaffar.txt",
    "license": "ACM/iiWAS copyright; author-hosted PDF"
  },
  {
    "what": "CPB code and data (Correlated Promotion Benchmark, arXiv 2609.30813)",
    "url": "https://github.com/lxy1134/iclr_2027",
    "how_to_use": "Reuse CPB-Static gold admission actions and CPB-Live lineage logs to test agreement-trust inflation under copies/paraphrases and to add a graph-structural (SIV) admission policy as a ninth arm",
    "license": "Not stated on the paper page; check repository"
  },
  {
    "what": "A-MAC paper (arXiv 2603.04549) \u2014 five-feature linear admission score with 5-fold CV on LoCoMo gold admission labels (N=225 test)",
    "url": "https://arxiv.org/html/2603.04549v1",
    "how_to_use": "Reimplement the logistic/linear gate exactly as described (ROUGE-L confidence, Sentence-BERT novelty, exp(\u22120.01h) recency, POS type prior, one LLM utility call); report admission F1 separately from QA accuracy and note the 0.583 is admission-label F1",
    "license": "arXiv preprint; no code link found"
  },
  {
    "what": "MemoryOS paper (arXiv 2506.06326) and repository \u2014 heat formula with \u03b1=\u03b2=\u03b3=1, \u03bc=1e7 s, \u03c4=5",
    "url": "https://github.com/BAI-LAB/MemoryOS",
    "how_to_use": "Implement the MemoryOS heat baseline with the exact constants; the extracted text is at /tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/memoryos_pdf/memoryos_raw.txt",
    "license": "Apache-2.0 (reported by search summary; verify in repo)"
  },
  {
    "what": "Memory-R1 (arXiv 2508.19828) \u2014 RL-trained ADD/UPDATE/DELETE/NOOP memory manager, 152 training QA pairs, LoCoMo/MSC/LongMemEval",
    "url": "https://arxiv.org/abs/2508.19828",
    "how_to_use": "Baseline for the learned-gate arm; cite v5 (Jan 2026) and check version-dependent gain figures (v5 reports +28% F1 for LLaMA-3.1-8B GRPO)",
    "license": "arXiv preprint"
  },
  {
    "what": "TokenGate (OpenReview 2mNNRc1MNo) \u2014 5,000 teacher-labelled admission examples on LongMemEval, twelve-view admission model",
    "url": "https://api2.openreview.net/notes/search?term=TokenGate+Write+Admission+Budgeted+Retrieval&source=all&limit=5",
    "how_to_use": "Source of per-candidate admission labels for LongMemEval admission precision/recall if the authors release them; otherwise replicate the teacher-labelling recipe",
    "license": "Anonymous ICLR 2027 submission; data availability unknown"
  },
  {
    "what": "MeClear (arXiv 2609.09115) and Causal Memory Policy (arXiv 2610.02070)",
    "url": "https://arxiv.org/abs/2609.09115",
    "how_to_use": "Replace exhaustive leave-one-fact-out re-answering with LOO screening + sampled Shapley (MeClear) or reserved-slot IPW utility estimation (CMP) to estimate per-fact accuracy contribution at bounded LLM cost",
    "license": "arXiv preprints"
  },
  {
    "what": "MEMAUDIT exact package-oracle protocol (arXiv 2605.02199)",
    "url": "https://arxiv.org/abs/2605.02199",
    "how_to_use": "Use its MILP-certified optimal write set as a budget-matched oracle for admission precision/recall and to separate representation quality from selection effects",
    "license": "arXiv preprint"
  },
  {
    "what": "ForgetEval and six-method Adapter Protocol (arXiv 2606.15903)",
    "url": "https://arxiv.org/abs/2606.15903",
    "how_to_use": "MIT-licensed adapter interface to plug the SIV gate into existing stores (Mem0, Letta, etc.) and reuse its ~130-line integration pattern",
    "license": "MIT (stated in abstract)"
  },
  {
    "what": "EngramRAG (arXiv 2609.32049) and STRATA (arXiv 2610.19386) \u2014 nearest 'degree/centrality in retention' neighbours with explicit formulas",
    "url": "https://arxiv.org/html/2609.32049",
    "how_to_use": "Cite as the closest prior use of topology (decay/GC); contrast with SIV's write-time use; reuse their log-scaling anti-hub trick to address speaker-hub dominance of Impact",
    "license": "arXiv preprints"
  },
  {
    "what": "HAGE code (RL-driven weighted graph evolution, arXiv 2605.09942)",
    "url": "https://github.com/FredJiang0324/HAGE_MVPReview",
    "how_to_use": "Potential graph-memory substrate with trainable edge features; check license before reuse",
    "license": "Unknown"
  },
  {
    "what": "Manufactured Confidence code (arXiv 2606.29279)",
    "url": "https://github.com/collapseindex/manufactured-confidence",
    "how_to_use": "Reuse its hedge/assertion probes to test whether SIV Trust (overlap) is insensitive to phrasing confidence",
    "license": "Unknown"
  },
  {
    "what": "AgentMemoryWorld Awesome-Agent-Memory 2026 list (TMLR 2026 resource)",
    "url": "https://raw.githubusercontent.com/AgentMemoryWorld/Awesome-Agent-Memory/main/papers/2026.md",
    "how_to_use": "Dated index of 2026 memory papers (coverage ends 2026-07-21 at time of reading); re-check for Aug\u2013Oct additions before submission",
    "license": "Repository license not checked"
  }
]

## risks

[
  "Fei et al. 2026 does not say agreement-based trust is reallocatable by sourceless writes; it shows top-k retrieval membership under PPR is reallocatable. Misquoting it as a trust result is a citable error; restate the objection as the paper's own inference about overlap Trust on a writable Gc.",
  "Several structure-as-signal papers appeared Sep\u2013Oct 2026 (EngramRAG, STRATA, Selective Forgetting, Beyond Freshness). Reviewers may argue 'degree/centrality in memory is not new'; the paper must position SIV as write-time novelty/impact scoring and include at least one of these as a baseline or ablation.",
  "A-MAC's gold admission labels on LoCoMo are of undocumented provenance; building admission precision/recall on them inherits that weakness. Prefer gold-evidence-derived labels (LoCoMo evidence spans, LongMemEval evidence sessions) or MEMAUDIT's oracle.",
  "GovMem, CPB and Manufactured Confidence all show that repetition/agreement is a poor trust signal and CPB shows source-type gating is the strongest simple policy; SIV Trust=|Et\u2229Ec|/|Et| is an agreement measure and will be attacked as poisonable unless combined with origin binding (Louck's lattice) in at least one arm.",
  "ConsolidationBench's own quality score fails to predict LoCoMo transfer (\u03c1=\u22120.24); a structural-delta score may likewise fail to correlate with QA deltas. Preregister the correlation test and report null results.",
  "SIVa under plain union is a new-triple ratio (al-Saffar's eq. 5) unless an inference step exists; the paper needs an explicit consolidation/inference operator or must present SIVp (degree-weighted) only.",
  "Semantic Scholar was unreachable (HTTP 429 x3) and the curated list covers only to 2026-07-21; Aug\u2013Oct 2026 coverage relies on arXiv API/OpenReview/Google Scholar and may miss venue-only or non-arXiv work.",
  "Many judgments rest on abstracts only (e.g., HAGE, SAGE graph engine, MemGuard, Eywa, Auto-Dreamer); a body-text structural admission rule could exist in one of them. Full texts were read only for GovMem, Epistemics, CPB, What-to-Admit, A-MAC, Fei et al., STRATA, EngramRAG, HeLa-Mem, GAM and the graph-memory survey.",
  "OpenReview ICLR 2027 submissions (Beyond Freshness, TokenGate, Executable Memory) are anonymous and under review; they may change or be withdrawn before this paper's submission.",
  "Memory-R1 reported gains differ across arXiv versions (v5: +28% F1; earlier: +68.9%); cite the specific version."
]

## confidence_notes

High confidence that no May–Oct 2026 paper found uses a graph-structural score for write-time admission, measures per-write graph deltas against QA accuracy, or tests connected-vs-disconnected priors: this rests on reading full HTML of the 11 most relevant papers, abstracts of ~80 more, three arXiv API queries (one returning zero results for graph edit distance + memory + agent), five OpenReview API queries and one Google Scholar query. Medium confidence on completeness: Semantic Scholar failed (429) three times, the curated 2026 list ends 2026-07-21, and most screening used abstracts, so a body-text structural rule could be missed. Numbers quoted from A-MAC, GovMem, CPB, Epistemics, Fei et al., STRATA, EngramRAG, HeLa-Mem, GAM, MemoryOS and al-Saffar were read from primary full texts or PDFs; numbers for other papers come from their arXiv abstract pages. The WebFetch tool caps verbatim quotes at ~125 characters, so some formulas were reconstructed from the tool's paraphrase of the source; the al-Saffar and MemoryOS equations were read directly from pdftotext output. Three facts marked verified=false come only from search-engine summaries (A-MemGuard details, MemoryOS license, Ori mnemos).