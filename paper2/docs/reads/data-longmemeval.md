# Reader: data:longmemeval

LONGMEMEVAL (Wu et al., arXiv 2410.10813, v1 2024-10-14, v2 2025-03-04, ICLR 2025). Code: github.com/xiaowu0162/LongMemEval (MIT, "Copyright (c) 2024 Di Wu"). Data licence MIT (HF card + paper ethics statement; sources ShareGPT Apache-2.0, UltraChat MIT).

DATA LOCATION. Official current data = Hugging Face `xiaowu0162/longmemeval-cleaned` (created 2025-09-19, MIT). Files and exact byte sizes (HF API): longmemeval_oracle.json 15,388,478; longmemeval_s_cleaned.json 277,383,467; longmemeval_m_cleaned.json 2,737,100,077 (total 3.03 GB). Direct URLs: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/<file>. The README's "Dataset Format" section still calls them longmemeval_s.json / longmemeval_m.json. Deprecated original: HF `xiaowu0162/longmemeval` (created 2024-10-13) with files `longmemeval_oracle` (15,388,478), `longmemeval_s` (278,025,796), `longmemeval_m` (2,745,274,681) — note NO .json extension in the repo tree, which is why the HF viewer errors. Original tarball `longmemeval_data.tar.gz` was on Google Drive id 1zJgtYRFhOh5zDQzzatiddfjYhFSnyQ80 (README @ commit 01055c0, 2025-02-05); cleaned copy on Drive id 1zo5C2sKsN3-2TUZt7kiRd2wsZLmyd-4y. Cleaning (2025-09): change-log sheet has 21 rows (question_id, session_id, issue, action); 17 filler sessions "session removed", 4 "no action"; issues = "answer interference", "info leak (no conflict)", "ambiguous reference". Oracle is byte-identical between original and cleaned (sha256 821a2034…620c).

SCHEMA (verified by parsing the files). Each file = JSON list of 500 dicts with keys question_id, question_type, question, answer, question_date, haystack_dates, haystack_session_ids, haystack_sessions, answer_session_ids. question_id examples: 'gpt4_2655b836', '0bb5a684', abstention 'gpt4_70e84552_abs' (suffix `_abs`, 30 questions). question_type counts (both S and oracle): temporal-reasoning 133, multi-session 133, knowledge-update 78, single-session-user 70, single-session-assistant 56, single-session-preference 30 (=500). The 30 abstention questions are embedded in those types (multi-session 12, temporal 6, knowledge-update 6, single-session-user 6) — "abstention" is NOT a question_type value. answer is str (468) or int (32). Dates are strings "%Y/%m/%d (%a) %H:%M", e.g. '2023/04/10 (Mon) 23:07'; haystack_dates sorted ascending in S/M, unsorted in oracle. haystack_sessions[i] is a list of turns {'role': 'user'|'assistant', 'content': str, optional 'has_answer': bool}; the has_answer key exists only on turns of evidence sessions (S: 10,960 turns with key — 896 True, 10,064 False — vs 235,790 without), so use turn.get('has_answer', False). answer_session_ids ⊆ haystack_session_ids always; evidence ids contain the substring 'answer' (e.g. 'answer_4be1b6b4_2', 'answer_b3070ec4_abs_1', 'answer_ultrachat_…'); fillers are 'sharegpt_<id>_<n>', 'ultrachat_…' or bare hex '85a1be56_1'. 13 S haystacks contain a duplicated filler session id (never an evidence id) — align sessions/ids/dates by position (zip), never by id.

SIZE (cleaned S, computed): 47.7 sessions/haystack (38–62), 493.5 turns/haystack, 10.3 turns/session; evidence sessions per question 1:176, 2:250, 3:41, 4:19, 5:11, 6:3. o200k_base tokens of all turn contents: mean 101.6k (95.2k–103.9k); JSON-serialised ≈111.8k; README says ~115k Llama-3 tokens (~40 sessions), paper says ~115k and ~50 sessions; M ≈500 sessions / ~1.5M tokens. Oracle: mean 5.6k tokens. Haystack dates span 2021-05-03→2024-02-20.

OFFICIAL SCORER (src/evaluation/evaluate_qa.py). `python evaluate_qa.py gpt-4o hyp.jsonl ../../data/longmemeval_oracle.json`; hyp = jsonl (or JSON list) of {question_id, hypothesis}; ref = any data file. model_zoo: 'gpt-4o'→'gpt-4o-2024-08-06', 'gpt-4o-mini'→'gpt-4o-mini-2024-07-18', 'llama-3.1-70b-instruct'→local vLLM at http://localhost:8001/v1. Single user message, temperature 0, n=1, max_tokens=10; label = ('yes' in response.strip().lower()). Prompt chosen by question_type; if '_abs' in question_id the abstention prompt is used instead ("I will give you an unanswerable question, an explanation, and a response … Please answer yes if the model correctly identifies the question as unanswerable…"). Type prompts: user/assistant/multi-session = contains-correct-answer, "If the response only contains a subset of the information required by the answer, answer no"; temporal adds "do not penalize off-by-one errors for the number of days"; knowledge-update accepts previous info "as long as the updated answer is the required answer"; preference treats answer as a "Rubric". Writes hyp+'.eval-results-<model>' with autoeval_label {model,label}; prints overall accuracy (mean over all) and per-type (count). print_qa_metrics.py (argv: in_file ref_file; asserts model=='gpt-4o-2024-08-06') prints per-type, "Task-averaged Accuracy" (mean of 6 type means), "Overall Accuracy", and "Abstention Accuracy" over the 30 _abs items (which also stay inside their type bins). Judge–human agreement (paper Table 6, GPT-4o): avg 0.98; preference 0.90, abstention 0.97, others 1.00.

GOLD EVIDENCE FOR ADMISSION P/R. Session level: answer_session_ids (890 non-abstention evidence sessions). Turn level: has_answer==True (896 turns; 842 user, 54 assistant; single-session-assistant evidence is on assistant turns, 51 of 56 questions have no user-side flag). Coverage gap: 45 non-abstention evidence sessions (ms 11/316, tr 32/280, ku 2/144) have no flagged turn → turn-level recall denominators must use flagged turns only, or fall back to session level. Official retrieval eval (run_retrieval.py/eval_utils.py) indexes USER turns only, ids sess_id+'_'+(1-based turn index), marks gold by 'answer' in id, renames evidence sessions lacking a flagged user turn 'answer'→'noans', skips _abs questions and questions with no user-side flag, reports recall_any/recall_all/NDCG@{5,10,50}. Abstention questions still carry answer_session_ids (and 10 True flags): exclude them from P/R. 44 questions have an evidence session dated AFTER question_date — do not time-filter candidates by question_date.

REPORTED LongMemEval_S NUMBERS (overall acc). Zep (arXiv 2501.13956): gpt-4o-mini 63.8 (full-context 55.4), gpt-4o 71.2 (60.2), GPT-4o judge. Nemori (2508.03341): gpt-4o-mini 64.2 (55.0), gpt-4.1-mini 74.6 (65.6). LightMem (2510.18866v4, GPT-4o-mini judge): GPT-4o-mini backbone FullText 56.80, NaiveRAG 61.00, LangMem 37.20, A-MEM 62.60, MemoryOS 44.80, Mem0 53.61, LightMem 64.29–68.64; Qwen3-30B-A3B: FullText 54.80, A-MEM 65.20, MemoryOS 49.60, Mem0 39.51, LightMem up to 70.20. Hindsight (2512.12818v1, judge GPT-OSS-120b): OSS-20B 83.6, OSS-120B 89.0, Gemini-3 Pro answerer 91.4; Supermemory 81.6/84.6/85.2 (GPT-4o/GPT-5/Gemini-3). Memory-R1 (2508.19828v5): zero-shot from LoCoMo, metrics F1/B1/J, GRPO J=55.40 (LLaMA-3.1-8B) and 57.80 (Qwen-2.5-7B); split and judge unspecified — not comparable.

Local copies for the loader work: /tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/lme_data/longmemeval_s_cleaned.json and longmemeval_oracle.json; repo clone at /tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/lme_repo; change-log CSV at /tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/lme_raw/changelog.csv.

## Facts (each with source and verification flag)

- [verified] Official current data is HF dataset xiaowu0162/longmemeval-cleaned (MIT, created 2025-09-19); files and byte sizes: longmemeval_oracle.json 15,388,478; longmemeval_s_cleaned.json 277,383,467; longmemeval_m_cleaned.json 2,737,100,077 (usedStorage 3,029,872,022).  
  source: https://huggingface.co/api/datasets/xiaowu0162/longmemeval-cleaned/tree/main
- [verified] README download commands: wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/{longmemeval_oracle.json,longmemeval_s_cleaned.json,longmemeval_m_cleaned.json}; cleaned data also at Google Drive id 1zo5C2sKsN3-2TUZt7kiRd2wsZLmyd-4y; change log at Google Sheet 16cHPu2B4XhgC-VvolIoWNs8wwm0Zkbpgu8H9x-qhxWg.  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/README.md
- [verified] Deprecated original HF dataset xiaowu0162/longmemeval (created 2024-10-13) lists files without .json extension: longmemeval_oracle 15,388,478 B, longmemeval_s 278,025,796 B, longmemeval_m 2,745,274,681 B; card says it is replaced by longmemeval-cleaned.  
  source: https://huggingface.co/api/datasets/xiaowu0162/longmemeval/tree/main
- [verified] README at commit 01055c0 (2025-02-05) pointed to the original release tarball longmemeval_data.tar.gz on Google Drive id 1zJgtYRFhOh5zDQzzatiddfjYhFSnyQ80 (unpack into data/).  
  source: https://github.com/xiaowu0162/LongMemEval
- [verified] The oracle file is byte-identical between original and cleaned HF repos (sha256 821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c).  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle
- [verified] Cleaning change log has 21 rows (Question ID, Session ID, Issue, Action): 17 'session removed', 4 'no action'; issue labels are 'answer interference', 'info leak (no conflict)', 'ambiguous reference'.  
  source: https://docs.google.com/spreadsheets/d/16cHPu2B4XhgC-VvolIoWNs8wwm0Zkbpgu8H9x-qhxWg/export?format=csv
- [verified] Each data file is a JSON list of 500 instances with keys question_id, question_type, question, answer, question_date, haystack_dates, haystack_session_ids, haystack_sessions, answer_session_ids (verified by parsing the downloaded S and oracle files).  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
- [verified] question_type counts (S and oracle): temporal-reasoning 133, multi-session 133, knowledge-update 78, single-session-user 70, single-session-assistant 56, single-session-preference 30; 30 abstention questions carry the '_abs' question_id suffix and are distributed as multi-session 12, temporal-reasoning 6, knowledge-update 6, single-session-user 6.  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] Date strings use format '%Y/%m/%d (%a) %H:%M' (e.g. '2023/04/10 (Mon) 23:07'); haystack_dates are sorted ascending in S (README: sorted for S and M, not for oracle); haystack dates span 2021-05-03 to 2024-02-20.  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] Turn dicts are {'role': 'user'|'assistant', 'content': str} plus 'has_answer': bool only on turns inside evidence sessions: in S 235,790 turns lack the key, 10,960 have it (896 True, 10,064 False); in oracle all 10,960 turns have it.  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] answer is a str in 468 instances and an int in 32 (e.g. '0a995998' -> 3).  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
- [verified] S cleaned statistics: 47.7 sessions/haystack (median 48, min 38, max 62), 493.5 turns/haystack (396-616), 10.34 turns/session (max 132); evidence sessions per question: 1:176, 2:250, 3:41, 4:19, 5:11, 6:3.  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] S cleaned token size with tiktoken o200k_base: content-only mean 101,571 tokens/haystack (95,244-103,899), JSON-serialised haystack ~111.8k; oracle mean 5,558 (max 21,824).  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] README: longmemeval_s 'roughly consumes 115k tokens (~40 history sessions) for Llama 3'; longmemeval_m 'roughly 500 sessions'; oracle has only evidence sessions; history compilation used max 80 filler sessions and enforce_json_length 115000 for S, 500 for M.  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/README.md
- [verified] Paper: LongMemEval_S about 115k tokens and roughly 50 sessions per history; LongMemEval_M 500 sessions, about 1.5M tokens; haystack mix 25% ShareGPT, 25% UltraChat, 50% simulated sessions; 30 questions converted to false-premise abstention questions; GPT-4o judge gpt-4o-2024-08-06 with >97% agreement with human experts; dataset MIT, ShareGPT Apache-2.0, UltraChat MIT.  
  source: https://arxiv.org/html/2410.10813v2
- [verified] Paper Table 6 judge-human agreement (GPT-4o): single-session-user 1.00, single-session-assistant 1.00, single-session-preference 0.90 (27/30), multi-session 1.00, knowledge-update 1.00, temporal-reasoning 1.00, abstention 0.97 (29/30), average 0.98.  
  source: https://arxiv.org/html/2410.10813v2
- [verified] arXiv 2410.10813: v1 14 Oct 2024, v2 4 Mar 2025, comments 'ICLR 2025', authors Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu.  
  source: https://arxiv.org/abs/2410.10813
- [verified] Repository LICENSE is MIT, 'Copyright (c) 2024 Di Wu'.  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/LICENSE
- [verified] evaluate_qa.py: usage 'python evaluate_qa.py metric_model hyp_file ref_file'; model_zoo maps 'gpt-4o'->'gpt-4o-2024-08-06', 'gpt-4o-mini'->'gpt-4o-mini-2024-07-18', 'llama-3.1-70b-instruct'->local vLLM at http://localhost:8001/v1; chat call with temperature 0, n=1, max_tokens=10; label = 'yes' in response.strip().lower(); abstention prompt used iff '_abs' in question_id; output written to hyp_file+'.eval-results-<metric_model>' with field autoeval_label {model,label}; prints overall mean accuracy and per-type accuracy with counts.  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/evaluate_qa.py
- [verified] Judge prompt for single-session-user/single-session-assistant/multi-session: 'I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.'  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/evaluate_qa.py
- [verified] Temporal-reasoning prompt adds: 'In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.'; knowledge-update prompt: 'If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.'; preference prompt uses 'Rubric: {}' and 'The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.'  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/evaluate_qa.py
- [verified] Abstention prompt: 'I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.'  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/evaluate_qa.py
- [verified] Hypothesis file is jsonl (or a JSON list) of objects with 'question_id' and 'hypothesis'; reference file is any of the data JSON files (README example uses longmemeval_oracle.json); hypotheses whose question_id is absent from the reference are skipped with a warning.  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/evaluate_qa.py
- [verified] print_qa_metrics.py takes 2 args (in_file ref_file), asserts autoeval_label.model == 'gpt-4o-2024-08-06', and prints per-type accuracy, 'Task-averaged Accuracy' (mean of the six type means), 'Overall Accuracy' (mean over all entries), and 'Abstention Accuracy' over entries with '_abs' in question_id; abstention entries are also counted inside their question_type bin.  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/print_qa_metrics.py
- [verified] README and scripts disagree: README says the log is saved as '[your_hypothesis_file].log' and invokes 'print_qa_metrics.py gpt-4o your_hypothesis_file.log ref', but evaluate_qa.py writes hyp_file+'.eval-results-gpt-4o' and print_qa_metrics.py requires exactly 2 positional args.  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/print_qa_metrics.py
- [verified] Official retrieval eval (src/retrieval/run_retrieval.py) indexes user turns only; turn ids are sess_id+'_'+str(i_turn+1) (1-based index over all turns incl. assistant); gold = corpus ids containing substring 'answer'; evidence sessions with no has_answer user turn have 'answer' replaced by 'noans' in their id; abstention questions and questions with no user-side has_answer turn are skipped when averaging metrics.  
  source: https://github.com/xiaowu0162/LongMemEval
- [verified] eval_utils.py defines recall_any (any gold doc in top-k), recall_all (all gold docs in top-k), NDCG@k with binary relevance, and evaluate_retrieval_turn2session which strips the trailing '_<turn>' to map turn ids to sessions; print_retrieval_metrics.py reports session-level recall_all@5/10 and ndcg_any@5/10 and turn-level recall_all@5/10/50, ndcg_any@5/10/50, excluding '_abs' questions.  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/print_retrieval_metrics.py
- [verified] Gold-evidence consistency in S cleaned: answer_session_ids is always a subset of haystack_session_ids; no session outside answer_session_ids has a has_answer=True turn; evidence session ids contain 'answer' (e.g. 'answer_4be1b6b4_2', 'answer_b3070ec4_abs_1', 'answer_ultrachat_...'), fillers are 'sharegpt_<id>_<n>', 'ultrachat_...' or bare hex like '85a1be56_1'.  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] has_answer=True turns in S: 896 total, 842 user-role and 54 assistant-role; by (type, role): single-session-user user 64/assistant 2; multi-session user 326/assistant 1; single-session-preference user 44; temporal-reasoning user 259; knowledge-update user 144; single-session-assistant assistant 51/user 5. 51 of 56 single-session-assistant questions have no user-side flag.  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] Turn-level gold coverage gap: of 890 non-abstention evidence sessions, 845 contain at least one has_answer=True turn; per type ssu 64/64, multi-session 305/316, ssp 30/30, temporal 248/280, knowledge-update 142/144, ssa 56/56 (45 evidence sessions have no flagged turn).  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] Abstention questions still carry answer_session_ids (21 with 2 ids, 6 with 1, 2 with 3, 1 with 4) and 10 has_answer=True turns; 49 of their evidence sessions have no flagged turn.  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] Timestamp anomalies in S cleaned: 42 questions have question_date earlier than the last haystack session date; 44 questions have at least one evidence session dated after question_date (e.g. gpt4_2655b836, gpt4_2487a7cb).  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] 13 S haystacks contain a duplicated session id (all fillers, e.g. '58bf7951' duplicates '07b7a667_1'; 'gpt4_c27434e8_abs' duplicates '8c64ce26_3'); none of the duplicates are evidence sessions.  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- [verified] Zep (arXiv 2501.13956, 20 Jan 2025) Table 2 LongMemEval_S: gpt-4o-mini Full-context 55.4% vs Zep 63.8%; gpt-4o Full-context 60.2% vs Zep 71.2%; Zep avg context 1.6k tokens vs 115k; judged with GPT-4o using the LongMemEval question-specific prompts; per-type gpt-4o Zep: preference 56.7, assistant 80.4, temporal 62.4, multi-session 57.9, knowledge-update 83.3, user 92.9.  
  source: https://arxiv.org/html/2501.13956v1
- [verified] Nemori (arXiv 2508.03341v1) Table 4 LongMemEval_S: gpt-4o-mini Full-context 55.0% vs Nemori 64.2%; gpt-4.1-mini Full-context 65.6% vs Nemori 74.6%; context 101K vs 3.7-4.8K tokens; LME judge model not stated (LoCoMo judge gpt-4o-mini).  
  source: https://arxiv.org/html/2508.03341v1
- [verified] LightMem (arXiv 2510.18866v4, 28 Feb 2026; judge GPT-4o-mini) Table 2 LongMemEval-S accuracy: GPT-4o-mini backbone FullText 56.80, NaiveRAG 61.00, LangMem 37.20, A-MEM 62.60, MemoryOS 44.80, Mem0 53.61, LightMem 64.29-68.64 (best r=0.7, th=512); Qwen3-30B-A3B backbone FullText 54.80, NaiveRAG 60.80, LangMem 50.80, A-MEM 65.20, MemoryOS 49.60, Mem0 39.51, LightMem up to 70.20; GLM-4.6 LightMem 73.20; per-type n: temporal 133, multi-session 133, knowledge-update 78, user 70, assistant 56, preference 30.  
  source: https://arxiv.org/html/2510.18866
- [verified] LightMem v1 (21 Oct 2025) abstract claimed 'up to 10.9% gains' in accuracy, token usage reduced 'up to 117x', API calls 'up to 159x', runtime 'over 12x' on LongMemEval with GPT and Qwen backbones; later versions report smaller gains (2.09-6.40% GPT, up to 7.67% Qwen).  
  source: https://arxiv.org/abs/2510.18866v1
- [verified] Hindsight (arXiv 2512.12818v1, 14 Dec 2025) Table 3 LongMemEval S setting (500 questions), judge GPT-OSS-120b at temperature 0: Hindsight (OSS-20B) 83.6, (OSS-120B) 89.0, (Gemini-3: memory GPT-OSS-120b, answer Gemini-3 Pro) 91.4; Full-context GPT-4o 60.2, Full-context OSS-20B 39.0; Zep (GPT-4o) 71.2; Supermemory 81.6 (GPT-4o), 84.6 (GPT-5), 85.2 (Gemini-3); the Full-context and Zep rows reuse Supermemory's GPT-4o-judged numbers.  
  source: https://arxiv.org/html/2512.12818
- [verified] vectorize-io/hindsight-benchmarks README reproduces the same LongMemEval table (Hindsight Gemini-3 overall 91.4%, per-type user 97.1, assistant 96.4, preference 80.0, multi-session 87.2, temporal 91.0, knowledge-update 94.9) without naming the judge or dataset version.  
  source: https://raw.githubusercontent.com/vectorize-io/hindsight-benchmarks/main/README.md
- [verified] Memory-R1 (arXiv 2508.19828v5, 14 Jan 2026) evaluates zero-shot on LongMemEval after training only on LoCoMo, reporting F1/BLEU-1/LLM-judge (J): LLaMA-3.1-8B GRPO 45.20/39.30/55.40 vs A-Mem 38.36/33.30/54.20 and Mem0 31.41/21.69/41.20; Qwen-2.5-7B GRPO 46.70/41.10/57.80 vs A-Mem 41.55/36.58/54.80 and Mem0 38.44/34.53/46.80; the LongMemEval split and judge model are not stated.  
  source: https://arxiv.org/html/2508.19828
- [verified] The project website lists five abilities and '500 questions of seven types' but has no leaderboard or download links.  
  source: https://xiaowu0162.github.io/long-mem-eval/
- [verified] LongMemEval-V2 (announced 2026/05) has 451 manually curated questions over web-agent trajectories, up to 500 trajectories and 115M tokens per haystack, data at HF xiaowu0162/longmemeval-v2.  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval-V2/main/README.md
- [verified] HF author listing shows exactly three datasets: xiaowu0162/longmemeval-cleaned (2025-09-19), xiaowu0162/longmemeval (2025-09-19, deprecated), xiaowu0162/longmemeval-v2 (2026-05-17).  
  source: https://huggingface.co/api/datasets?author=xiaowu0162
- [verified] README retrieval note: 'for evaluating the retrieval, we always skip the 30 abstention instances. This is because these instances generally refer to non-existing events and do not have a ground truth answer location.'; data-name to paper-name mapping: single_hop=single-session-user, implicit_preference_v2=single-session-preference, assistant_previnfo=single-session-assistant, two_hop and multi_session_synthesis=multi-session, temp_reasoning_implicit/explicit=temporal-reasoning, knowledge_update=knowledge-update.  
  source: https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/README.md

## reusable_assets

[
  {
    "what": "LongMemEval cleaned dataset (oracle / S / M JSON files, 500 instances each)",
    "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
    "how_to_use": "wget .../resolve/main/longmemeval_s_cleaned.json and longmemeval_oracle.json; json.load; iterate zip(haystack_session_ids, haystack_dates, haystack_sessions) by position; gold sessions = answer_session_ids, gold turns = turn.get('has_answer', False); skip '_abs' ids for admission P/R. Already downloaded to /tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/lme_data/",
    "license": "MIT"
  },
  {
    "what": "Official QA judge script with the six verbatim prompt templates and the 'yes'-substring labeling rule",
    "url": "https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/evaluate_qa.py",
    "how_to_use": "Copy get_anscheck_prompt() verbatim into the scorer; call gpt-4o-2024-08-06 with temperature 0, max_tokens 10; label = 'yes' in resp.lower(); pass abstention='_abs' in question_id; write jsonl with autoeval_label {model,label}",
    "license": "MIT"
  },
  {
    "what": "Aggregation script: per-type, task-averaged, overall, abstention accuracy",
    "url": "https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/print_qa_metrics.py",
    "how_to_use": "Run as 'python print_qa_metrics.py <eval-results file> <data json>' (two args, not the three the README shows); report both Overall Accuracy and Task-averaged Accuracy and say which one you use",
    "license": "MIT"
  },
  {
    "what": "Retrieval metric helpers (recall_any, recall_all, NDCG, turn->session mapping) and the official turn-id / 'answer'-in-id gold convention",
    "url": "https://github.com/xiaowu0162/LongMemEval",
    "how_to_use": "Clone (local copy at /tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/lme_repo); reuse src/retrieval/eval_utils.py for Recall@k/NDCG@k on admitted memory, and mirror process_item_flat_index() turn ids sess_id+'_'+(1-based index) so your admission log is comparable to the paper's retrieval logs",
    "license": "MIT"
  },
  {
    "what": "Cleaning change log (21 rows: question_id, session_id, issue, action)",
    "url": "https://docs.google.com/spreadsheets/d/16cHPu2B4XhgC-VvolIoWNs8wwm0Zkbpgu8H9x-qhxWg/export?format=csv",
    "how_to_use": "Use to state exactly which 17 filler sessions differ between original and cleaned S/M when comparing to pre-Sept-2025 numbers (Zep, Nemori, LightMem v1); local copy /tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/lme_raw/changelog.csv",
    "license": "not stated (Google Sheet shared by the authors)"
  },
  {
    "what": "Deprecated original dataset (pre-cleaning S/M) for reproducing older leaderboard rows",
    "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval",
    "how_to_use": "Files are served at .../resolve/main/longmemeval_s and longmemeval_m (no .json extension); only needed if you must replicate Zep/Nemori-era numbers on the original haystacks",
    "license": "MIT"
  },
  {
    "what": "Hindsight benchmark table of LongMemEval_S results (Full-context, Zep, Supermemory, Hindsight) with per-type numbers",
    "url": "https://raw.githubusercontent.com/vectorize-io/hindsight-benchmarks/main/README.md",
    "how_to_use": "Source for the related-work comparison table; mark judge differences (GPT-OSS-120b for Hindsight rows vs GPT-4o for the others)",
    "license": "not checked"
  },
  {
    "what": "Custom-history corpus (attributes, questions+evidence sessions, ShareGPT/UltraChat fillers, simulated sessions) and sample_haystack_and_timestamp.py",
    "url": "https://drive.google.com/file/d/1loTKBdywbCfYL5h5zwfnVcqlh7QwnBQm/view?usp=sharing",
    "how_to_use": "Lets you build haystacks of controlled length/filler ratio for ablations (e.g., vary distractor count to stress admission precision); run data/custom_history/sample_haystack_and_timestamp.py task n_questions min_filler max_filler enforce_json_length",
    "license": "MIT (repo) / ShareGPT Apache-2.0, UltraChat MIT (sources)"
  }
]

## risks

[
  "Dataset version drift: numbers published before Sept 2025 (Zep 63.8/71.2, Nemori 64.2, LightMem v1) were measured on the original haystacks; the cleaned release removed 17 filler sessions from S/M (oracle unchanged). State which version you use and do not mix rows without a footnote.",
  "Judge heterogeneity: official metric is gpt-4o-2024-08-06 yes/no; Hindsight uses GPT-OSS-120b, LightMem uses GPT-4o-mini, Memory-R1 reports F1/BLEU-1/J with unstated judge and split. Cross-paper comparisons are not like-for-like; re-run baselines under the official judge.",
  "Judge parsing quirk: label = 'yes' in response.lower(), so any response containing 'yes' (e.g. 'no, not yes') is scored correct; max_tokens=10 limits but does not eliminate this.",
  "Two aggregations exist (Overall Accuracy over 500 vs Task-averaged over 6 types) and abstention items are double-counted inside their type bins; pick one and report abstention separately, since an admission policy that drops evidence can inflate abstention accuracy spuriously.",
  "Turn-level gold is incomplete: 45 of 890 non-abstention evidence sessions carry no has_answer=True turn, and single-session-assistant evidence is on assistant turns (51/56 questions have no user-side flag). Per-turn admission recall must define its denominator on flagged turns only or fall back to session-level recall for those items; decide whether assistant turns are candidates at all.",
  "Abstention questions carry pseudo-evidence sessions (answer_session_ids non-empty, 10 True flags); exclude '_abs' items from admission precision/recall exactly as the official retrieval eval does.",
  "Timestamp anomalies: 44 questions have an evidence session dated after question_date; any admission/consolidation rule that discards content newer than the question will delete gold evidence.",
  "13 S haystacks contain duplicate filler session ids; keying sessions by id will silently drop sessions or mis-assign dates. Always zip by position.",
  "has_answer key is absent (not False) on non-evidence turns in S/M; naive turn['has_answer'] raises KeyError. The official code asserts the key only for 'answer' sessions.",
  "README/script mismatches (log filename '.log' vs '.eval-results-gpt-4o'; print_qa_metrics argc) will break copy-pasted commands.",
  "Memory footprint: S is 277 MB JSON (~0.5-1 GB in Python), M is 2.7 GB (~500 sessions, ~1.5M tokens per question) and is impractical for per-turn LLM-gated baselines (Mem0-style, Memory-R1) at full scale; budget for S only or sub-sample M.",
  "Token-count claims differ by tokenizer: ~115k (Llama-3, README/paper) vs ~101.6k content-only / ~111.8k serialised with o200k_base; Nemori reports 101K and LightMem 105.07k. Say which tokenizer you quote.",
  "Zep's Table 3 per-type gpt-4o single-session-user row is identical to the gpt-4o-mini row (81.4/92.9), a likely typo in the source; quote Zep per-type numbers with caution.",
  "Nemori's LongMemEval judge is not stated in the paper, and Memory-R1 never states whether it used S or oracle, so neither can be placed on the same axis as the official metric.",
  "LongMemEval-V2 (451 agentic questions, May 2026) now exists; reviewers may ask why v1 was chosen (answer: v1 is the dialogue-memory setting all compared systems report)."
]

## confidence_notes

High confidence (directly computed from downloaded primary files or read from raw source code): every schema, count, token, has_answer, timestamp and duplicate-id statistic; the full scorer behaviour and prompt texts; the HF file names/sizes; licences; the oracle checksum identity; the change-log contents. High confidence (two independent sources agree): Hindsight table (arXiv HTML and GitHub README), per-type n's (own count matches LightMem Table 7). Medium-high confidence (single source, extracted from arXiv HTML by the WebFetch summariser, not read character-by-character): Zep Table 2/3 values, Nemori Table 4 values, LightMem Table 2/7 values, Memory-R1 Table 5 values, paper Table 6 agreement figures and the 25/25/50 haystack mix. Not verified: paper Figure 9(a) per-type counts (image only; superseded by direct count), ShareGPT/UltraChat licence claims beyond the paper's own statement, and Google-Drive file sizes (links not downloaded). Web search was available; the arXiv HTML pages were read in full for the first 100k characters only (LongMemEval paper read in two chunks).