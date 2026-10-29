# Data notes: LoCoMo and LongMemEval-S as the loader and the scorer need them

Status: v0.1, 18 Oct 2026. Written from the reader reports `docs/reads/data-locomo.md` and
`docs/reads/data-longmemeval.md`, from the two data files on disk, and from `paper2/data.py` and
`paper2/evaluate.py`. It is the reference for every loader, scorer and per-category table in the paper;
the pre-registration (`00-preregistration.md`), the design critique (`02-design-critique.md`) and the
critique response (`05-critique-response.md`) are the design documents it must agree with, and where it
does not, section 5 says so.

Provenance tags used throughout:

- (verified): the reader report marks the fact as read from the primary file, page or code.
- (VERIFY): the reader report marks it as inferred or taken from a secondary source.
- (computed): computed on 18 Oct 2026 from the copies on disk (`data/raw/locomo/locomo10.json`,
  `data/raw/longmemeval/longmemeval_s_cleaned.json`, identical by SHA-256 to the readers' scratchpad
  copies) with two read-only scripts kept outside the data directories,
  `/tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/locomo_quirks.py` and
  `/tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/lme_haystack_sizes.py`
  (run with `python3 -I`, data path passed as an argument). Every (computed) aggregate that the reader
  also reports agrees with the reader's figure, with one exception: the number of observation items whose
  `dia_id` is a list (section 1.3), where `data-locomo.md` says 7 and this recount gives 10; section 1.3
  records both figures and why 10 is the one to quote.
- (counted from a named file): the change-log statistics of section 2.1 come from the reader's downloaded
  copy of the change-log sheet, not from a `data/raw` file; they name that file and the counting method
  inline instead of carrying the (computed) tag.

Nothing in this file is an experiment result; numbers from `paper2/out` do not appear here.

---

## 1. LoCoMo (Maharana et al., ACL 2024)

### 1.1 Which file, where, and its hash

| item | value | status |
|---|---|---|
| Download URL | `https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json` (branch `main`) | verified |
| Size | 2,805,274 bytes | verified; the copy at `data/raw/locomo/locomo10.json` has this size (computed) |
| SHA-256 | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` | verified; the copy on disk matches (computed) |
| Paper of record | Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents", Proc. ACL 2024 Vol. 1, pp. 13851–13870, DOI 10.18653/v1/2024.acl-long.747 | verified |
| What the file is | a 10-conversation subset of the 50 conversations described in arXiv 2402.17753 v1, "keeping the longest conversations with high-quality annotations" (README note 1); images are not released (README note 2) | verified |
| What the file is not | the arXiv v1 dataset (50 conversations, 7,512 QA: single-hop 2,705, multi-hop 1,104, temporal 1,547, open-domain 285, adversarial 1,871; about 300 turns and 9K tokens per conversation). Cite the ACL version and the GitHub file for every statistic | verified |
| Repository files worth knowing | `README.MD` (upper-case extension), `LICENSE.txt`, `task_eval/evaluation.py` (scorer), `task_eval/evaluation_stats.py` (aggregation), `task_eval/gpt_utils.py` (QA prompt), `task_eval/get_facts.py` and `generative_agents/memory_utils.py` (observation generation), `task_eval/get_session_summaries.py`, `prompt_examples/fact_generation_examples_new.json` | verified |

### 1.2 Licence and what it permits

- `LICENSE.txt` in the repository is Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0); the ACL paper's Appendix B.2 says the dataset "will be released under the CC BY-NC 4.0
  DEED license" (verified). The reader's reusable-asset note records no separate code licence in the
  repository (its probes for `LICENSE` and `LICENSE.md` at the repository root returned 404; the response
  bodies are on disk), so `task_eval/*.py` presumably falls under the same file; this is an asset note,
  not a flagged reader fact (VERIFY).
- The two conditions in the licence's name are the two that bind us: attribution, and no commercial use.
  Consequences for this project: every artefact we release that contains LoCoMo text (turn text,
  observations, session summaries, triples extracted from them, per-question answer logs that quote the
  gold) may be used for non-commercial purposes only and must attribute the dataset; a memory store built
  from LoCoMo cannot be shipped in a commercial product (this follows the reading already recorded in
  `04-submission-path.md` and `06-cost-model.md`). Whether a released derived file must also carry the
  licence notice itself is to be checked against the licence text before the code release (VERIFY).
- `paper2/evaluate.py` is a function-by-function port of `task_eval/evaluation.py`, with the attribution
  and the fetched file's SHA-256 (`8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd`) in
  its module docstring. Because the source file sits under the repository's CC BY-NC licence, the licence
  under which `paper2` itself is released has to be compatible with that port, or the port has to be
  replaced by calling the official script (VERIFY before the code release).
- The pre-registration's frozen section 13 said both datasets are MIT. That is wrong for LoCoMo
  (`04-submission-path.md` and `06-cost-model.md` already say so); `00-preregistration.md` section 13 now
  reads "LoCoMo under CC BY-NC 4.0 [corrected 18 Oct 2026; the frozen text said MIT]" and its log item 15
  records the correction (checked on disk).
- The images were never distributed (verified); the reader's risk list adds that `img_url` links may rot
  (its expectation, not something it checked). Anything about the image-dependent questions therefore
  rests on the BLIP captions only.

### 1.3 Schema as a loader needs it

Top level: a JSON list of 10 objects, `sample_id` in {conv-26, conv-30, conv-41, conv-42, conv-43, conv-44,
conv-47, conv-48, conv-49, conv-50}, each with keys `conversation`, `qa`, `observation`, `session_summary`,
`event_summary` (verified).

| field | type and format | quirks the loader must handle | status |
|---|---|---|---|
| `conversation.speaker_a`, `speaker_b` | str | speaker names repeat across conversations ("John" is a speaker in 3); key entities by `(sample_id, name)` | verified |
| `conversation.session_N` | list of turns, N contiguous 1..K | iterate over keys matching `^session_(\d+)$`, never over the `*_date_time` keys (see next row) | verified |
| `conversation.session_N_date_time` | str, `"1:56 pm on 8 May, 2023"`; parse with `"%I:%M %p on %d %B, %Y"` (all 288 strings match); dates strictly increase within a conversation | conv-26 carries 16 orphan keys `session_20_date_time` .. `session_35_date_time` with no `session_20`..`session_35` (computed; the reader reports the same range) | verified |
| turn | `{speaker: str, dia_id: str, text: str}` plus optional `img_url` (always a 1-element list), `blip_caption`, `query` (icrawler search query), `re-download` (True) | 1,226 turns carry a caption, 910 an `img_url`, 888 a `query`, 206 `re-download`; 316 have a caption but no `img_url` | verified |
| `dia_id` | `"D{session}:{turn}"`, 1-indexed; the session prefix equals the session number in all 5,882 turns | the same ids are referenced from `qa[].evidence` and from `observation` | verified |
| `observation.session_N_observation` | `{speaker: [[sentence, dia_id], ...]}` | the second element is a string in 2,531 items and a list of ids in 10 (8 lists of 2 ids, 1 of 3, 1 of 4; 2,541 in all). `data-locomo.md` says 7 items carry multiple ids; the recount here gives 10, and the reader's own `analyze2.py` explains the gap: it prints only `most_common(8)` of its irregular-item counter, which shows six lists and one comma-separated string and cuts off the other four lists (and four comma-separated strings). 10 is the figure to quote. Five of the 2,531 strings are themselves comma-separated id lists (`'D26:14, D26:34, D26:42'` in conv-44 session 26; two in conv-48 session 22; one each in conv-49 session 4 and conv-50 session 12). `paper2/data.py` accepts list and string (`ids = did if isinstance(did, list) else [did]`) but does not split those five strings, so each becomes one `turn_id` that matches no turn | computed |
| `session_summary.session_N_summary` | str, generated (section 1.7) | not used by the pipeline; mean 132.4 tokens per summary (ACL Table 5) | verified |
| `event_summary.events_session_N` | `{speaker: [events], date}`, human-annotated | ground truth for the event-summarisation task, not for QA; 669 entries | verified |
| `qa[]` | `{question: str, answer: str or int, evidence: [dia_id, ...], category: int}`; category-5 records carry `adversarial_answer` instead of `answer` (444 of 446; 2 carry both) | see section 1.4 | verified |

Counts (verified unless marked): 10 conversations; 272 sessions (per conversation 19, 19, 32, 29, 29, 28,
31, 30, 25, 30); 5,882 turns (369–689 per conversation; 10–47 per session, mean 21.6); 1,986 QA; 2,541
observations; 669 annotated events; conversation date spans 166–293 days; about 200,339 cl100k tokens of
"speaker: text [+caption]" (about 20.0K per conversation). The ACL paper's own figures are 588.2 turns,
27.2 sessions and 16,618.1 tokens per conversation (Table 1), 21.6 turns per session, 29.8 tokens per
turn, 19.2 tokens per observation, 35.8 events and 91.2 images per conversation (Table 5). Quote the
formatting and tokenizer with any token count: the three published or measured figures (9K arXiv v1,
16,618 ACL, about 20,034 cl100k) describe different things.

Per conversation (computed; the unit of inference in the LoCoMo arm, `02-design-critique.md` section C):

| conversation | sessions | turns | QA | cat 1 | cat 2 | cat 3 | cat 4 | cat 5 |
|---|---|---|---|---|---|---|---|---|
| conv-26 | 19 | 419 | 199 | 32 | 37 | 13 | 70 | 47 |
| conv-30 | 19 | 369 | 105 | 11 | 26 | 0 | 44 | 24 |
| conv-41 | 32 | 663 | 193 | 31 | 27 | 8 | 86 | 41 |
| conv-42 | 29 | 629 | 260 | 37 | 40 | 11 | 111 | 61 |
| conv-43 | 29 | 680 | 242 | 31 | 26 | 14 | 107 | 64 |
| conv-44 | 28 | 675 | 158 | 30 | 24 | 7 | 62 | 35 |
| conv-47 | 31 | 689 | 190 | 20 | 34 | 13 | 83 | 40 |
| conv-48 | 30 | 681 | 239 | 21 | 42 | 10 | 118 | 48 |
| conv-49 | 25 | 509 | 196 | 37 | 33 | 13 | 73 | 40 |
| conv-50 | 30 | 568 | 204 | 32 | 32 | 7 | 87 | 46 |
| total | 272 | 5,882 | 1,986 | 282 | 321 | 96 | 841 | 446 |

conv-30 has no open-domain (category 3) question, so a leave-one-conversation-out fold that tests on
conv-30 has an empty category-3 cell (computed).

### 1.4 Malformed and irregular records

All of these are in the released file; none is a loader bug (computed; the reader lists the same items).

| kind | where | entry | what `paper2/data.py` does |
|---|---|---|---|
| malformed evidence string (compound, `;`) | conv-26 q37, category 1 | `'D8:6; D9:17'` | dropped unless `repair_evidence=True`, which splits on `;`/`,`/whitespace and keeps the parts only if every part names a real turn |
| malformed evidence string (bare prefix) | conv-42 q88, category 4 | `'D'` | dropped; not repairable |
| malformed evidence string (stray colon) | conv-43 q18, category 1 | `'D:11:26'` | dropped unless repaired (canonicalised to `D11:26`, kept only if that turn exists) |
| malformed evidence strings (space-separated lists) | conv-49 q31, q38, q46, all category 3 | `'D9:1 D4:4 D4:6'`, `'D22:1 D22:2 D9:10 D9:11'`, `'D21:18 D21:22 D11:15 D11:19'` | dropped unless repaired (split on whitespace) |
| well-formed but dangling id (no such turn) | conv-42 q58 (cat 1) `'D10:19'`; conv-47 q38 (cat 1) `'D4:36'`; conv-50 q69 (cat 2) `'D30:05'` | | dropped; `'D30:05'` is canonicalised to `D30:5` under repair and kept only if that turn exists |
| empty evidence list | conv-26 q30, q46; conv-50 q39, q42 (all category 3, "Would ..." questions) | `[]` | kept with no evidence turns; `evidence_recall` and `admission_pr` return NaN for them, so they drop out of recall denominators |
| integer `answer` | conv-26 q1, q26, q49, q72 (category 2, answer `2022`); conv-26 q40 (category 1, `2`), q75 (category 1, `3`) | int | `str()` |
| category 5 with both `answer` and `adversarial_answer` | conv-26 q167, q178 (`answer='No'`, `adversarial_answer='Yes'`) | | `adversarial` is `category == 5` (not key presence, which was the critique's A1.2); `answer` is used as the gold when present, `adversarial_answer` otherwise |

Totals: 6 malformed strings, 3 dangling ids (9 entries that match no turn; `DROPPED_EVIDENCE_IDS` is 9
without repair), 4 empty lists, 6 integer answers, 2 double-keyed category-5 records (computed and
verified). Evidence distribution otherwise: 1,559 single-id and 427 multi-id questions; 1,064 of 2,815
evidence references point at caption-bearing turns, which matters because the images are not released
(verified).

### 1.5 Question categories: integer codes, names, counts

The central trap. The ACL paper's prose lists the categories as (1) single-hop, (2) multi-hop, (3)
temporal, (4) open-domain, (5) adversarial. The released file and the official code use a different
numbering (verified):

| code | official name | n | share | mean evidence turns per question | shape |
|---|---|---|---|---|---|
| 1 | multi-hop | 282 | 14.2% | 3.13 | aggregates several turns ("What exercises has John done?"); comma-split partial F1 in the scorer; 98% have at least 2 evidence ids |
| 2 | temporal | 321 | 16.1% | 1.17 | "When did/was ..." with a date answer; 83% start "When/How long"; the official prompt appends "Use DATE of CONVERSATION to answer with an approximate date." |
| 3 | open-domain (commonsense/world knowledge) | 96 | 4.8% | 2.08 | needs world knowledge; 44% contain would/likely; 11 golds carry `;`-separated alternatives, and the scorer keeps only the text before the first `;` |
| 4 | single-hop | 841 | 42.3% | 1.07 | one evidence turn, plain token F1 |
| 5 | adversarial | 446 | 22.4% | 1.03 | unanswerable by design; the gold is abstention; the trap answer sits in `adversarial_answer` |

Counts and shares are ACL Table 5 and the file (verified); the mean evidence turns are from the
`paper2/data.py` module docstring, computed on the file. Evidence that the mapping is right, all
verified: the Table 5 counts match the integer counts in the file only this way; `task_eval/evaluation.py`
scores category 1 with the comma-split "multi-hop eval" F1 and categories 2, 3, 4 under the comment
"single-hop, temporal, open-domain"; `evaluation_stats.py` prints the categories in key order
[4, 1, 2, 3, 5], which is the paper's Single / Multi / Temporal / Open / Adversarial column order; the
A-MEM reproduction README states the same mapping; the content heuristics above agree.

`paper2/data.py` carries exactly this table (`LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal",
3: "open-domain", 4: "single-hop", 5: "adversarial"}`), after the swap found by the critique (A1.1) and
logged as the pre-registration's erratum (section 6.1 and log item 10; `05-critique-response.md` item 1
cites log items 10–12 for the whole bundle, of which item 11 concerns hypothesis labels). Every per-category
number produced before 18 Oct 2026 17:00 UTC carried swapped names (pre-registration log item 10).

Five integer-to-name conventions are in circulation; only overall scores are label-invariant (verified
unless marked):

| convention | 1 | 2 | 3 | 4 | used by | status |
|---|---|---|---|---|---|---|
| A, official | multi-hop | temporal | open-domain | single-hop | LoCoMo code, A-MEM, Kumiho, Nemori | verified |
| B, Mem0 harness | single-hop | temporal | multi-hop | open-domain | Mem0 paper, Memobase, Backboard/Hindsight, Memory-R1 | verified for the Memobase fork's `categories` list only. Mem0's own code never names codes; its assignment is inferred from the exact reproduction of its 66.88 overall and from its relabelling of A-MEM's columns (VERIFY). Hindsight's is inferred from the exact reproduction of its 89.61 overall (VERIFY). Memory-R1's is only approximately consistent (62.57 recomputed vs 62.74 published) because its test split is 1,307 questions, not 1,540 (reader's confidence notes; VERIFY). Backboard enters only through the rows Hindsight copied from it (VERIFY) |
| C, MemoryOS | single-hop | multi-hop | temporal | open-domain | MemoryOS paper (its table relabels A-MEM and MemGPT rows this way) | inferred from the relabelled rows (VERIFY) |
| D, Zep README | multi-hop | open-domain | temporal | single-hop | getzep/zep-papers locomo_eval README | verified by recomputing Zep's released grades file by integer |
| E, Memobase "Zep*" row | single-hop | multi-hop | open-domain | temporal | Memobase README, copied into Hindsight | verified by matching Zep's recomputed integers |

Rules for this paper, following from the above: print the integer code next to the official name in
every table; when quoting another paper's per-category number, re-derive its convention from its own
counts or weighted overall (the integer counts 282/321/96/841 make this possible) and say which
convention it used; never copy a per-category row across conventions; compare overall scores only when
the metric is also the same (section 1.6).

### 1.6 Official scoring rules

From `task_eval/evaluation.py` (verified):

1. `normalize_answer`: remove commas, lower-case, strip punctuation, remove the tokens a / an / the / and,
   collapse whitespace.
2. Tokens are Porter-stemmed (`nltk.stem.PorterStemmer()`); `f1_score` is multiset-overlap token F1.
3. Category 1: split prediction and gold on `,`; score = mean over gold sub-answers of the max F1 over
   predicted sub-answers.
4. Categories 2, 3, 4: plain `f1_score`; category 3's gold is truncated before the first `;` first.
5. Category 5: 1 if `'no information available'` or `'not mentioned'` occurs in the lower-cased output,
   else 0. The gold is not consulted. The official QA prompt poses category-5 questions as a two-option
   choice "(a) Not mentioned in the conversation (b) {trap answer}" (order randomised), and the code
   reads `qa['answer']` where the released file stores the trap under `adversarial_answer`; refusal
   accuracy therefore depends heavily on the answer prompt.
6. `evaluation_stats.py`: per-category accuracy printed in key order [4, 1, 2, 3, 5]; "Overall accuracy"
   is the mean over all 1,986 questions including category 5.
7. Official QA prompt (`gpt_utils.py`): "Based on the above context, write an answer in the form of a
   short phrase for the following question. Answer with exact words from the context whenever possible.";
   category 2 gets the DATE suffix quoted in section 1.5.

What the paper's own metric text says (verified): "For each category, we calculate the F1 score for
exact matches, following the normalization of both the predicted and the actual ground truth answers"
and "We instruct the LLMs to replicate the exact wording in the conversation when feasible and employ
the F1 partial match metric".

Published anchors under this metric (ACL Table 2, order single / multi / temporal / open / adversarial /
overall, verified): human 95.1 / 85.8 / 92.6 / 75.4 / 89.4 / 87.9; gpt-4-turbo 128K 72.3 / 51.5 / 51.4 /
38.5 / 15.7 / 51.6; claude-3-sonnet 200K 70.7 / 38.1 / 26.9 / 52.2 / 2.5 / 42.8; gpt-3.5-turbo 16K
52.6 / 36.7 / 24.3 / 24.0 / 14.8 / 35.9. Table 3 RAG (DRAGON retriever, gpt-3.5-turbo reader, dialog
turns top-25): 59.9 / 38.7 / 37.2 / 25.0 / 12.8 / 41.0 with recall@25 87.1 / 62.5 / 83.5 / 52.6 / 66.3 /
76.7. The official RAG databases are dialog turns formatted `(date_time) speaker said, "text"`,
observations, or session summaries; top-k 5/10/25/50 for dialog and observation, 2/5/10 for summaries;
retrieval recall is the fraction of gold evidence ids among retrieved `dia_id`s (verified).

How `paper2/evaluate.py` implements it: `normalize_answer`, `f1_score`, `f1_multi_answer`, the
category-3 truncation and `OFFICIAL_ABSTAIN` are a port of the script; the Porter stemmer is a port of
`nltk/stem/porter.py` (NLTK 3.10.3) in the `NLTK_EXTENSIONS` mode the script instantiates, so no nltk
dependency is needed; `official_locomo_score(pred, gold, category_id)` dispatches on the integer code.
Its docstring records a token-for-token check of the stemmer against nltk on Martin Porter's 23,531-word
vocabulary and on every LoCoMo and LongMemEval gold token. `exact_match` (normalised string equality) is
this paper's extra metric; the official `exact_match_score` is token-set equality and is unused by the
official QA evaluation. The earlier, non-official forms (`normalize_answer_legacy`, `f1_score_legacy`,
the eleven-phrase `ABSTAIN` list) are kept only for reproducing pre-fix numbers and for LongMemEval,
where no official string rule exists.

Metric variants in circulation, none interchangeable with the official one (verified unless marked):

| protocol | metric | questions | used by |
|---|---|---|---|
| official | Porter-stemmed multiset F1, comma split (cat 1), `;` cut (cat 3), substring refusal (cat 5), overall over 1,986 | all 1,986 | LoCoMo paper; Kumiho ("following the official LoCoMo metric", pinned answer model gpt-4o-2024-08-06, four-category F1 on 1,540 plus refusal accuracy on 446) |
| A-MEM style | set-based token F1 after lower-casing and replacing `. , ! ?` with spaces: no stemming, no article removal, no comma split; BLEU-1 via nltk with smoothing method1 | all five categories (A-MEM) or four (Mem0, MemoryOS, Memory-R1) | A-MEM (`utils.py`), copied by Mem0 (`evaluation/metrics/utils.py`, "Borrowed from ... AgenticMemory"), MemoryOS (`evalution_loco.py`), Memory-R1 |
| Mem0 "J" | gpt-4o-mini judge, temperature 0, JSON mode, lenient prompt adapted from MemGPT ("as long as it touches on the same topic as the gold answer, it should be counted as CORRECT"; same date or time period counts for time questions); Mem0 reports mean ± sd over 10 judge runs; answer model gpt-4o-mini | category 5 skipped, 1,540 | Mem0 (commit aae5989; the `evaluation/` directory no longer exists on `main`), Memobase (fork), Zep (`locomo_eval`, structured `is_correct`), Nemori (same wording), LightMem (ACC, overall only) |
| other judges | GPT-OSS-120B at temperature 0 (Hindsight); Qwen3-14B strict judge requiring all gold information, no contradiction, no unsupported detail, correct temporal granularity (LoCoMo_refined, 1,382 revised questions); gemini-2.5-flash with three-level labels (LoCoMo-Plus) | varies | Hindsight, LoCoMo_refined, LoCoMo-Plus |

Two measured facts about judges (verified): LoCoMo_refined found the lenient judge agreed with humans
on 43.67% of 300 samples against 86.33% for its strict judge, and re-scored Mem0 at 48.91% (−15.56
points); Kumiho's README states that judge accuracy inflates scores 1.5–2x over token F1 and that an
unpinned gpt-4o drifted its own result from 0.565 to a reproducible 0.531. The pre-registration's choice
of the official F1 for LoCoMo (section 6.7) stands; a Mem0-compatible J (gpt-4o-mini, verbatim
`ACCURACY_PROMPT`, category 5 skipped) is optional and is reported with convention-B labels only when
set beside Mem0-convention tables (`06-cost-model.md` line D'' budgets it).

### 1.7 How the released observations and summaries were made, and what "LLM-free" may mean

- Observations (`observation.session_N_observation`) were generated by `gpt-3.5-turbo`
  (`scripts/generate_observations.sh` -> `task_eval/get_facts.py` ->
  `generative_agents.memory_utils.get_session_facts`) with `CONVERSATION2FACTS_PROMPT` ("Write a concise
  and short list of all possible OBSERVATIONS about each speaker ... include the dialog id ... objective
  factual information ... Avoid abstract observations") and in-context examples from
  `prompt_examples/fact_generation_examples_new.json`; the output per session is
  `{speaker: [[observation, dia_id], ...]}` (verified). The whole session is the prompt input, so each
  observation was written with every later turn of its session visible, and the `dia_id` attached to it
  is the model's own attribution.
- Session summaries were generated by `gpt-3.5-turbo` (`run_chatgpt`, model `'chatgpt'`, temperature
  1.0, 256 tokens) with the prompt "Generate a concise summary of the following conversation using exact
  words from the conversation wherever possible. The summary should contain all facts about the two
  speakers, as well as references to time.", prefixed by the session `date_time` and one
  `speaker said, "text" and shared <caption>.` line per turn (verified).
- The dialogues themselves were machine-generated and human-edited: personas, event graphs and dialogue
  generation used gpt-3.5-turbo (the event-graph step mentions text-davinci-003); image captions are
  BLIP-2; annotators edited about 15% of turns and removed or substituted about 19% of images (verified).
  `event_summary` is human-annotated (verified).
- Model-name discrepancy: the critique (B(i), "GPT-4 observations") names GPT-4, as did the
  pre-registration's section 6.1 and `paper2/README.md` when this file was first written; the generation
  scripts the reader read name `gpt-3.5-turbo` (verified). The pre-registration (section 6.1 and log item
  15, as revised on 18 Oct 2026) and `paper2/README.md` ("generated with gpt-3.5-turbo") now say
  gpt-3.5-turbo (checked on disk); the critique's B(i) still says GPT-4. The paper should say
  "gpt-3.5-turbo, per the released generation scripts" unless a primary source for GPT-4 is found (VERIFY).
  The argument below does not depend on which model.

What this implies for the oracle arm, consistent with the pre-registration's rephrased section 6.1 and
the critique's B(i):

1. "LLM-free" refers to the write-path decision only: zero model calls happen at the gate, given an
   extractor. It never refers to the pipeline, because in the oracle arm the candidates are the
   benchmark's released model-generated observations, produced offline with the whole session visible.
2. The observations pre-select. They exist for 40% of turns and cover 76% of gold evidence turns
   (pre-registration section 6.1); they cover all evidence turns of 158/282 multi-hop, 270/321 temporal,
   50/96 open-domain, 653/841 single-hop and 359/446 adversarial questions (pre-registration section 6.1,
   corrected names). Among turns that have an observation the gold-evidence rate is 0.451, among turns
   that do not it is 0.101 (`02-design-critique.md` B(i)). Most of the admission has happened before any
   policy runs, and admit-all's evidence-hit is the ceiling of the arm.
3. The observation-to-turn link is the generating model's own attribution, and evidence-hit is scored on
   `source_turns` taken from it; a real extractor would set `source_turns = [turn.id]`. The oracle arm's
   hit rate is therefore optimistic relative to any real extractor (critique B(i)).
4. Streaming session-level observations turn by turn leaks within-session look-ahead into the "online"
   write path (critique A6.7); the paper must say so wherever the oracle arm is reported.
5. The measures under test (al-Saffar and Heileman's Trust, Impact and SIV, and this paper's redesigns)
   are computed over the generating model's paraphrases in this arm, not over dialogue text.
6. The LLM arm (both datasets) replaces the released observations with this project's own extractor,
   whose model name, temperature and prompt are reported; "LLM-free" there still means the gate only.

Required phrasing, quoted verbatim from critique B(i) (`02-design-critique.md`): "zero LLM calls at the
gate, given an extractor; in the oracle arm the extractor is the benchmark's released GPT-4 session
observations, which were produced with the whole session visible and which cover 76% of gold evidence
turns and 40% of all turns; `n_edges` and the candidate text are LLM outputs". Never "LLM-free pipeline",
never "no LLM in extraction". Two notes on using it: "GPT-4" is the critique's wording and should read
"gpt-3.5-turbo" (the released generation scripts name gpt-3.5-turbo, verified; discrepancy note above),
and `n_edges` is the code's name for the number of observations written for a turn (critique B(i)).

### 1.8 Pitfalls other papers fell into

| pitfall | what happened | guard for this paper | status |
|---|---|---|---|
| category relabelling | Mem0 attached names to integers as 1 single / 2 temporal / 3 multi / 4 open and relabelled A-MEM's published columns to match; Memobase hard-codes that list; Hindsight's 89.61 overall reproduces only under it; MemoryOS relabelled A-MEM and MemGPT rows as 1 single / 2 multi / 3 temporal / 4 open; Zep's README labels cat 2 "Open Domain" and cat 3 "Temporal"; Memobase's "Zep*" row uses a sixth assignment | section 1.5 rules | verified for the Memobase list, Zep's recomputed integers and the relabelled rows; Mem0's, MemoryOS's, Hindsight's and Memory-R1's own assignments are inferred (VERIFY, section 1.5 table) |
| copied, not reproduced, baselines | Hindsight copies Backboard's rows and Memobase copies Mem0 (verified); Mem0 appears to copy A-MEM's Table 1 with relabelled columns (reader's inference, VERIFY). The reader's risk list states that Backboard's 90.00 is not independently reproducible and lists Backboard's numbers among its unverified items | every baseline number in the paper is from our own run or is marked "as published by X under X's metric" | verified; Backboard's reproducibility is the reader's statement (VERIFY) |
| metric mixing | official stemmed F1, A-MEM set F1, lenient gpt-4o-mini judge, GPT-OSS-120B judge and a strict Qwen3-14B judge appear in one literature; "no standardized LoCoMo leaderboard" (Kumiho) | one metric per table; section 1.6 variants table | verified |
| category 5 | the official code reads `qa['answer']` while the data stores `adversarial_answer`; the official rule is a substring check tied to a two-option prompt; most memory papers drop category 5 (1,540 questions, Mem0: "ground truth answers were unavailable") | report the official overall (1,986, cat 5 by abstention) and the four-category 1,540 figure, and say which | verified |
| disputed reproductions | Zep's rebuttal: Mem0 assigned the user role to both speakers, appended timestamps to message text instead of `created_at`, ran searches sequentially; Zep's rerun 75.14 vs 65.99 reported by Mem0 | speakers kept as named speakers with `session_N_date_time` as the timestamp (what `paper2/data.py` does) | verified |
| answer-model drift | Kumiho's 0.565 -> 0.531 from an unpinned gpt-4o | pin model snapshots and judge versions; report F1 alongside any judge | verified |
| gold-label noise | image-dependent questions whose captions lack the answer, wrong speaker attribution, ambiguous questions, inconsistent temporal golds (Zep, LoCoMo_refined); 1,064/2,815 evidence references are on caption-bearing turns | per-category intervals; LoCoMo_refined (1,382 questions) as an optional robustness set | verified |
| length and disclosure effects | LoCoMo-Plus: EM/F1/BLEU/ROUGE peak near reference length; disclosing the task inflates temporal and adversarial scores | fixed answer prompt and `max_tokens` across policies; no category hint beyond the official DATE suffix | verified |
| admission labels of unknown provenance | A-MAC's "approximately 1,500 candidate memories with ground-truth admission labels" are undocumented; its code labels a turn positive iff its `dia_id` is in any `evidence` list, 70/15/15 conversation split with seed 42, and its 0.583 is admission F1 on about 100–225 test candidates, not QA | our turn label is defined explicitly (gold evidence turn of some question) and is used for diagnostics, not as the QA claim (critique B(ii)) | verified |
| sample size | only 10 conversations, 96 open-domain questions, 18 distinct speaker names; Memory-R1's 152/81/1,307 split shows how fast the test set shrinks | conversation is the unit of inference; leave-one-conversation-out; wide per-category intervals expected | verified |
| version confusion | arXiv v1 (50 conversations, 7,512 QA) vs the released file (10, 1,986) | cite the ACL version and the GitHub file | verified |

---

## 2. LongMemEval-S (Wu et al., ICLR 2025; arXiv 2410.10813, v1 14 Oct 2024, v2 4 Mar 2025)

### 2.1 Which files, where, and their hashes

Official current data: Hugging Face dataset `xiaowu0162/longmemeval-cleaned` (created 2025-09-19, MIT)
(verified). Direct URLs: `https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/<file>`.

| file | bytes (HF API) | SHA-256 | status |
|---|---|---|---|
| `longmemeval_s_cleaned.json` (LongMemEval-S, the pre-registered dataset) | 277,383,467 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` | size verified; the copy at `data/raw/longmemeval/longmemeval_s_cleaned.json` has that size and this hash (computed); no published hash exists, so re-check against a fresh download (VERIFY) |
| `longmemeval_oracle.json` (evidence sessions only) | 15,388,478 | `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c` | verified; byte-identical between the original and the cleaned repo; the copy on disk matches (computed) |
| `longmemeval_m_cleaned.json` (about 500 sessions, about 1.5M tokens per question) | 2,737,100,077 | not downloaded | verified size; out of scope (impractical for per-turn gates, `06-cost-model.md`) |

Version history that matters for comparability (verified):

- The deprecated original, HF `xiaowu0162/longmemeval` (created 2024-10-13), lists `longmemeval_oracle`
  (15,388,478 B), `longmemeval_s` (278,025,796 B) and `longmemeval_m` (2,745,274,681 B) with no `.json`
  extension (which is why the HF viewer errors). The original tarball `longmemeval_data.tar.gz` was on
  Google Drive id `1zJgtYRFhOh5zDQzzatiddfjYhFSnyQ80` (README at commit 01055c0, 2025-02-05); the cleaned
  copy is on Drive id `1zo5C2sKsN3-2TUZt7kiRd2wsZLmyd-4y`; the change log is Google Sheet
  `16cHPu2B4XhgC-VvolIoWNs8wwm0Zkbpgu8H9x-qhxWg`.
- The cleaning (September 2025) is documented in a 21-row change log (question id, session id, issue,
  action): 17 rows "session removed", 4 "no action"; issue labels "answer interference", "info leak (no
  conflict)", "ambiguous reference" (verified). Counted on 18 Oct 2026 from the reader's downloaded copy
  of the sheet, `/tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/lme_raw/changelog.csv`
  (1,564 bytes; a header row `Question ID,Session ID,Issue,Action,,,` (four named plus three unnamed columns), then 21
  data rows; read with Python's `csv` module under `python3 -I`, cells stripped before grouping; not a
  `data/raw` file, so these figures do not carry the (computed) tag): 11 answer-interference rows (all
  removed), 5 info-leak rows (3 removed, 2 no action), 5 ambiguous-reference rows (3 removed, 2 no action);
  the 17 removals touch 16 distinct questions (`60472f9c` loses two sessions); three question ids
  (`c8c3f81d`, `9ea5eabc`, `dad224aa`) carry trailing spaces in the sheet, so strip before joining; the
  seventh column of the first three data rows holds a legend of the three issue labels (rows 2 and 3 carry legend
  text naming a different issue from their own; only row 1's matches), not per-row comments. Only filler sessions were removed (none
  of the 17 removed session ids contains `answer`, same count); the oracle is unchanged (verified,
  byte-identical).
- Zep (arXiv 2501.13956, 20 Jan 2025) and Nemori (arXiv 2508.03341v1, August 2025) predate the cleaned
  release of 2025-09-19, so their numbers were measured on the original haystacks (verified by date).
  LightMem v1 is dated 21 Oct 2025, after the cleaned release; no flagged reader fact says which haystack
  version LightMem used (the reader's risk list groups it with the pre-cleaning rows), so its dataset
  version is (VERIFY). State the version used and never mix rows without a footnote.
- The README's "Dataset Format" section still calls the files `longmemeval_s.json` / `longmemeval_m.json`
  (verified).
- LongMemEval-V2 (announced 2026-05, 451 questions over web-agent trajectories, HF
  `xiaowu0162/longmemeval-v2`) exists; v1 is used because it is the dialogue-memory setting every
  compared system reports (verified).

### 2.2 Licence and what it permits

- Repository LICENSE: MIT, "Copyright (c) 2024 Di Wu"; the HF card and the paper's ethics statement give
  the data as MIT; sources ShareGPT (Apache-2.0) and UltraChat (MIT) (verified; the source licences are
  the paper's own statement, not independently checked, VERIFY).
- MIT permits redistribution of derived files (triples, admission logs, per-question judgments) with the
  licence's attribution condition; the exact notice to attach is to be read from the LICENSE text before
  release (VERIFY). The pre-registration's section 13 ("LongMemEval-cleaned under MIT") is right for this
  dataset.
- The scorer and retrieval scripts (`src/evaluation/evaluate_qa.py`, `print_qa_metrics.py`,
  `src/retrieval/eval_utils.py`) are MIT and may be vendored with attribution (verified).

### 2.3 Schema as a loader needs it

Each file is a JSON list of 500 records with keys `question_id`, `question_type`, `question`, `answer`,
`question_date`, `haystack_dates`, `haystack_session_ids`, `haystack_sessions`, `answer_session_ids`
(verified).

| field | type and format | quirks | status |
|---|---|---|---|
| `question_id` | str, e.g. `'gpt4_2655b836'`, `'0bb5a684'`, `'gpt4_70e84552_abs'` | the `_abs` suffix marks the 30 abstention (false-premise) questions; "abstention" is not a `question_type` value. The meaning of the `gpt4_` prefix is not documented in the reader report (VERIFY) | verified |
| `question_type` | one of six strings (counts below) | the README maps the internal data names to the paper names: `single_hop` = single-session-user, `implicit_preference_v2` = single-session-preference, `assistant_previnfo` = single-session-assistant, `two_hop` and `multi_session_synthesis` = multi-session, `temp_reasoning_implicit/explicit` = temporal-reasoning, `knowledge_update` = knowledge-update | verified |
| `answer` | str (468) or int (32, e.g. `'0a995998'` -> 3) | `str()`; for `_abs` questions the field fills the `Explanation: {}` slot of the judge's abstention prompt: `evaluate_qa.py` reads `ans = qid2qdata[question_id]['answer']` (line 98) and formats the abstention template with `(question, answer, response)` (lines 41-42; the reader's copy of the script on disk, `/tmp/claude-0/-home-user/2409cfde-09f3-5d29-bda5-edc456faaa00/scratchpad/lme_repo/src/evaluation/evaluate_qa.py`) | verified (code on disk) |
| `question_date`, `haystack_dates[i]` | str `"%Y/%m/%d (%a) %H:%M"`, e.g. `'2023/04/10 (Mon) 23:07'` | `haystack_dates` sorted ascending in S and M, unsorted in the oracle; dates span 2021-05-03 to 2024-02-20; 42 questions have `question_date` earlier than the last haystack session date and 44 have an evidence session dated after `question_date`: do not filter candidates by `question_date` | verified |
| `haystack_session_ids[i]`, `haystack_sessions[i]` | parallel lists; a session is a list of turns `{'role': 'user' or 'assistant', 'content': str, optional 'has_answer': bool}` | align ids, dates and sessions by position (`zip`), never by id: 13 S haystacks list one filler session id twice (e.g. the haystack of question `58bf7951` lists session `07b7a667_1` twice, and that of question `gpt4_c27434e8_abs` lists `8c64ce26_3` twice; `58bf7951` is a question id, not a session id), never an evidence id (verified; recounted on disk: 13 haystacks, each with exactly one id occurring twice); `paper2/data.py` gives the repeat the id `<sid>~2` | verified |
| `has_answer` | bool, present only on turns of evidence sessions | in S, 235,790 turns lack the key and 10,960 have it (896 True, 10,064 False); in the oracle all 10,960 turns have it; use `turn.get('has_answer', False)` (naive indexing raises `KeyError`; `paper2/data.py` also accepts the string `"true"`) | verified |
| `answer_session_ids` | list of session ids, always a subset of `haystack_session_ids` | evidence ids contain the substring `'answer'` (`'answer_4be1b6b4_2'`, `'answer_b3070ec4_abs_1'`, `'answer_ultrachat_...'`); fillers are `'sharegpt_<id>_<n>'`, `'ultrachat_...'` or bare hex such as `'85a1be56_1'`; no session outside `answer_session_ids` has a `has_answer=True` turn | verified |
| turn `content` | str | one turn instance in one haystack contains the literal end-of-text marker `<\|endoftext\|>` (the GPT special-token string); `tiktoken` raises on it unless `disallowed_special=()` (or an explicit allow-list) is passed, so any token count must state how special tokens were handled (computed) | computed |

Question types (S and oracle; the 30 abstention questions are embedded in the types) (verified):

| `question_type` | n | of which `_abs` |
|---|---|---|
| temporal-reasoning | 133 | 6 |
| multi-session | 133 | 12 |
| knowledge-update | 78 | 6 |
| single-session-user | 70 | 6 |
| single-session-assistant | 56 | 0 |
| single-session-preference | 30 | 0 |
| total | 500 | 30 |

Gold evidence for admission precision and recall (verified):

- Session level: `answer_session_ids`; 890 non-abstention evidence sessions. Evidence sessions per
  question over all 500: 1: 176, 2: 250, 3: 41, 4: 19, 5: 11, 6: 3; over the 470 non-abstention
  questions: 1: 170, 2: 229, 3: 39, 4: 18, 5: 11, 6: 3 (computed).
- Turn level: `has_answer == True`; 896 turns (842 user, 54 assistant). By type and role:
  single-session-user user 64 / assistant 2; multi-session user 326 / assistant 1;
  single-session-preference user 44; temporal-reasoning user 259; knowledge-update user 144;
  single-session-assistant assistant 51 / user 5. 51 of the 56 single-session-assistant questions have no
  user-side flag: if assistant turns are not candidates, those questions have no reachable evidence, and
  H3's "two single-session types" clause has to say which of them is testable.
- Coverage gap: 45 of the 890 non-abstention evidence sessions contain no flagged turn (multi-session
  11/316, temporal-reasoning 32/280, knowledge-update 2/144; single-session-user 64/64, preference 30/30,
  assistant 56/56 fully covered). Turn-level recall denominators must use flagged turns only, or fall
  back to the session level for those items.
- Abstention questions still carry `answer_session_ids` (21 with 2 ids, 6 with 1, 2 with 3, 1 with 4)
  and 10 `has_answer=True` turns; 49 of their evidence sessions have no flagged turn. Exclude `_abs` items
  from admission precision and recall, as the official retrieval evaluation does ("these instances
  generally refer to non-existing events and do not have a ground truth answer location").
- Official retrieval evaluation conventions (`src/retrieval/run_retrieval.py`, `eval_utils.py`,
  `print_retrieval_metrics.py`): indexes user turns only; turn ids are `sess_id + '_' + (1-based index
  over all turns including assistant)`; gold = corpus ids containing `'answer'`; an evidence session with
  no flagged user turn has `'answer'` replaced by `'noans'` in its id; `_abs` questions and questions with
  no user-side flag are skipped; metrics `recall_any`, `recall_all`, NDCG@{5,10,50}, with
  `evaluate_retrieval_turn2session` stripping the trailing `_<turn>` to map turns to sessions.
  `paper2/data.py` names turns `<session_id>#<i>` with a 0-based index: convert before comparing an
  admission log with a published retrieval log.

### 2.4 Per-question haystack sizes (LongMemEval-S cleaned)

Computed on the file on disk; the reader's verified aggregates (47.7 sessions per haystack, 38–62;
493.5 turns, 396–616; 10.34 turns per session, max 132; content-only o200k_base mean 101,571 tokens,
95,244–103,899) agree. Tokens below are `tiktoken` `o200k_base` over turn `content` only (no role labels,
no JSON); the reader's JSON-serialised figure is about 111.8k per haystack; the README says "roughly 115k
tokens (~40 history sessions) for Llama 3"; the paper says about 115k tokens and roughly 50 sessions;
Nemori reports 101K and LightMem 105.07k. State the tokenizer and the formatting with any figure.

| question_type | n | sessions mean / median / min / max | turns mean / median / min / max | chars mean | o200k tokens mean |
|---|---|---|---|---|---|
| multi-session | 133 | 47.2 / 47 / 38 / 54 | 490.0 / 490 / 409 / 567 | 489,790 | 101,512 |
| temporal-reasoning | 133 | 47.5 / 47 / 41 / 61 | 492.6 / 489 / 429 / 614 | 488,850 | 101,497 |
| knowledge-update | 78 | 47.5 / 48 / 39 / 55 | 488.3 / 487 / 396 / 550 | 488,894 | 101,598 |
| single-session-user | 70 | 48.9 / 50 / 41 / 57 | 502.6 / 498 / 453 / 616 | 488,582 | 101,547 |
| single-session-assistant | 56 | 48.7 / 48 / 42 / 62 | 501.1 / 495 / 439 / 606 | 489,958 | 101,655 |
| single-session-preference | 30 | 47.6 / 48 / 42 / 55 | 491.1 / 494 / 437 / 548 | 490,588 | 101,992 |
| all | 500 | 47.7 / 48 / 38 / 62 | 493.5 / 491 / 396 / 616 | 489,298 | 101,571 |

Distribution over the 500 haystacks (computed):

| sessions per haystack | n | | turns per haystack | n | | o200k content tokens | n |
|---|---|---|---|---|---|---|---|
| 35–39 | 2 | | 350–399 | 1 | | 94,000–95,999 | 1 |
| 40–44 | 86 | | 400–449 | 30 | | 96,000–97,999 | 3 |
| 45–49 | 255 | | 450–499 | 279 | | 98,000–99,999 | 45 |
| 50–54 | 143 | | 500–549 | 168 | | 100,000–101,999 | 241 |
| 55–59 | 12 | | 550–599 | 18 | | 102,000–103,999 | 210 |
| 60–64 | 2 | | 600–649 | 4 | | | |

The haystacks are tightly size-matched by construction (the README's `enforce_json_length 115000` for S,
max 80 filler sessions; verified): the token range is 95,244–103,899, so every question costs about the
same to ingest, and a full-context prompt is above Claude Haiku 5.5's 100k price tier
(`06-cost-model.md`). Extremes: smallest `28dc39ac` (multi-session, 38 sessions, 431 turns), largest
`7e00a6cb` (single-session-assistant, 62 sessions, 564 turns) (computed).

Totals (computed): 23,867 session instances and 246,750 turn instances over the 500 haystacks; 19,195
distinct session ids, so one extraction per unique session saves 19.6% of the instances (the figure
`06-cost-model.md` section 3 asked to have verified; its 25,112 / 19,829 / 21% row describes the
deprecated original file, from `models-cost.md`). The gate itself still runs per haystack, because
ADD/UPDATE/DELETE/NOOP decisions depend on the memory built so far (`models-cost.md` risk list). The
oracle file has mean 5,558 o200k tokens per question (max 21,824) (verified).

### 2.5 Official scoring rules

Judge (`src/evaluation/evaluate_qa.py`; verified): `python evaluate_qa.py <metric_model> <hyp_file>
<ref_file>`; `hyp_file` is jsonl (or a JSON list) of `{question_id, hypothesis}`; `ref_file` is any of
the data files (hypotheses whose `question_id` is absent from the reference are skipped with a warning).
`model_zoo`: `'gpt-4o'` -> `gpt-4o-2024-08-06`, `'gpt-4o-mini'` -> `gpt-4o-mini-2024-07-18`,
`'llama-3.1-70b-instruct'` -> local vLLM at `http://localhost:8001/v1`. One user message, temperature 0,
n=1, `max_tokens=10`; label = `'yes' in response.strip().lower()`. The prompt is chosen by
`question_type`; if `'_abs'` is in the `question_id` the abstention prompt is used instead. Output is
written to `hyp_file + '.eval-results-<metric_model>'` with `autoeval_label {model, label}`; the script
prints overall accuracy (mean over all) and per-type accuracy with counts.

Prompt for single-session-user, single-session-assistant and multi-session (verbatim, verified):

```
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {}

Correct Answer: {}

Model Response: {}

Is the model response correct? Answer yes or no only.
```

Type-specific additions (verbatim, verified): temporal-reasoning adds "In addition, do not penalize
off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc.,
and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's
response is still correct."; knowledge-update adds "If the response contains some previous information
along with an updated answer, the response should be considered as correct as long as the updated answer
is the required answer."; single-session-preference uses "Rubric: {}" in place of the correct answer and
"The model does not need to reflect all the points in the rubric. The response is correct as long as it
recalls and utilizes the user's personal information correctly."

Abstention prompt (verbatim, verified):

```
I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {}

Explanation: {}

Model Response: {}

Does the model correctly identify the question as unanswerable? Answer yes or no only.
```

Aggregation (`print_qa_metrics.py`; verified): two positional arguments (`in_file ref_file`); it asserts
`autoeval_label.model == 'gpt-4o-2024-08-06'`; it prints per-type accuracy, "Task-averaged Accuracy"
(mean of the six type means), "Overall Accuracy" (mean over all 500) and "Abstention Accuracy" over the
30 `_abs` items, which also stay inside their type bins (double counting). Pick one of the two
aggregations, say which, and report abstention separately; an admission policy that drops evidence can
inflate abstention accuracy for free.

Known script quirks (verified): the README says the log is saved as `<hyp>.log` and invokes
`print_qa_metrics.py gpt-4o <hyp>.log <ref>` with three arguments, but `evaluate_qa.py` writes
`<hyp>.eval-results-gpt-4o` and `print_qa_metrics.py` takes exactly two; any response containing the
substring "yes" (for example "no, not yes") is scored correct, `max_tokens=10` limits but does not remove
this; `06-cost-model.md` treats the judge as deterministic by construction (temperature 0, 10 tokens),
which is an assumption about the API, not a measured fact (VERIFY by re-judging a sample).

Judge quality (paper Table 6, GPT-4o; verified): agreement with human experts single-session-user 1.00,
single-session-assistant 1.00, single-session-preference 0.90 (27/30), multi-session 1.00,
knowledge-update 1.00, temporal-reasoning 1.00, abstention 0.97 (29/30), average 0.98; the paper's text
says above 97%.

This project's position (pre-registration section 6.7): LongMemEval-S accuracy with the official judge
prompt per question type, plus F1 as an LLM-free fallback. `paper2/evaluate.py`'s LongMemEval path
(`category_id == 0`) scores token F1 and EM against the gold and `_abs` questions by the lenient
eleven-phrase `ABSTAIN` list; neither is official and both must be labelled as fallbacks. The judge must
be `gpt-4o-2024-08-06` with the six verbatim prompts to be called official (`06-cost-model.md` budgets
this on the API), and it should come from a different model family than the gate to avoid same-family
preference (`models-cost.md`).

Published LongMemEval-S numbers for context, not comparable across rows (verified): Zep (arXiv
2501.13956, GPT-4o judge, original haystacks) gpt-4o-mini 63.8 vs full-context 55.4, gpt-4o 71.2 vs
60.2; Nemori (2508.03341, judge not stated) gpt-4o-mini 64.2 vs 55.0, gpt-4.1-mini 74.6 vs 65.6;
LightMem (2510.18866v4, GPT-4o-mini judge) GPT-4o-mini backbone FullText 56.80, NaiveRAG 61.00, LangMem
37.20, A-MEM 62.60, MemoryOS 44.80, Mem0 53.61, LightMem 64.29–68.64; Hindsight (2512.12818v1, GPT-OSS-120b
judge) 83.6 / 89.0 / 91.4, Supermemory 81.6 / 84.6 / 85.2, with its Full-context and Zep rows reused
from Supermemory's GPT-4o-judged numbers; Memory-R1 (2508.19828v5) reports F1/BLEU-1/J zero-shot with
split and judge unspecified. Zep's Table 3 gpt-4o single-session-user row is identical to its
gpt-4o-mini row (81.4 / 92.9), a likely typo (verified); LightMem v1's abstract claimed "up to 10.9%
gains", later versions report 2.09–6.40% (GPT) and up to 7.67% (Qwen) (verified).

### 2.6 How the haystacks were generated, and "LLM-free" on this dataset

- Haystack composition: 25% ShareGPT, 25% UltraChat, 50% simulated sessions; 30 questions were converted
  to false-premise abstention questions (paper, verified). The history compilation used at most 80
  filler sessions and `enforce_json_length 115000` for S, 500 sessions for M (README, verified). The
  custom-history corpus (attributes, questions with evidence sessions, fillers, simulated sessions) and
  `data/custom_history/sample_haystack_and_timestamp.py` are released, so haystacks of controlled length
  and filler ratio can be built for ablations (verified).
- Unlike LoCoMo, LongMemEval ships no released observations or summaries. Every candidate fact in this
  arm comes from this project's extractor (the LLM arm's small instruct model at temperature 0, or the
  LLM-free GLiNER2 / spaCy arms of `06-cost-model.md`), so the "LLM-free" claim at the gate carries the
  extractor's name with it, and the official metric itself is an LLM judge. An LLM-free end-to-end number
  on LongMemEval-S exists only in the F1 fallback, which is not the official metric.
- Half of each haystack is simulated sessions (paper, verified). How the questions and their evidence
  sessions were authored is not covered by the reader report, and the `gpt4_` prefix on some question ids
  is undocumented there (VERIFY before describing the questions' provenance in the paper); the evidence
  turns carry `has_answer` flags set by the dataset authors (verified).

### 2.7 Pitfalls other papers fell into

| pitfall | what happened | guard for this paper | status |
|---|---|---|---|
| dataset version drift | Zep 63.8/71.2 (Jan 2025) and Nemori 64.2 (Aug 2025) predate the 2025-09-19 cleaned release and were measured on the original haystacks; LightMem v1 (21 Oct 2025) post-dates it and its dataset version is not established; the cleaned release removed 17 filler sessions from S and M | state "longmemeval_s_cleaned.json, SHA-256 d6f2...a442" in the paper; footnote every pre-September-2025 row and name the version for any later row that does not state it | verified by date for Zep and Nemori; LightMem's version VERIFY |
| judge heterogeneity | official gpt-4o-2024-08-06 yes/no; Hindsight GPT-OSS-120b; LightMem gpt-4o-mini; Nemori unstated; Memory-R1 unstated split and judge | run every baseline we report under the official judge; quote others as "as published, judge X" | verified |
| copied rows | Hindsight's Full-context and Zep rows reuse Supermemory's GPT-4o-judged numbers next to its own GPT-OSS-120b-judged rows | no mixed-judge tables | verified |
| two aggregations | Overall (500) vs Task-averaged (6 types), abstention items double-counted inside type bins | say which; report abstention separately | verified |
| oracle-vs-S confusion | the oracle file (1.9 sessions per question, no distractors) was the loader default until the critique (A1.3) | `paper2/data.py` defaults to S; every summary records the data file hash | verified (on disk) |
| incomplete turn-level gold | 45 evidence sessions with no flagged turn; single-session-assistant evidence on assistant turns | section 2.3 denominators; decide whether assistant turns are candidates | verified |
| time filtering | 44 questions have evidence dated after `question_date` | never filter or decay by `question_date` | verified |
| id keying | 13 haystacks with duplicated filler ids | zip by position; `~n` suffix | verified |
| tokenizer claims | 115k (Llama-3), 101.6k content-only / 111.8k serialised (o200k), 101K (Nemori), 105.07k (LightMem), 122k (4-chars-per-token heuristic in `models-cost.md`) | name tokenizer and formatting | verified |
| memory footprint | S is 277 MB JSON (about 2.4 GB peak when parsed, `paper2/data.py` docstring); M is 2.7 GB | S only; parse once, build Conversations lazily (`limit`) | verified (on disk) |
| copy-pasted commands | README/script mismatch on the log filename and argument count | run the scripts as the code, not the README, says | verified |

---

## 3. What the two datasets mean for the pipeline's shared contract

| requirement | LoCoMo | LongMemEval-S | where it lives today |
|---|---|---|---|
| candidate unit | a dialogue turn (`dia_id`), 5,882 per run; in the oracle arm the released observations (2,541) attached to their `dia_id`s | a turn `<session_id>#<i>` inside one question's haystack (about 493 per haystack, 246,750 instances); user turns only in the official retrieval convention | `paper2/data.py`, `paper2/extract.py` |
| memory stream | one conversation = one stream (10 streams), sessions in `session_N` order with `session_N_date_time` | one question = one stream (500 streams), haystack sessions in date order; the gate is reset per stream | `paper2/run.py` |
| gold for admission P/R | `qa[].evidence` turn ids after cleaning (9 unusable entries, 4 empty lists); category 5 evidence turns reported separately from value claims, since admitting them has zero or negative value and only 2 of 1,076 positive turns are exclusively adversarial evidence (critique B(ii)) | session level `answer_session_ids`; turn level `has_answer=True`; `_abs` excluded; 45 uncovered sessions handled by denominator rule | `paper2/evaluate.py::admission_pr` |
| unit of inference | conversation (10 clusters; leave-one-conversation-out) | question (500 independent haystacks), stratified by the six types | `paper2/stats.py` (critique section C) |
| official QA metric | Porter-stemmed token F1 with category dispatch, overall over 1,986; no judge | gpt-4o-2024-08-06 yes/no judge with six prompts; Overall and Task-averaged; abstention separately | `paper2/evaluate.py` (LoCoMo); judge not yet wired (`06-cost-model.md`) |
| LLM-free fallback metric | the official metric already is LLM-free | token F1 / EM and the lenient abstention list, labelled non-official | `paper2/evaluate.py` |
| per-category reporting | integer code and official name in every table; conv-30 has no category-3 question | six `question_type` values; abstention inside and beside the types | this file, sections 1.5 and 2.3 |
| licence of released artefacts | CC BY-NC 4.0: non-commercial, attribution | MIT: attribution | section 1.2, 2.2 |

---

## 4. Checklist: before any number is reported

1. **Dataset identity.** The SHA-256 of the loaded file equals the recorded hash: LoCoMo
   `79fa87e9...98ff4`, LongMemEval-S cleaned `d6f21ea9...a442`, oracle `821a2034...620c`. The hash is in
   the run's `summary.json` (the README says every summary carries the data file hash); the paper names
   the file and the hash once, in the data section.
2. **Category mapping check (LoCoMo).** Assert the integer counts {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
   and that the name table equals `paper2.data.LOCOMO_CATEGORIES` (1 multi-hop, 2 temporal, 3
   open-domain, 4 single-hop, 5 adversarial). Every table prints code and official name. Any number
   quoted from another paper's per-category column is re-derived to its convention (section 1.5 table)
   before it is placed next to ours, and the convention is named in the caption. Per-category figures
   produced before 18 Oct 2026 17:00 UTC carried swapped names (pre-registration log item 10); they are
   regenerated from `paper2/out` by the table script, never relabelled by hand (critique A10).
3. **Licence check.** LoCoMo is CC BY-NC 4.0: no commercial use, attribution required, images not
   released; the pre-registration's section 13 "both MIT" is corrected in place and in its log item 15; any released
   artefact containing LoCoMo text is labelled non-commercial; the compatibility of the vendored scorer
   port with the code-release licence is settled (section 1.2). LongMemEval is MIT: attribution kept. The
   baseline code licences that matter if code is vendored are A-MAC MIT, MemoryOS Apache-2.0 and LightMem
   MIT (`baselines-code.md`, verified), LongMemEval scripts MIT (`data-longmemeval.md`, verified), and
   Mem0 Apache-2.0 (recorded only in the readers' reusable-asset `license` fields in `baselines-code.md`
   and `data-locomo.md`, not as a flagged fact; VERIFY against the repository's LICENSE before vendoring);
   CoreNLP and OpenIE6 are GPL-3.0 (`models-cost.md`, verified) and stay separate processes (the reader's
   recommendation, adopted here).
4. **Malformed-id handling (LoCoMo).** The load reports `DROPPED_EVIDENCE_IDS` (9 without repair) and
   `REPAIRED_EVIDENCE_IDS`; the run records whether `repair_evidence` was on; the 4 empty-evidence
   category-3 questions are NaN in recall denominators and are counted as excluded; integer answers are
   stringified; `adversarial` is `category == 5` (not key presence); the two double-keyed category-5
   records are scored by the official substring rule like the other 444.
5. **Malformed-id handling (LongMemEval-S).** Sessions, ids and dates are aligned by position; the 13
   duplicated filler ids are kept as distinct sessions; `has_answer` is read with `.get`; `_abs` questions
   are excluded from admission precision and recall and reported as abstention accuracy; the 45 evidence
   sessions without a flagged turn are handled by a stated denominator rule; nothing is filtered by
   `question_date`; token counts state the tokenizer and how `<|endoftext|>` was handled; turn ids are
   converted to the official `sess_id_<1-based>` form before any comparison with published retrieval logs.
6. **Scorer parity test (LoCoMo).** Run the official `task_eval/evaluation.py` (the fetched file's hash
   `8e3be5d5...fd6fd` is recorded in `paper2/evaluate.py`; nltk is required for the official script and is
   not installed in the build sandbox) and `paper2.evaluate.official_locomo_score` on one fixed
   prediction file that covers all five categories, including comma-separated multi-hop predictions,
   `;`-carrying open-domain golds, the six integer answers and both abstention substrings, and require
   zero per-question differences. Today `paper2/tests/test_paper2.py::test_scoring` checks only the
   abstention substring and two F1/EM values, and the stemmer check lives in the `evaluate.py` docstring,
   not in the test suite; until the parity run is in `paper2/tests`, the paper says "a port of the
   official scorer", not "the official scorer". Report the official overall (1,986, category 5 by
   abstention) and the four-category figure (1,540) and say which one each table shows.
7. **Scorer parity test (LongMemEval-S).** The judge call is `gpt-4o-2024-08-06`, temperature 0,
   `max_tokens=10`, the six verbatim prompts of section 2.5, label = `'yes'` substring, abstention prompt
   iff `_abs`; `print_qa_metrics.py` (or an equivalent that asserts the model string) produces Overall,
   Task-averaged and Abstention accuracy, and the paper states which aggregation it reports. The token-F1
   fallback and the lenient `ABSTAIN` list are labelled non-official wherever they appear. Any published
   row placed beside ours names its judge and its dataset version.
8. **"LLM-free" wording.** Every LoCoMo oracle-arm number carries the required phrasing of section 1.7;
   the generating model is named as the generation scripts name it (`gpt-3.5-turbo`, verified), or the
   discrepancy with the pre-registration's "GPT-4" is resolved from a primary source first; every
   LongMemEval number names the extractor.
9. **Size and version footnotes.** LongMemEval-S haystack sizes are quoted from section 2.4 with the
   tokenizer named; LoCoMo token counts quote the formatting; rows measured on the original (pre-cleaning)
   haystacks or under another judge are footnoted.

---

## 5. Open items that belong to other files

- `00-preregistration.md` section 13: LoCoMo is CC BY-NC 4.0, not MIT (also noted in
  `04-submission-path.md` and `06-cost-model.md`); corrected in place with log item 15 (checked on disk),
  so this item is closed.
- `02-design-critique.md` B(i): "GPT-4 observations" vs the generation scripts' `gpt-3.5-turbo`
  (section 1.7); `00-preregistration.md` section 6.1 (log item 15) and `paper2/README.md` have been
  corrected on disk.
- `paper2/tests`: no parity test against the official LoCoMo script yet (checklist item 6).
- `06-cost-model.md` section 3: the cleaned LongMemEval-S file has 19,195 unique session ids over 23,867
  instances (computed here), which replaces the VERIFY it carries for that row.
- `docs/reads/data-locomo.md`: its "7 items carry multiple ids" undercounts; the file has 10 list-valued
  `dia_id` items plus 5 comma-separated strings (section 1.3), and the reader's own `analyze2.py` shows
  why (`most_common(8)` truncation). The reader report, not this file, is the one to correct.
