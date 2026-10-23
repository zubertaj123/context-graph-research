# Can the LLM-free extractor emit non-speaker subjects? A measurement, 19 Oct 2026

Question (critique E1, `08-extraction-check.md` "Consequence for the paper"): every memory in the LoCoMo
oracle arm is two speaker stars with 0 non-hub-to-non-hub edges, so the "structural" features measure
object recurrence around the speakers, not topology. Could a cheap, LLM-free change to `extract.py`
produce entity-to-entity edges? Probe script: `paper2/docs/probe_nonspeaker.py` (run from the repo root with `.venv/bin/python -I paper2/docs/probe_nonspeaker.py`) (spaCy
`en_core_web_sm`, all 10 conversations, 2,541 observation sentences); numbers below are from it.

## 1. The grammatical subject of the observation sentences

| subject of the main verb | sentences | share |
|---|---|---|
| a speaker's name | 2,150 | 84.6% |
| no parsable subject (conjoined subjects, parse failure) | 165 | 6.5% |
| "speaker's X" (possessive: kids, friends, wedding) | 152 | 6.0% |
| common noun ("The support group has made Caroline ...", "Painting is ...") | 40 | 1.6% |
| other named entity (PERSON 4, ORG 3, GPE 2) | 9 | 0.4% |
| pronoun / verb / number / aux | 25 | 1.0% |

**2.3% of sentences have a non-speaker subject at all.** The star shape is a property of the benchmark's
released observations (gpt-3.5-turbo summaries written as "Caroline did X"), not of the rule that
converts them. No subject-selection change can create edges the sentences do not state.

## 2. Three LLM-free rules, simulated on top of the current extraction

Tuple = (new edges, non-hub vertices with degree >= 2, non-hub-to-non-hub edges); current column first.

| conv | sentences | current: non-hub V / deg>=2 / nn-edges | A: relation noun + appositive name | B: non-speaker subject keeps its clause | C: co-mention (object <-> mentioned entity) |
|---|---|---|---|---|---|
| conv-26 | 184 | 161 / 17 / 0 | (0, 17, 0) | (6, 20, 6) | (14, 40, 14) |
| conv-30 | 169 | 124 / 16 / 0 | (0, 16, 0) | (5, 18, 5) | (4, 22, 4) |
| conv-41 | 324 | 262 / 36 / 0 | (0, 36, 0) | (6, 42, 6) | (18, 59, 18) |
| conv-42 | 266 | 224 / 24 / 0 | (0, 24, 0) | (2, 27, 2) | (17, 49, 17) |
| conv-43 | 267 | 239 / 38 / 0 | (0, 38, 0) | (6, 44, 6) | (60, 94, 59) |
| conv-44 | 277 | 210 / 35 / 0 | (0, 35, 0) | (3, 36, 3) | (35, 64, 34) |
| conv-47 | 268 | 258 / 27 / 0 | (0, 27, 0) | (3, 29, 3) | (49, 89, 49) |
| conv-48 | 291 | 239 / 29 / 0 | (0, 29, 0) | (5, 30, 5) | (24, 56, 24) |
| conv-49 | 240 | 191 / 27 / 0 | (0, 27, 0) | (1, 29, 1) | (14, 40, 13) |
| conv-50 | 255 | 213 / 28 / 0 | (0, 28, 0) | (2, 30, 2) | (64, 76, 59) |

- **Rule A** ("Caroline's husband John works at Dell" -> also (John, works_at, Dell)): **zero** edges in
  the corpus. Relation nouns with an appositive name do not occur in the observations.
- **Rule B** (keep the non-speaker subject: "The support group made Caroline feel accepted" ->
  (support group, made, accepted)): 1 to 6 edges per conversation, and the samples are bad
  (`caroline value -> promote -> lgbtq right`, `friend -> be -> relationship`). Not worth the precision cost.
- **Rule C** (link the primary object to every other named entity in the same sentence with
  `co_mentioned_with`): 4 to 64 edges per conversation; non-hub-to-non-hub share would go from 0% to
  roughly 2 to 20% of live edges and the deg >= 2 share from ~0.13 to ~0.28. But the edge is not a
  relation: `classical music -> co_mentioned_with -> perfect`, `classical music -> co_mentioned_with ->
  ed sheeran s`. It is the "second pass that links objects to each other" that `08-extraction-check.md`
  already names as an admission decision the paper would have to justify, and it would turn anchoring
  into a within-sentence co-occurrence count. A reviewer would read it as manufacturing topology.

## 3. Would an LLM extractor on the raw turns do better on LoCoMo?

Raw dialogue turns, first three conversations, spaCy NER for entities that are not the two speakers:

| | turns | share |
|---|---|---|
| turns | 1,451 | |
| with >= 1 third-party named entity | 171 | 11.8% |
| with >= 2 third-party named entities (an entity-to-entity fact is possible) | 15 | 1.0% |

LoCoMo conversations are about the two speakers' own lives; third parties are mentioned, rarely related
to each other. An LLM extractor with a closed schema would add (speaker, has_friend, Amy)-type edges,
which still hang off the hubs. Entity-to-entity topology on LoCoMo is a dataset limit, not an extractor
limit.

## 4. LongMemEval-S is where the topology can exist

User turns, first 40 questions (all single-session-user), spaCy NER as above:

| | user turns | share |
|---|---|---|
| user turns | 9,929 | |
| with >= 1 third-party named entity | 4,090 | 41.2% |
| with >= 2 third-party named entities | 1,537 | 15.5% |

Examples: "The Tim Ferriss Show" / Naval Ravikant, "The Daily" / The New York Times, Furminator / Luna /
Petco. Fifteen times LoCoMo's density of multi-entity turns. Most of those turns sit in filler
sessions, so the admission signal and the topology would both be tested on the realistic case where
most candidates are off-topic.

## 5. Recommendation

1. Do not change `extract.py`'s subject rule for the LoCoMo oracle arm; keep the E1 fallback wording
   ("turn-level admission features over an entity-centric fact store") and cite section 1 as the
   reason: 2.3% of the benchmark's own observations have a non-speaker subject.
2. Do not add the co-mention pass as a way of producing structure. If it is tried at all, report it as
   a separate ablation labelled "co-occurrence edges", with the precondition table before and after.
3. Put the entity-to-entity question on LongMemEval-S with the LLM extraction arm (closed predicate
   vocabulary, non-speaker subjects allowed when the subject is a named entity), which is already the
   week 5-7 item in `04-submission-path.md`. Before that run, decide the model stack and budget
   (`06-cost-model.md`); GLiNER2 (schema-based entity and relation extraction, 74M-340M, Apache-2.0) is
   the LLM-free candidate for the same arm and would keep the "zero gate-time LLM calls" line intact.
4. Print the graph-shape precondition table for every LongMemEval-S run; make a structural claim only
   where non-hub-to-non-hub edges exceed the stated threshold.
