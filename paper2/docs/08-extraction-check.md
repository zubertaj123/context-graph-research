# Hand-check of the LLM-free entity layer (spaCy path), conversation conv-26, 18 Oct 2026

Backend active: `spacy` (`en_core_web_sm`), see `paper2/extract.py`. Candidates are the benchmark's
gpt-3.5-turbo session observations rendered to triples by the dependency rule. This is a qualitative
check of what the rule does; the quantitative precondition table is in the regenerated results note.

## 18 consecutive primary triples (predicate | subject | object : type) with their source sentence

| verdict | triple | sentence |
|---|---|---|
| ok | attended (Caroline, LGBTQ support group : Thing) | Caroline attended an LGBTQ support group recently … |
| ok | plans_to (Caroline, continue education : Thing) | Caroline is planning to continue her education … |
| ok | run (Melanie, charity race : Thing) | Melanie ran a charity race for mental health last Saturday. |
| ok | research (Caroline, adoption agency : Thing) | Caroline is researching adoption agencies … |
| ok | choose (Caroline, adoption agency : Thing) | Caroline chose an adoption agency that helps LGBTQ+ folks … |
| ok | has_child (Melanie, Melanie's kid : Person) | Melanie's kids are excited about summer break … |
| ok | feels (Caroline, excited : Thing) | Caroline is excited to create a family … |
| ok | believes (Caroline, conversations about gender identity inclusion necessary : Thing) | Caroline believes conversations about gender identity and inclusion are necessary … |
| weak | manage (Melanie, Melanie's kid : Person) | Melanie is currently managing kids and work … |
| weak | did (Melanie, swimming : Thing) | Melanie is going swimming with the kids … |
| weak | realize (Melanie, self care : Thing) | Melanie is realizing the importance of self-care … |
| weak | give (Caroline, talk : Thing) | Caroline gave a talk at a school event … |
| wrong | made (Caroline, accepted : Thing) | The support group has made Caroline feel accepted … (subject should be the group) |
| wrong | made (Melanie, lake sunrise : Thing) | Melanie painted a lake sunrise last year … (predicate lost "painted") |
| wrong | has (Melanie, painting : NamedEntity) | Painting is a fun way for Melanie … (capitalised common noun typed as a name) |
| wrong | carve (Melanie, me : Thing) | Melanie carves out me-time each day … (hyphenated compound split) |
| wrong | hobby (Caroline, transitioning : Thing) | Caroline started transitioning three years ago. (canonical map over-applies "started" → hobby) |
| wrong | is_a (Melanie, inspiring other : Thing) | Melanie is supportive of Caroline and proud of her … (complement mis-selected) |

Rough precision of the primary triple on this sample: 8 ok, 4 weak, 6 wrong of 18. A 200-triple sample
with two annotators is the pre-registered requirement before any structural number is interpreted
(critique item 2); this is not that sample.

## Consolidation on conv-26 (admit everything)

One functional invalidation in this check: "Melanie has a husband and kids who keep her motivated"
was superseded by "Melanie's favorite part of her wedding was marrying her partner …" (same person,
different surface form: `married_to husband` vs `married_to partner`). That is a **false**
invalidation: the rule fires on surface-form change, not on a change of fact. Every invalidation in the
sweep must be hand-checked before SIVa's consolidation term is interpreted; expect the precision to be
low until objects of functional predicates are entity-resolved (husband = partner = spouse).

## Graph shape after admit-all

160 non-hub vertices, 10.6% with degree ≥ 2 (painting 7, Caroline's friend 5, art 5, Melanie's kid 4,
journey 3, adoption agency 2, …), **0 edges between non-hub vertices**, 160 non-hub components. The
memory is still two stars with some repeated leaves: the subject of every candidate is a speaker, so no
rule over speaker-centred observations can produce entity-to-entity edges. Entity types: Thing 132,
Person 16, Pet 7, NamedEntity 2, Place 2, Org 2, Date 1, Event 1.

Consequence for the paper (critique E1 fallback): on this arm the honest wording is "turn-level
admission features over an entity-centric fact store", not "graph-structural". Entity-to-entity
structure needs an extractor that emits non-speaker subjects (the LLM arm with a closed schema) or a
second pass that links objects to each other (co-mention within a session), which is itself an
admission decision the paper would have to justify.

## After the second repair (type-guarded functional predicates, canonical spouse/partner objects, labelled fallbacks)

Numbers from the repair agent's run over all ten conversations (scripts and outputs under the session
scratchpad, `fix2_stats_*.txt`), to be confirmed by the regenerated sweep:

| quantity | before repair | after repair |
|---|---|---|
| candidates (primary triples) | 3,200 (2,549) | 3,199 (2,549) |
| distinct object strings; recurring ≥ 2 | 2,157; 355 | 2,142; 365 |
| triples with a functional predicate | 16 (≥ 8 false) | 8 (all true on inspection) |
| sentences taking the rule fallback | 103 / 2,541 (4.1%), unlabelled | 74 / 2,541 (2.9%), method `*_rule_fallback` |
| bare `is` predicates | 12 | 6, all from the labelled fallback |
| closed-vocabulary predicates | 1,828 (57%) | 1,879 (59%) |
| functional invalidations, admit-all, 10 conversations | 14 (0 of 3 distinct ones true) | 0 |

The eight functional triples: Melanie married_to "Melanie's spouse" (twice, now merging instead of
contradicting), Jolene dating "Jolene's partner" (three sentences), Jolene studies engineering,
Deborah dating "Deborah's partner", Evan dating "Evan's partner". Dropped as false: residence from
"staying in" / "will be in" (now `stayed_in`), employer from gerunds (now `works_on`), `studies`
"upcoming final" (now `study_for`), `married_to` "woman" (now bare `marry`). Dropped as true but
type-rejected: Dave works_at "car maintenance shop" (common-noun employer, now `work_at`).

What this settles: on LoCoMo's released observations the functional rule is now correct but has
nothing to do, because the observations record no change of residence, employer or partner. The
consolidation term of SIVa (and the pre-registration's section-11 claim) therefore cannot be tested on
this arm at all; LongMemEval-S knowledge-update questions and the LLM-extraction arm are where it must
be tested, as log item 12 already says. Known residuals: a generic "X's spouse" followed by a named
spouse would still read as a contradiction (needs resolution in `graph.resolve`); parser-limited
sentences such as "Jolene studies engineering." take the labelled fallback.
