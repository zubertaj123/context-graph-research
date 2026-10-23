"""Candidate triples for admission.

Three extractors share one output type so the admission policy never knows which produced them:
  OracleObservationExtractor  LoCoMo's own per-session GPT-4 observations (tier llm-summary, evidence-linked)
  HeuristicExtractor          first-person sentences from user turns (tier extractor-inferred)
  LLMExtractor                schema-constrained triples from a chat model (paper2.llm adapters)

Sentence-to-triple rule (critique E1, A4.6, B(iv), E2(b)). Two backends, chosen lazily on first use and
reported in EXTRACTION_BACKEND ("spacy" | "rule"):

  spacy  en_core_web_sm dependency parse + NER. Subject = the known name heading the nominal subject (or
         owning it: "Melanie's kids ..."), else the speaker. Predicate = lemma of the main verb, folded into
         the closed vocabulary CANONICAL_PREDICATES with light (verb, preposition) / (verb, object-class)
         patterns; copulas never yield the bare "is" (attr -> is_a, acomp -> feels or a mapped relation,
         "32 years old" -> has_age). Object = the head noun of the first complement of the verb extended with
         its compound / adjectival / proper-noun children ("the LGBTQ support group" -> "LGBTQ support
         group", "Boston, MA" -> "Boston"); a noun introduced by "named"/"called" yields the name ("puppy
         named Max" -> "Max", type Pet). Relation nouns ("her kids", "a dog named Luna") become
         has_child / has_pet / married_to ... with owner-qualified CANONICAL objects ("Melanie's child";
         "Melanie's spouse" whether the sentence said husband, wife or spouse; "Melanie's partner" for
         partner / boyfriend / girlfriend; marrying one's partner makes them the spouse) so they recur
         within a speaker, never merge across speakers and are never read as a contradiction between their
         own synonyms. FUNCTIONAL relations are type-guarded (critique B(iv), see _guard): lives_in /
         moved_to / born_in need a Place object, works_at an Org / Place / named employer, works_as a role
         noun, married_to / dating / engaged_to a person, studies a field; any other object gets a
         non-functional predicate (live_in, move_to, work_at, stay_in, stayed_in, works_on, is_in, the bare
         lemma, or feels). Types: Person / Place / Org / Event / Date / Pet / Thing (NER labels mapped by
         NER_TYPES; capitalised unknowns stay NamedEntity for the graph's resolution rules).
  rule   the original deterministic token rule (suffix stripper, 6-token object chunk), unchanged, used
         when spaCy or the model is missing or when PAPER2_EXTRACT_BACKEND=rule. On the spaCy path a
         sentence the parse rule cannot handle (verb without complement, bare noun phrase) takes this rule
         for that sentence only; the extractors mark such triples method='<extractor>_rule_fallback' and
         the module counts them in FALLBACK_SENTENCES (of EXTRACTED_SENTENCES distinct sentences).

Either way one sentence gives ONE primary triple plus the 'mentions' (named entities) and 'mentions_date'
(absolute dates) extras. The rule is deterministic and not under test; `python -m paper2.extract` runs the
module self-check (_selfcheck) over the probe sentences that fix the functional-predicate contract.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Iterable, Iterator

from .data import Conversation, Observation, Turn

STOP = set("a an the this that these those my your his her their our its some any very really just also to of in on at for with and or but so as is are was were be been being am do does did have has had will would can could should may might not no yes it they them we you i me he she him".split())
BE = {"is", "are", "was", "were", "am", "be", "been", "being", "'s"}
VERB_SUFFIX = [("ies", "y"), ("ied", "y"), ("sses", "ss"), ("shes", "sh"), ("ches", "ch"), ("xes", "x"), ("ing", ""), ("ed", ""), ("s", "")]
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DATE_RE = re.compile(rf"\b(\d{{1,2}}\s+(?:{MONTHS})(?:,?\s+\d{{4}})?|(?:{MONTHS})\s+\d{{1,2}}(?:,?\s+\d{{4}})?|\d{{4}})\b", re.I)

# ---------------------------------------------------------------------------
# Tiers. One table on the extractor side so every extractor's default confidence is the tier's (A6.8).
# ---------------------------------------------------------------------------
DEFAULT_TIER_CONFIDENCE = {"verbatim-user": 1.0, "extractor-inferred": 0.8, "llm-summary": 0.6, "poisoned": 0.4, "poisoned-lower": 0.4}

# ---------------------------------------------------------------------------
# Closed predicate vocabulary (E2(b)). Verb lemmas not in CANONICAL_PREDICATES keep their lemma.
# ---------------------------------------------------------------------------
CANONICAL_PREDICATES: dict[str, str] = {
    # residence / origin
    "live": "lives_in", "reside": "lives_in", "move": "moved_to", "relocate": "moved_to", "born": "born_in", "bear": "born_in",
    # work / study
    "work": "works_at", "employ": "works_at", "study": "studies", "major": "studies", "graduate": "studied_at",
    # relationships
    "marry": "married_to", "married": "married_to", "wed": "married_to", "date": "dating", "engaged": "engaged_to", "befriend": "friend_of",
    # possession
    "own": "owns", "buy": "owns", "purchase": "owns", "have": "has",
    # health
    "diagnose": "diagnosed_with",
    # attitudes
    "like": "likes", "love": "likes", "enjoy": "likes", "adore": "likes", "prefer": "likes", "dislike": "dislikes", "hate": "dislikes",
    "value": "values", "appreciate": "values", "cherish": "values", "admire": "values", "respect": "values",
    "believe": "believes", "think": "believes", "want": "wants", "wish": "wants", "hope": "wants",
    # intentions / events / activities
    "plan": "plans_to", "intend": "plans_to", "aim": "plans_to", "aspire": "plans_to", "consider": "plans_to", "decide": "plans_to",
    "attend": "attended", "participate": "attended", "visit": "visited", "travel": "visited", "join": "member_of", "belong": "member_of",
    "celebrate": "celebrated", "play": "hobby", "practice": "hobby", "practise": "hobby",
    "make": "made", "create": "made", "build": "made", "craft": "made", "paint": "made", "draw": "made", "write": "made", "compose": "made",
    # copula-derived (produced by the copula handler, listed so the vocabulary is closed)
    "is_a": "is_a", "feels": "feels", "has_age": "has_age", "has_child": "has_child", "has_parent": "has_parent",
    "has_sibling": "has_sibling", "has_pet": "has_pet", "did": "did",
}
# (verb lemma, preposition) -> predicate; applied when the object is that preposition's complement. The object
# type is checked afterwards by _guard (stay/be + in is presence, never residence; spaCy lemmatises 'engaged' to
# 'engage', 'married' to 'marry', 'born' to 'bear').
PREP_PREDICATES: dict[tuple[str, str], str] = {
    ("live", "in"): "lives_in", ("live", "at"): "lives_in", ("reside", "in"): "lives_in", ("reside", "at"): "lives_in",
    ("stay", "in"): "stayed_in", ("stay", "at"): "stayed_in", ("be", "in"): "stayed_in",
    ("move", "to"): "moved_to", ("relocate", "to"): "moved_to", ("be", "from"): "born_in", ("bear", "in"): "born_in", ("grow", "in"): "born_in",
    ("work", "at"): "works_at", ("work", "for"): "works_at", ("work", "in"): "works_at", ("employ", "at"): "works_at", ("employ", "by"): "works_at",
    ("work", "as"): "works_as", ("be", "as"): "works_as", ("work", "on"): "works_on",
    ("study", "at"): "studied_at", ("graduate", "from"): "studied_at", ("enrol", "in"): "studied_at", ("enroll", "in"): "studied_at", ("major", "in"): "studies",
    ("marry", "to"): "married_to", ("wed", "to"): "married_to", ("engage", "to"): "engaged_to",
    ("go", "to"): "visited", ("travel", "to"): "visited", ("fly", "to"): "visited", ("trip", "to"): "visited", ("be", "into"): "hobby",
    ("take", "to"): "visited", ("bring", "to"): "visited", ("spend", "with"): "spends_time_with",
    ("diagnose", "with"): "diagnosed_with", ("suffer", "from"): "diagnosed_with", ("participate", "in"): "attended", ("look", "into"): "plans_to",
}
# copula + adjectival complement (+ preposition) -> predicate; anything else is (s, feels, "<adj> <prep> <obj>").
# Keyed on the adjective as spaCy gives it (ADJ lemma 'married' / 'engaged') and on the verb lemma for the
# participle reading ('got married to' -> marry).
ACOMP_PREDICATES: dict[tuple[str, str], str] = {
    ("married", "to"): "married_to", ("marry", "to"): "married_to", ("engaged", "to"): "engaged_to", ("engage", "to"): "engaged_to",
    ("interested", "in"): "likes", ("passionate", "about"): "likes",
    ("fond", "of"): "likes", ("keen", "on"): "likes", ("allergic", "to"): "diagnosed_with", ("diagnosed", "with"): "diagnosed_with",
}
# head noun of a copula attribute whose "of"-complement is what the speaker likes ("a big fan of pottery").
LIKER_NOUNS = {"fan", "lover", "enthusiast", "admirer", "aficionado"}
# relation nouns -> (predicate, object type); the object is owner-qualified ("Melanie's kids") unless named.
RELATION_NOUNS: dict[str, tuple[str, str]] = {}
for _n in "kid child son daughter baby toddler twin stepson stepdaughter".split():
    RELATION_NOUNS[_n] = ("has_child", "Person")
for _n in "husband wife spouse".split():
    RELATION_NOUNS[_n] = ("married_to", "Person")
for _n in "boyfriend girlfriend partner".split():
    RELATION_NOUNS[_n] = ("dating", "Person")
for _n in "fiance fiancee fiancé fiancée".split():
    RELATION_NOUNS[_n] = ("engaged_to", "Person")
for _n in "brother sister sibling".split():
    RELATION_NOUNS[_n] = ("has_sibling", "Person")
for _n in "mother father mom dad mum parent grandmother grandfather grandma grandpa".split():
    RELATION_NOUNS[_n] = ("has_parent", "Person")
for _n in "friend bestie".split():
    RELATION_NOUNS[_n] = ("friend_of", "Person")
for _n in "dog cat puppy kitten pet hamster rabbit bunny bird parrot fish turtle horse pig lizard snake gecko goldfish".split():
    RELATION_NOUNS[_n] = ("has_pet", "Pet")
PERSON_NOUNS = {"mentor", "colleague", "coworker", "neighbor", "neighbour", "boss", "teacher", "roommate", "cousin", "aunt", "uncle", "niece", "nephew", "family"}
# one canonical surface per owner for relation nouns that name the same person or group ("Melanie's spouse"
# whether the sentence said husband, wife or spouse), so a relation never contradicts its own synonyms
RELATION_SURFACE = {"husband": "spouse", "wife": "spouse", "boyfriend": "partner", "girlfriend": "partner", "fiancee": "fiance", "fiancé": "fiance",
                    "fiancée": "fiance", "kid": "child", "baby": "child", "toddler": "child", "mom": "mother", "mum": "mother", "dad": "father",
                    "grandma": "grandmother", "grandpa": "grandfather", "bestie": "friend", "puppy": "dog", "kitten": "cat", "bunny": "rabbit"}
COUPLE_PREDICATES = {"married_to", "dating", "engaged_to"}
# couple verbs: "Caroline and Sam got married / are engaged / are dating" -> the conjoined person is the partner;
# marrying one's partner or fiance makes them the spouse
COUPLE_VERBS = {"marry": "married_to", "wed": "married_to", "engage": "engaged_to", "date": "dating"}
# verbs whose relation-noun object states the relation: "has / adopted / got a dog named Max" -> has_pet Max
POSSESSION_VERBS = {"have", "adopt", "get", "rescue", "foster", "welcome", "raise"}
# semantically empty heads: "the importance of X" -> X; "spends time with Y" -> prefer the prepositional object
LIGHT_NOUNS = {"importance", "time", "way", "lot", "kind", "sort", "role", "part", "source", "step", "sense", "idea", "thing", "bit", "amount", "number", "piece"}
# gerund complements whose own object is the activity ("started playing acoustic guitar" -> "acoustic guitar")
ACTIVITY_VERBS = {"play", "learn", "practice", "practise", "do", "take", "study", "teach"}
# main verbs whose infinitival complement is the object ("plans to continue her education" -> "continue education")
INTENT_VERBS = {"plan", "want", "hope", "intend", "decide", "aim", "aspire", "try", "consider", "need", "wish", "look", "prepare"}
# predicates whose clausal complement is a proposition rather than a mis-attached noun
ATTITUDE_PREDICATES = {"believes", "wants", "feels", "values", "likes", "dislikes", "plans_to", "find", "mention", "know", "realize", "say", "learn",
                       "understand", "acknowledge", "note", "explain", "share", "discuss", "recall", "remember", "notice", "see", "hear", "show"}
# adjectives that add nothing to an entity name
NOISE_ADJ = {"new", "own", "big", "little", "small", "great", "good", "nice", "recent", "same", "other", "first", "last", "next", "several", "many",
             "few", "favorite", "favourite", "whole", "entire", "various", "different", "certain", "such", "huge", "tough", "amazing"}

FUNCTIONAL_PREDICATES: set[str] = {"lives_in", "moved_to", "works_at", "works_as", "married_to", "dating", "engaged_to", "studies", "born_in", "has_age"}
SYMMETRIC_PREDICATES: set[str] = {"married_to", "dating", "engaged_to", "has_sibling", "friend_of"}
INVERSE_PREDICATES: dict[str, str] = {"has_child": "child_of", "child_of": "has_child", "parent_of": "child_of", "has_parent": "has_child"}
PROFILE_PREDICATES: set[str] = set(CANONICAL_PREDICATES.values()) | set(PREP_PREDICATES.values()) | set(ACOMP_PREDICATES.values()) | \
    {p for p, _ in RELATION_NOUNS.values()} | set(INVERSE_PREDICATES) | set(INVERSE_PREDICATES.values()) | \
    {"mentions", "mentions_date", "stayed_in", "works_on", "born_on", "is_in", "is_from"}

NER_TYPES = {"PERSON": "Person", "GPE": "Place", "LOC": "Place", "FAC": "Place", "ORG": "Org", "EVENT": "Event", "DATE": "Date", "TIME": "Date",
             "WORK_OF_ART": "Thing", "PRODUCT": "Thing", "NORP": "Thing", "LANGUAGE": "Thing"}
MENTION_LABELS = {"PERSON", "GPE", "LOC", "FAC", "ORG", "EVENT", "WORK_OF_ART"}

# ---------------------------------------------------------------------------
# Backend selection: spaCy is optional and imported lazily; the flags below say which path is active.
# ---------------------------------------------------------------------------
SPACY_MODEL = "en_core_web_sm"
SPACY_AVAILABLE: bool | None = None      # None until the first parse is attempted
EXTRACTION_BACKEND: str = "rule"         # "spacy" once the model has loaded; read after the first extraction or call backend()
_NLP = None


def _load_spacy():
    global SPACY_AVAILABLE, EXTRACTION_BACKEND, _NLP
    if SPACY_AVAILABLE is not None:
        return _NLP
    if os.environ.get("PAPER2_EXTRACT_BACKEND", "").lower() == "rule":
        SPACY_AVAILABLE, EXTRACTION_BACKEND = False, "rule"
        return None
    try:
        import spacy  # noqa: WPS433 (optional dependency)
        _NLP = spacy.load(SPACY_MODEL, exclude=["senter"])
        SPACY_AVAILABLE, EXTRACTION_BACKEND = True, "spacy"
    except Exception:            # ImportError, OSError (model missing), anything else: fall back silently
        _NLP, SPACY_AVAILABLE, EXTRACTION_BACKEND = None, False, "rule"
    return _NLP


def backend() -> str:
    """'spacy' or 'rule'; loads the model on first call."""
    _load_spacy()
    return EXTRACTION_BACKEND


@dataclass
class Triple:
    s: str
    p: str
    o: str
    s_type: str = "Person"
    o_type: str = "Thing"
    source_turns: list[str] = field(default_factory=list)
    session: str = ""
    speaker: str = ""
    observed_at: datetime | None = None
    tier: str = "extractor-inferred"       # verbatim-user | extractor-inferred | llm-summary | poisoned
    confidence: float = 0.8
    method: str = "rule"
    text: str = ""                          # the sentence the triple came from (for retrieval and the LLM arm)
    truth: bool | None = None               # set for injected poison; None for real facts

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.s, self.p, self.o)


# ---------------------------------------------------------------------------
# Lemmatiser (A4.6): spaCy when available, the original suffix stripper otherwise.
# ---------------------------------------------------------------------------
def _rule_lemma(word: str) -> str:
    w = word.lower()
    if w in BE:
        return "is"
    for suf, rep in VERB_SUFFIX:
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not (suf == "s" and w.endswith("ss")):
            return w[: -len(suf)] + rep
    return w


@lru_cache(maxsize=65536)
def _spacy_lemma(w: str, pos: str | None = None) -> str:
    """spaCy lemma of one word. A bare word is POS-ambiguous ('lives' -> 'life' as a noun), so by default the
    word is parsed as the verb of a minimal clause ("She lives" -> live, "She values" -> value); pos naming
    another part of speech (e.g. 'NOUN') parses the word alone for its nominal reading."""
    nlp = _load_spacy()
    if nlp is None:
        return _rule_lemma(w)
    if pos is None or pos.upper() in ("VERB", "AUX"):
        doc = nlp("She " + w)
        if len(doc) >= 2 and doc[1].pos_ in ("VERB", "AUX"):
            return doc[1].lemma_.lower()
    doc = nlp(w)
    return doc[0].lemma_.lower() if len(doc) else w


def lemma(word: str, pos: str | None = None) -> str:
    """Lemma of one word read as a verb ('lives' -> live, 'values' -> value, 'moved' -> move); pass pos='NOUN'
    for the nominal reading ('lives' -> life). BE forms give 'is', as the fallback rule always did."""
    w = word.lower()
    if w in BE:
        return "is"
    return _spacy_lemma(w, pos) if _load_spacy() is not None else _rule_lemma(w)


def entity_type(surface: str) -> str:
    if DATE_RE.fullmatch(surface.strip()):
        return "Date"
    if surface[:1].isupper():
        return "NamedEntity"
    return "Thing"


# predicates whose meaning depends on the preposition: "works on X" is not works_at, "married for 5 years" is not
# married_to, "studying for finals" is not studies
PREP_SENSITIVE = {"lives_in", "moved_to", "stayed_in", "works_at", "works_as", "works_on", "studies", "studied_at", "born_in", "married_to", "dating",
                  "engaged_to", "visited", "diagnosed_with", "member_of"}
# verbs whose relation needs its preposition: a bare direct object is no residence / employer / relocation fact
# ("lives a busy life", "works two jobs", "moved the couch" -> live, work, move)
PREP_REQUIRED = {"live", "reside", "stay", "move", "relocate", "work", "employ", "bear", "born", "graduate", "major", "enrol", "enroll"}
PERSON_TYPES = {"Person", "NamedEntity"}
EMPLOYER_TYPES = {"Org", "Place", "NamedEntity"}
TYPE_GUARDED = {"lives_in", "moved_to", "stayed_in", "born_in", "works_at", "works_as", "married_to", "dating", "engaged_to", "studies"}
# study objects that are neither a field nor an institution ("studying hard for upcoming finals")
STUDY_NON_FIELDS = {"final", "exam", "test", "midterm", "quiz", "material", "note", "lesson", "chapter", "book", "textbook", "night", "hour", "week", "lot", "hard"}
_NUMERIC_ENTS = ("DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY")
_BARE = {"bear": "born", "be": "is", "married": "marry", "engaged": "engage"}      # so the fallback never spells the relation itself


def canonical_predicate(lemma_: str, prep: str | None = None) -> str:
    """Fold a verb lemma (and the preposition its object hangs from) into the closed vocabulary; unknown
    lemmas are returned unchanged. A preposition that the vocabulary does not know keeps a prep-sensitive
    verb out of the relation it would otherwise map to ("work with" -> work_with, "study for" -> study_for,
    never works_at / studies), and a verb that needs its preposition gets nothing from a bare direct object
    ("live" alone -> live, not lives_in). The object TYPE is checked afterwards by the extractor (_guard)."""
    if prep:
        if (lemma_, prep) in PREP_PREDICATES:
            return PREP_PREDICATES[(lemma_, prep)]
        canon = CANONICAL_PREDICATES.get(lemma_, lemma_)
        if canon in PREP_SENSITIVE or lemma_ in PREP_REQUIRED:
            return f"{_BARE.get(lemma_, lemma_)}_{prep}"
        return canon
    if lemma_ in PREP_REQUIRED:
        return _BARE.get(lemma_, lemma_)
    return CANONICAL_PREDICATES.get(lemma_, lemma_)


def _is_role_noun(tok, otype: str) -> bool:
    return tok is not None and tok.pos_ in ("NOUN", "PROPN") and otype in ("Thing", "NamedEntity") and tok.lemma_.lower() not in LIGHT_NOUNS \
        and tok.ent_type_ not in _NUMERIC_ENTS


def _is_field(tok, otype: str) -> bool:
    """A field of study: a common / proper noun that is not an exam, a time span or a light noun, or a bare
    gerund naming a discipline ('studying engineering')."""
    if tok is None:
        return False
    if tok.pos_ == "VERB":
        return tok.tag_ == "VBG" and not any(c.dep_ in ("dobj", "prep", "xcomp", "ccomp") for c in tok.children)
    return tok.pos_ in ("NOUN", "PROPN") and otype in ("Thing", "NamedEntity", "Org") and tok.lemma_.lower() not in STUDY_NON_FIELDS \
        and tok.lemma_.lower() not in LIGHT_NOUNS and tok.ent_type_ not in _NUMERIC_ENTS


def _guard(pred: str, lemma_: str, prep: str | None, otype: str, tok=None) -> str:
    """A FUNCTIONAL relation fires only on the object types that license it (critique B(iv)); any other object
    gets the non-functional predicate on the right:
      lives_in / moved_to            Place            else live_in / move_to      ("lives in a small apartment", "moved to the couch")
      stayed_in (stay / be + in)     Place            else stay_in / is_in        ("stays in shape")
      born_in                        Place            else born_on (a date), is_from / born otherwise
      works_at                       Org/Place/name   else work_at / work_in / work_for ("works at a car maintenance shop")
      works_as                       role noun        else work_as / is_as
      married_to/dating/engaged_to   person           else the bare lemma (+ preposition): marry, date, engage_to
      studies                        field            else study"""
    if pred not in TYPE_GUARDED:
        return pred
    bare = _BARE.get(lemma_, lemma_)
    joined = f"{bare}_{prep}" if prep else bare
    if pred in ("lives_in", "moved_to", "stayed_in"):
        return pred if otype == "Place" else joined
    if pred == "born_in":
        if otype == "Place":
            return pred
        if otype == "Date" or (tok is not None and (tok.pos_ == "NUM" or tok.ent_type_ in ("DATE", "TIME"))):
            return "born_on"
        return "is_from" if lemma_ == "be" else "born"
    if pred == "works_at":
        return pred if otype in EMPLOYER_TYPES else joined
    if pred == "works_as":
        return pred if _is_role_noun(tok, otype) else joined
    if pred in COUPLE_PREDICATES:
        return pred if otype in PERSON_TYPES else joined
    if pred == "studies":
        return pred if _is_field(tok, otype) else joined
    return pred


def _activity_predicate(nlp_lemma: str) -> str:
    """Predicate for a verbal / gerund complement: 'worked ... building websites' is works_on, else the folded lemma."""
    return "works_on" if nlp_lemma == "work" else canonical_predicate(nlp_lemma)


# ---------------------------------------------------------------------------
# Fallback rule (the original, unchanged)
# ---------------------------------------------------------------------------
def _rule_sentence_to_triples(sentence: str, default_subject: str, known_names: Iterable[str] = ()) -> list[tuple[str, str, str]]:
    text = sentence.strip().rstrip(".")
    toks = re.findall(r"[A-Za-z0-9'\-]+", text)
    if not toks:
        return []
    names = {n.lower(): n for n in known_names}
    subj = default_subject
    start = 0
    possessive = False
    head = toks[0].lower()
    if head.endswith("'s") and head[:-2] in names:            # "Caroline's dog Max ..." -> (Caroline, has_dog, Max ...)
        subj, start, possessive = names[head[:-2]], 1, True
    elif head in names:
        subj = names[head]
        start = 1
    elif head in ("i", "my", "we", "i'm", "i've", "our"):
        start = 1
        possessive = head in ("my", "our")
    pi = None
    for i in range(start, len(toks)):
        w = toks[i].lower()
        if w in BE or (w not in STOP and len(w) > 2):
            pi = i
            break
    if pi is None:
        return []
    pred = ("has_" + toks[pi].lower()) if possessive else _rule_lemma(toks[pi])
    obj_toks = [t for t in toks[pi + 1: pi + 9] if t.lower() not in STOP][:6]
    if not obj_toks:
        return []
    obj = " ".join(obj_toks)
    out = [(subj, pred, obj)]
    for m in DATE_RE.finditer(text):
        if m.group(0).lower() not in obj.lower():
            out.append((subj, "mentions_date", m.group(0)))
    for t in toks[pi + 1:]:
        if t[:1].isupper() and t.lower() in names and names[t.lower()] != subj:
            out.append((subj, "mentions", names[t.lower()]))
    return out


# ---------------------------------------------------------------------------
# spaCy rule
# ---------------------------------------------------------------------------
_QUOTES = "\"'“”‘’`"
_SUBJ_PRON = {"i", "we", "she", "he", "they", "you"}
_POSS_PRON = {"my", "our", "her", "his", "their", "your"}


def _children(tok, *deps):
    return [c for c in tok.children if c.dep_ in deps]


def _first(tok, *deps):
    cs = _children(tok, *deps)
    return cs[0] if cs else None


def _named_as(tok):
    """'a puppy named Max' / "a group called 'X'" -> the PROPN token that names it, else None."""
    for acl in _children(tok, "acl"):
        if acl.lemma_ in ("name", "call", "dub"):
            for c in acl.children:
                if c.dep_ in ("oprd", "dobj", "attr", "xcomp") and c.pos_ in ("PROPN", "NOUN", "X"):
                    return c
    return None


def _phrase(tok) -> str:
    """Head noun extended with its compound / adjectival / numeral / proper-noun children (left of the head,
    so 'Boston, MA' gives 'Boston'); common-noun heads are lemmatised so 'kids' and 'kid' coincide."""
    parts = []
    for c in tok.lefts:
        if c.dep_ in ("compound", "nmod", "nummod", "flat") or (c.dep_ == "amod" and c.lemma_.lower() not in NOISE_ADJ and c.pos_ in ("ADJ", "NOUN", "PROPN", "VERB")):
            parts.extend([m.text for m in c.lefts if m.dep_ in ("compound", "amod") and m.lemma_.lower() not in NOISE_ADJ] + [c.text])
    for c in tok.rights:
        if c.dep_ in ("compound", "flat") and c.pos_ == "PROPN":
            parts.append(c.text)
    head = tok.text if tok.pos_ in ("PROPN", "NUM") or tok.ent_type_ else tok.lemma_
    parts.append(head)
    return " ".join(p.strip(_QUOTES) for p in parts if p.strip(_QUOTES)).strip()


def _owner(tok, names: dict[str, str], subject: str) -> str | None:
    """Known name or possessive pronoun owning `tok`, resolved to a name (pronouns -> the subject)."""
    poss = _first(tok, "poss")
    if poss is None:
        return None
    if poss.text.lower() in names:
        return names[poss.text.lower()]
    if poss.text.lower() in _POSS_PRON:
        return subject
    return None


def _relation_surface(head_lemma: str) -> str:
    return RELATION_SURFACE.get(head_lemma, head_lemma)


def _appos_name(tok):
    """'her husband John' -> the apposed proper noun (only asked for relation / person nouns, so 'Boston, MA' is untouched)."""
    return next((c for c in tok.children if c.dep_ == "appos" and c.pos_ == "PROPN" and c.ent_type_ in ("PERSON", "")), None)


def _object_of_noun(tok, names: dict[str, str], subject: str) -> tuple[str, str, str | None]:
    """(object string, object type, relation predicate or None) for a nominal complement token. Relation
    nouns give the owner-qualified canonical surface ("Melanie's spouse") unless the sentence names the
    person / pet ("a puppy named Max", "her husband John")."""
    if tok.text.lower() in names:
        return names[tok.text.lower()], "Person", None
    named = _named_as(tok)
    head_lemma = tok.lemma_.lower()
    rel = RELATION_NOUNS.get(head_lemma)
    if named is None and (rel or head_lemma in PERSON_NOUNS):
        named = _appos_name(tok)
    if named is not None:
        name = _phrase(named)
        return names.get(name.lower(), name), (rel[1] if rel else ("Person" if head_lemma in PERSON_NOUNS else _ner_or_thing(named))), (rel[0] if rel else None)
    if rel:
        owner = _owner(tok, names, subject) or subject
        return f"{owner}'s {_relation_surface(head_lemma)}", rel[1], rel[0]
    if head_lemma in PERSON_NOUNS:
        owner = _owner(tok, names, subject)
        return (f"{owner}'s {head_lemma}" if owner else _phrase(tok)), "Person", None
    return _phrase(tok), _ner_or_thing(tok), None


def _ner_or_thing(tok) -> str:
    if tok.ent_type_ in NER_TYPES:
        return NER_TYPES[tok.ent_type_]
    if tok.pos_ == "NUM":
        return "Thing"
    return "NamedEntity" if tok.pos_ == "PROPN" or tok.text[:1].isupper() else "Thing"


def _verb_object_string(v, intent: bool) -> str:
    """Object string for a verbal complement: infinitive + object head ('continue education') for intent verbs,
    the bare activity ('camping', 'painting') otherwise."""
    obj = _first(v, "dobj", "attr", "pobj")
    if obj is None:
        prep = _first(v, "prep")
        if prep is not None:
            obj = _first(prep, "pobj")
    if intent:
        s = v.lemma_.lower()
        if obj is not None:
            s += " " + (_phrase(obj) if obj.pos_ != "PRON" else obj.text.lower())
        else:
            nested = _first(v, "xcomp", "ccomp")                        # "continue volunteering"
            if nested is not None and nested.pos_ == "VERB":
                s += " " + _verb_object_string(nested, False)
        return s
    if obj is not None and obj.pos_ in ("NOUN", "PROPN") and v.lemma_.lower() in ACTIVITY_VERBS:
        return _phrase(obj)
    return v.text.lower()


def _clause_string(tok, limit: int = 6) -> str:
    """A proposition ("talking about inclusivity is crucial") as a short content-token string, the way the
    fallback rule chunks objects; such objects do not recur and are kept only so the predicate survives."""
    toks = [t.text for t in tok.subtree if t.text.lower() not in STOP and not t.is_punct]
    return " ".join(toks[:limit])


def _relation_object(nlp_lemma: str, rel_pred: str | None, obj: str, otype: str):
    """A relation-noun complement (rel_pred = its predicate from _object_of_noun) states the relation itself:
    'has / adopted a dog named Max' -> has_pet Max, 'married her partner / fiance' -> married_to "<Owner>'s
    spouse". None when the verb says nothing of the kind."""
    if not rel_pred:
        return None
    if nlp_lemma in POSSESSION_VERBS:
        return rel_pred, obj, otype
    if nlp_lemma in ("marry", "wed") and rel_pred in COUPLE_PREDICATES:
        return "married_to", re.sub(r"'s (?:partner|fiance|spouse)$", "'s spouse", obj), "Person"
    return None


def _complement(verb, names, subject, nlp_lemma: str):
    """Pick the primary complement of the main verb. Returns (pred, obj, o_type) or None.
    Priority: direct object > infinitival/gerund complement > prepositional object > predicate adjective.
    Every mapped predicate passes _guard, so a FUNCTIONAL relation needs an object of the right type."""
    # direct object
    dobj = _first(verb, "dobj")
    has_prep_obj = any(_first(p, "pobj", "pcomp") is not None for p in _children(verb, "prep"))
    if dobj is not None and dobj.pos_ != "PRON":
        if dobj.pos_ == "VERB":                         # "enjoys painting animals"
            return _guard(_activity_predicate(nlp_lemma), nlp_lemma, None, "Thing", dobj), _verb_object_string(dobj, nlp_lemma in INTENT_VERBS), "Thing"
        obj, otype, rel = _object_of_noun(dobj, names, subject)
        relp = _relation_object(nlp_lemma, rel, obj, otype)
        if relp is not None:
            return relp
        if nlp_lemma in ("turn", "be") and dobj.pos_ == "NUM":
            return "has_age", dobj.text, "Thing"
        light = dobj.lemma_.lower() in LIGHT_NOUNS
        if light:                                       # "the importance of X" -> X
            of = next((p for p in _children(dobj, "prep") if p.lemma_ == "of"), None)
            pobj = _first(of, "pobj", "pcomp") if of is not None else None
            if pobj is not None and pobj.pos_ != "PRON":
                if pobj.pos_ == "VERB":
                    return _guard(_activity_predicate(nlp_lemma), nlp_lemma, None, "Thing", pobj), _verb_object_string(pobj, False), "Thing"
                obj, otype, rel = _object_of_noun(pobj, names, subject)
                return _guard(canonical_predicate(nlp_lemma), nlp_lemma, None, otype, pobj), obj, otype
        # a possessed companion ("took her kids to a workshop") or an empty head ("spends time with friends") defers
        # to a non-temporal prepositional object when there is one
        companion = bool(rel) and otype == "Person" and nlp_lemma != "have" and _first(dobj, "poss") is not None
        if not (companion or light) or not has_prep_obj:
            return _guard(canonical_predicate(nlp_lemma), nlp_lemma, None, otype, dobj), obj, otype
        deferred = dobj
    else:
        deferred = None
    # clausal complement: for an attitude verb the proposition, chunked ("believes talking about inclusivity is
    # crucial"); otherwise the small clause's subject noun is the object the parser mis-attached ("made a
    # stained glass window showcasing ...")
    cc = _first(verb, "ccomp")
    if cc is not None and (cc.pos_ == "AUX" or (cc.pos_ == "VERB" and _first(cc, "nsubj", "nsubjpass") is not None)):
        cs = _first(cc, "nsubj", "nsubjpass")
        if canonical_predicate(nlp_lemma) not in ATTITUDE_PREDICATES and cs is not None and cs.pos_ in ("NOUN", "PROPN"):
            obj, otype, _ = _object_of_noun(cs, names, subject)
            return _guard(canonical_predicate(nlp_lemma), nlp_lemma, None, otype, cs), obj, otype
        return _guard(canonical_predicate(nlp_lemma), nlp_lemma, None, "Thing", None), _clause_string(cc), "Thing"
    # verbal complement: "plans to continue ...", "started playing ...", "went camping", "is looking forward to attending ..."
    xc = _first(verb, "xcomp", "ccomp", "advcl", "pcomp")
    if xc is not None and xc.pos_ == "VERB" and (xc.dep_ != "advcl" or xc.tag_ == "VBG"):
        if nlp_lemma == "look" and any(c.lemma_ == "forward" for c in verb.children):
            return "plans_to", _verb_object_string(xc, True), "Thing"
        if nlp_lemma in INTENT_VERBS:
            return canonical_predicate(nlp_lemma), _verb_object_string(xc, True), "Thing"
        if nlp_lemma in ("start", "begin", "take", "pick"):
            return "hobby", _verb_object_string(xc, False), "Thing"
        if nlp_lemma == "go":
            return "did", _verb_object_string(xc, False), "Thing"
        return _guard(_activity_predicate(nlp_lemma), nlp_lemma, None, "Thing", xc), _verb_object_string(xc, False), "Thing"
    # prepositional object (first preposition, or "look forward to")
    for prep in _children(verb, "prep", "agent", "dative"):
        pobj = _first(prep, "pobj", "pcomp")
        if pobj is None or (deferred is not None and (pobj.ent_type_ in ("DATE", "TIME") or DATE_RE.fullmatch(pobj.text))):
            continue
        p = prep.lemma_.lower()
        if pobj.pos_ == "VERB":
            if nlp_lemma == "look" and p == "to":
                return "plans_to", _verb_object_string(pobj, True), "Thing"
            return _guard(canonical_predicate(nlp_lemma, p), nlp_lemma, p, "Thing", pobj), _verb_object_string(pobj, False), "Thing"
        if pobj.pos_ == "PRON":
            continue
        obj, otype, rel = _object_of_noun(pobj, names, subject)
        pred = canonical_predicate(nlp_lemma, p)
        if pred == nlp_lemma and nlp_lemma not in CANONICAL_PREDICATES:       # unmapped verb: keep the preposition
            pred = f"{nlp_lemma}_{p}"
        return _guard(pred, nlp_lemma, p, otype, pobj), obj, otype
    if deferred is not None:                                                # deferred direct object, no usable preposition
        obj, otype, _ = _object_of_noun(deferred, names, subject)
        return _guard(canonical_predicate(nlp_lemma), nlp_lemma, None, otype, deferred), obj, otype
    # object predicative / nominal attribute of a non-copular verb ("became a counselor")
    attr = _first(verb, "attr", "oprd")
    if attr is not None and attr.pos_ != "PRON":
        if attr.pos_ == "NUM":
            return "has_age", attr.text, "Thing"
        obj, otype, _ = _object_of_noun(attr, names, subject)
        return ("is_a" if nlp_lemma in ("become", "be", "remain") else _guard(canonical_predicate(nlp_lemma), nlp_lemma, None, otype, attr)), obj, otype
    acomp = _first(verb, "acomp")
    if acomp is not None:
        return _acomp_triple(acomp, names, subject)
    return None


def _acomp_triple(acomp, names, subject):
    adj = acomp.lemma_.lower() if acomp.pos_ != "VERB" else acomp.text.lower()
    # "32 years old"
    if adj == "old":
        for c in acomp.subtree:
            if c.pos_ == "NUM":
                return "has_age", c.text, "Thing"
    prep = _first(acomp, "prep")
    if prep is not None:
        pobj = _first(prep, "pobj", "pcomp")
        if pobj is not None:
            p = prep.lemma_.lower()
            key = (adj, p) if (adj, p) in ACOMP_PREDICATES else (acomp.lemma_.lower(), p)
            if key in ACOMP_PREDICATES and pobj.pos_ != "PRON":
                obj, otype, _ = _object_of_noun(pobj, names, subject) if pobj.pos_ != "VERB" else (_verb_object_string(pobj, False), "Thing", None)
                pred = ACOMP_PREDICATES[key]
                guarded = _guard(pred, acomp.lemma_.lower(), p, otype, pobj if pobj.pos_ != "VERB" else None)
                if guarded == pred:
                    return pred, obj, otype
                if acomp.pos_ == "VERB":                                         # "got married to a woman" -> (marry, woman)
                    return acomp.lemma_.lower(), obj, otype
                # "is married to her work": a figure of speech, kept as a feeling like any other predicate adjective
            tail = _verb_object_string(pobj, False) if pobj.pos_ == "VERB" else (_phrase(pobj) if pobj.pos_ != "PRON" else pobj.text.lower())
            return "feels", f"{adj} {prep.text.lower()} {tail}", "Thing"
    return "feels", adj, "Thing"


def _copula_triple(root, names, subject):
    """Root is a copular 'be': attr -> is_a (or likes for 'fan of X'), acomp -> feels / mapped, prep -> mapped
    ("is in Japan" -> stayed_in, never lives_in; "is from Texas" -> born_in; anything else is_<prep>)."""
    attr = _first(root, "attr")
    if attr is not None and attr.pos_ != "PRON":
        if attr.pos_ == "NUM" or (attr.lemma_ == "year" and _first(attr, "nummod") is not None):
            n = attr if attr.pos_ == "NUM" else _first(attr, "nummod")
            return "has_age", n.text, "Thing"
        if attr.lemma_.lower() in LIKER_NOUNS:
            of = _first(attr, "prep")
            pobj = _first(of, "pobj") if of is not None else None
            if pobj is not None and pobj.pos_ != "PRON":
                obj, otype, _ = _object_of_noun(pobj, names, subject) if pobj.pos_ != "VERB" else (_verb_object_string(pobj, False), "Thing", None)
                return "likes", obj, otype
        obj, otype, rel = _object_of_noun(attr, names, subject)
        return "is_a", obj, otype
    acomp = _first(root, "acomp")
    if acomp is not None:
        return _acomp_triple(acomp, names, subject)
    for prep in _children(root, "prep"):
        pobj = _first(prep, "pobj", "pcomp")
        if pobj is None or pobj.pos_ == "PRON":
            continue
        p = prep.lemma_.lower()
        pred = canonical_predicate("be", p)
        if pred == "be":
            pred = "is_" + p
        if pobj.pos_ == "VERB":
            return _guard(pred, "be", p, "Thing", None), _verb_object_string(pobj, False), "Thing"
        obj, otype, _ = _object_of_noun(pobj, names, subject)
        return _guard(pred, "be", p, otype, pobj), obj, otype
    return None


def _main_verb(doc):
    """The sentence root if verbal / copular; otherwise the first finite VERB or AUX."""
    root = doc[:].root
    if root.pos_ in ("VERB", "AUX"):
        return root
    for t in doc:
        if t.pos_ in ("VERB", "AUX") and t.dep_ in ("ROOT", "ccomp", "conj", "relcl", "acl"):
            return t
    return None


def _subject(verb, names: dict[str, str], default_subject: str) -> tuple[str, object | None]:
    """(subject name, nominal-subject token or None). A known name heading or owning the subject wins;
    pronouns and anything else fall back to the speaker."""
    ns = _first(verb, "nsubj", "nsubjpass", "csubj")
    if ns is None and verb.dep_ != "ROOT":                     # conj / xcomp verbs inherit the head's subject
        ns = _first(verb.head, "nsubj", "nsubjpass")
    if ns is None:                                             # parser hung the subject on the auxiliary ("Caroline and Sam are engaged")
        for aux in _children(verb, "aux", "auxpass"):
            ns = _first(aux, "nsubj", "nsubjpass")
            if ns is not None:
                break
    if ns is None:
        return default_subject, None
    if ns.text.lower() in names:
        return names[ns.text.lower()], ns
    for c in ns.children:
        if c.dep_ == "poss" and c.text.lower() in names:
            return names[c.text.lower()], ns
    for c in ns.subtree:
        if c.ent_type_ == "PERSON" and c.text.lower() in names:
            return names[c.text.lower()], ns
    return default_subject, ns


def _couple_triple(verb, ns, names: dict[str, str]):
    """'Caroline and Sam are engaged / got married / are dating' -> (engaged_to | married_to | dating, Sam): the
    person conjoined with the subject is the partner when the verb or its predicate adjective is a couple word
    that names no one else."""
    word = verb.lemma_.lower()
    if word in COUPLE_VERBS:
        if _first(verb, "dobj") is not None or any(_first(p, "pobj") is not None for p in _children(verb, "prep")):
            return None
    else:
        ac = _first(verb, "acomp")
        word = {"married": "marry", "engaged": "engage"}.get(ac.lemma_.lower(), ac.lemma_.lower()) if ac is not None else ""
        if word not in COUPLE_VERBS or _first(ac, "prep") is not None:
            return None
    partner = next((c for c in ns.children if c.dep_ == "conj" and (c.text.lower() in names or c.ent_type_ == "PERSON" or c.pos_ == "PROPN")), None)
    if partner is None:
        return None
    return COUPLE_VERBS[word], names.get(partner.text.lower(), _phrase(partner)), "Person"


def _spacy_sentence_to_triples(doc, default_subject: str, known_names: Iterable[str]) -> list[tuple[str, str, str, str]] | None:
    names = {n.lower(): n for n in known_names}
    verb = _main_verb(doc)
    if verb is None:
        return None
    subject, ns = _subject(verb, names, default_subject)
    vlemma = verb.lemma_.lower()
    primary = None
    plain_noun_subject = ns is not None and ns.pos_ in ("NOUN", "PROPN") and ns.text.lower() not in names and ns.text.lower() not in _SUBJ_PRON
    # possessive non-pronoun subject that is a relation noun: "Melanie's kids are excited ..." -> (Melanie, has_child, Melanie's child);
    # "Melanie's wedding was ..." (past) -> (Melanie, married_to, Melanie's spouse)
    if plain_noun_subject:
        head = ns.lemma_.lower()
        rel = RELATION_NOUNS.get(head)
        owner = _owner(ns, names, subject)
        if rel:
            named = _named_as(ns)
            if named is None and vlemma in ("name", "call"):                 # "My dog is named Rex"
                named = _first(verb, "oprd", "attr", "dobj")
            if named is None:
                named = _appos_name(ns)                                        # "Caroline's husband John works ..."
            obj = _phrase(named) if named is not None and named.pos_ in ("PROPN", "NOUN", "X") else f"{owner or subject}'s {_relation_surface(head)}"
            primary = (rel[0], names.get(obj.lower(), obj), rel[1])
        elif head == "wedding" and owner and (verb.tag_ in ("VBD", "VBN") or any(c.tag_ == "VBD" for c in verb.children if c.dep_ in ("aux", "auxpass"))):
            primary = ("married_to", f"{owner}'s spouse", "Person")
    if primary is None and ns is not None:                                     # "Caroline and Sam are engaged."
        primary = _couple_triple(verb, ns, names)
    if primary is None:
        if vlemma == "be" and verb.pos_ == "AUX" and not _children(verb, "dobj"):
            if plain_noun_subject:
                # the complement describes the subject noun, not the speaker: "Running has been great ..." ->
                # (Melanie, has, running); "Caroline's favorite book is X" -> (Caroline, likes, X)
                fav = any(c.dep_ == "amod" and c.lemma_.lower() in ("favorite", "favourite") for c in ns.children)
                attr = _first(verb, "attr")
                if fav and attr is not None and attr.pos_ != "PRON":
                    obj, otype, _ = _object_of_noun(attr, names, subject)
                    primary = ("likes", obj, otype)
                else:
                    obj, otype, _ = _object_of_noun(ns, names, subject)
                    primary = ("has", obj, otype)
            else:
                primary = _copula_triple(verb, names, subject)
        else:
            primary = _complement(verb, names, subject, vlemma)
            if primary is None:                                                # "is networking and updating his business plan"
                for cv in _children(verb, "conj"):
                    if cv.pos_ == "VERB":
                        primary = _complement(cv, names, subject, cv.lemma_.lower())
                        if primary is not None:
                            break
            if primary is None and plain_noun_subject:
                obj, otype, _ = _object_of_noun(ns, names, subject)
                primary = (_guard(canonical_predicate(vlemma), vlemma, None, otype, ns), obj, otype)
    if primary is None:
        return None
    pred, obj, otype = primary
    if not obj or obj.lower() == subject.lower():
        return None
    if _first(verb, "neg") is not None:
        pred = {"likes": "dislikes", "dislikes": "likes"}.get(pred, "not_" + pred)
    out = [(subject, pred, obj, otype)]
    text = doc.text
    for m in DATE_RE.finditer(text):
        if m.group(0).lower() not in obj.lower():
            out.append((subject, "mentions_date", m.group(0), "Date"))
    seen = {obj.lower(), subject.lower()}
    for ent in doc.ents:
        if ent.label_ not in MENTION_LABELS:
            continue
        name = re.sub(r"^(?:the|a|an)\s+", "", ent.text.strip(_QUOTES).strip(), flags=re.I)
        if not name or name.lower() in seen or name.lower() in obj.lower() or obj.lower() in name.lower() or ent.label_ == "PERSON" and len(name) < 3:
            continue
        etype = "Person" if name.lower() in names else NER_TYPES.get(ent.label_, "Thing")
        if ent.label_ == "PERSON" and ent.root.dep_ == "oprd" and ent.root.head.lemma_ == "name" and RELATION_NOUNS.get(ent.root.head.head.lemma_.lower(), ("", ""))[1] == "Pet":
            etype = "Pet"
        seen.add(name.lower())
        out.append((subject, "mentions", names.get(name.lower(), name), etype))
    for n_low, n in names.items():                              # known names NER missed
        if n_low not in seen and re.search(rf"\b{re.escape(n)}\b", text) and n_low not in obj.lower():
            seen.add(n_low)
            out.append((subject, "mentions", n, "Person"))
    return out


_CACHE: dict[tuple[str, str, tuple[str, ...]], tuple[list[tuple[str, str, str, str]], bool]] = {}
# Fallback accounting (distinct (sentence, subject, names) keys, i.e. cache misses): on the spaCy path a sentence
# whose parse yields no triple takes the original token rule; FALLBACK_SENTENCES / EXTRACTED_SENTENCES is the
# share. Under PAPER2_EXTRACT_BACKEND=rule (or no model) the rule is the backend, not a fallback, and stays at 0.
FALLBACK_SENTENCES: int = 0
EXTRACTED_SENTENCES: int = 0


def _extract(sentence: str, default_subject: str, known_names: Iterable[str]) -> tuple[list[tuple[str, str, str, str]], bool]:
    """(typed triples, took_rule_fallback) for one sentence, memoised."""
    global FALLBACK_SENTENCES, EXTRACTED_SENTENCES
    key = (sentence, default_subject, tuple(known_names))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    out: list[tuple[str, str, str, str]] | None = None
    fell_back = False
    nlp = _load_spacy()
    text = sentence.strip()
    if nlp is not None and text:
        try:
            out = _spacy_sentence_to_triples(nlp(text), default_subject, key[2])
        except Exception:        # a parse the rule code did not foresee: fall back for this sentence only
            out = None
        fell_back = out is None
    if out is None:
        out = [(s, p, o, entity_type(o)) for s, p, o in _rule_sentence_to_triples(sentence, default_subject, key[2])]
    EXTRACTED_SENTENCES += 1
    if fell_back:
        FALLBACK_SENTENCES += 1
    if len(_CACHE) > 200_000:
        _CACHE.clear()
    _CACHE[key] = (out, fell_back)
    return out, fell_back


def sentence_to_typed_triples(sentence: str, default_subject: str, known_names: Iterable[str] = ()) -> list[tuple[str, str, str, str]]:
    """Like sentence_to_triples but each tuple carries the object type as a fourth field."""
    return _extract(sentence, default_subject, known_names)[0]


def sentence_to_triples(sentence: str, default_subject: str, known_names: Iterable[str] = ()) -> list[tuple[str, str, str]]:
    """Deterministic rule: one primary (subject, predicate, object) per sentence plus 'mentions' /
    'mentions_date' extras. Subject = the known name heading (or owning) the grammatical subject, else the
    speaker. See the module docstring for the two backends; `backend()` says which is active."""
    return [(s, p, o) for s, p, o, _ in sentence_to_typed_triples(sentence, default_subject, known_names)]


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------
def _confidence(tier: str, confidence: float | None) -> float:
    return float(confidence) if confidence is not None else DEFAULT_TIER_CONFIDENCE.get(tier, 0.8)


class OracleObservationExtractor:
    """LoCoMo only: the dataset's GPT-4 per-session observations become candidates, one candidate set per
    turn. They are LLM summaries written with the whole session visible, hence tier 'llm-summary' (0.6)."""

    def __init__(self, tier: str = "llm-summary", confidence: float | None = None):
        self.tier, self.confidence = tier, _confidence(tier, confidence)

    def candidates(self, conv: Conversation) -> Iterator[tuple[Turn, list[Triple]]]:
        by_turn: dict[str, list[Observation]] = {}
        for ob in conv.observations:
            for tid in ob.turn_ids:
                by_turn.setdefault(tid, []).append(ob)
        for turn in conv.turns():
            obs = by_turn.get(turn.id, [])
            triples: list[Triple] = []
            for ob in obs:
                typed, fell_back = _extract(ob.sentence, ob.speaker, conv.speakers)
                method = "oracle_observation_rule_fallback" if fell_back else "oracle_observation"
                for s, p, o, ot in typed:
                    triples.append(Triple(s=s, p=p, o=o, s_type="Person", o_type=ot, source_turns=ob.turn_ids,
                                          session=turn.session, speaker=ob.speaker, observed_at=turn.at, tier=self.tier,
                                          confidence=self.confidence, method=method, text=ob.sentence))
            yield turn, triples


class HeuristicExtractor:
    """Any dataset: first-person sentences in user turns become (user, pred, obj) triples. A placeholder
    that keeps the pipeline LLM-free; the paper's LLM arm replaces it. Tier 'extractor-inferred' (0.8)."""

    def __init__(self, speakers: Iterable[str] = ("user",), tier: str = "extractor-inferred", confidence: float | None = None):
        self.speakers, self.tier = set(speakers), tier
        self.confidence = _confidence(tier, confidence)

    def candidates(self, conv: Conversation) -> Iterator[tuple[Turn, list[Triple]]]:
        for turn in conv.turns():
            triples: list[Triple] = []
            if turn.speaker in self.speakers or turn.speaker in conv.speakers:
                for sent in re.split(r"(?<=[.!?])\s+", turn.text):
                    if re.match(r"^(I|I'm|I've|My|We|Our)\b", sent.strip()):
                        subj = turn.speaker if turn.speaker != "user" else "user"
                        typed, fell_back = _extract(sent, subj, conv.speakers)
                        method = "heuristic_rule_fallback" if fell_back else "heuristic"
                        for s, p, o, ot in typed:
                            triples.append(Triple(s=s, p=p, o=o, o_type=ot, source_turns=[turn.id], session=turn.session,
                                                  speaker=turn.speaker, observed_at=turn.at, tier=self.tier, confidence=self.confidence,
                                                  method=method, text=sent.strip()))
            yield turn, triples


class LLMExtractor:
    """Schema-constrained extraction through paper2.llm; one call per turn, cached on disk."""

    PROMPT = ("Extract the durable facts about the speaker from this conversational turn as JSON: "
              '{"triples": [{"subject": ..., "predicate": ..., "object": ..., "confidence": 0-1}]}. '
              "Use the speaker's name as subject for first-person statements, short lowercase predicates "
              "(lives_in, works_at, likes, has_pet, plans_to, attended, ...), and concrete objects. "
              "Return an empty list for small talk.\n\nSpeaker: {speaker}\nTurn: {text}")

    def __init__(self, client, tier: str = "extractor-inferred", speakers: Iterable[str] | None = None):
        self.client, self.tier, self.speakers = client, tier, (set(speakers) if speakers else None)

    def candidates(self, conv: Conversation) -> Iterator[tuple[Turn, list[Triple]]]:
        import json
        for turn in conv.turns():
            if self.speakers and turn.speaker not in self.speakers:
                yield turn, []
                continue
            raw = self.client.complete_json(self.PROMPT.format(speaker=turn.speaker, text=turn.text), cache_key=f"extract:{conv.id}:{turn.id}")
            triples: list[Triple] = []
            for t in (raw or {}).get("triples", []):
                try:
                    pred = str(t["predicate"]).lower().replace(" ", "_")
                    triples.append(Triple(s=str(t["subject"]), p=CANONICAL_PREDICATES.get(pred, pred), o=str(t["object"]),
                                          o_type=entity_type(str(t["object"])), source_turns=[turn.id], session=turn.session, speaker=turn.speaker,
                                          observed_at=turn.at, tier=self.tier, confidence=float(t.get("confidence", 0.7)), method="llm", text=turn.text))
                except (KeyError, TypeError, ValueError):
                    continue
            yield turn, triples


# ---------------------------------------------------------------- self-check (python -m paper2.extract)
# (sentence, speaker, expected predicate or None when the parser hands the sentence to the rule fallback)
_SELFCHECK = [
    ("Melanie has a busy life.", "Melanie", "has"),
    ("Caroline works two jobs.", "Caroline", "work"),
    ("Caroline works night shifts.", "Caroline", None),
    ("Melanie moved to the couch.", "Melanie", "move_to"),
    ("Melanie is dating a letter.", "Melanie", "date"),
    ("Tim will be staying in Galway.", "Tim", "stayed_in"),
    ("Calvin will be in Japan for a few months.", "Calvin", "stayed_in"),
    ("Evan stays in shape.", "Evan", "stay_in"),
    ("Caroline was in Paris last week for a conference.", "Caroline", "stayed_in"),
    ("Caroline is married to her work.", "Caroline", "feels"),
    ("Caroline is married to Sam.", "Caroline", "married_to"),
    ("Caroline is engaged to Sam.", "Caroline", "engaged_to"),
    ("Caroline got engaged to Sam last month.", "Caroline", "engaged_to"),
    ("Caroline lives in Boston.", "Caroline", "lives_in"),
    ("Melanie moved to Chicago in 2023.", "Melanie", "moved_to"),
    ("James works at Dell.", "James", "works_at"),
    ("James works as a nurse.", "James", "works_as"),
]
_SELFCHECK_FUNCTIONAL = {"Caroline is married to Sam.", "Caroline is engaged to Sam.", "Caroline got engaged to Sam last month.",
                         "Caroline lives in Boston.", "Melanie moved to Chicago in 2023.", "James works at Dell.", "James works as a nurse."}


def _selfcheck(verbose: bool = True) -> None:
    """Assert the functional-predicate contract on the probe sentences (critique B(iv) / E2(b)); raises on failure."""
    names = ["Caroline", "Melanie", "Tim", "Calvin", "Evan", "James"]
    if backend() != "spacy":
        raise RuntimeError(f"self-check needs the spaCy backend, active backend is {EXTRACTION_BACKEND!r}")
    functional = set()
    for sentence, speaker, expected in _SELFCHECK:
        typed, fell_back = _extract(sentence, speaker, names)
        s, p, o, ot = typed[0]
        flag = "FUNCTIONAL" if p in FUNCTIONAL_PREDICATES else ""
        if verbose:
            print(f"{sentence:55s} -> ({s}, {p}, {o!r}, {ot}){'  [rule fallback]' if fell_back else ''}  {flag}")
        if p in FUNCTIONAL_PREDICATES:
            functional.add(sentence)
        if expected is not None:
            assert p == expected, (sentence, p, expected)
    assert functional == _SELFCHECK_FUNCTIONAL, ("functional sentences", sorted(functional ^ _SELFCHECK_FUNCTIONAL))
    # relation nouns: one canonical spouse / partner object per owner; marrying one's partner makes them the spouse
    husband = sentence_to_typed_triples("Melanie has a husband and kids who keep her motivated.", "Melanie", names)[0]
    marrying = sentence_to_typed_triples("Melanie's favorite part of her wedding was marrying her partner.", "Melanie", names)[0]
    wife = sentence_to_typed_triples("Melanie has a wife.", "Melanie", names)[0]
    assert husband[1:] == marrying[1:] == wife[1:] == ("married_to", "Melanie's spouse", "Person"), (husband, marrying, wife)
    assert sentence_to_typed_triples("Melanie has a partner.", "Melanie", names)[0][1:] == ("dating", "Melanie's partner", "Person")
    # non-person objects never reach a couple relation; a date is not a birthplace
    assert sentence_to_typed_triples("Evan got married to a woman from Canada.", "Evan", names)[0][1] == "marry"
    assert sentence_to_typed_triples("Caroline was born in 1990.", "Caroline", names)[0][1] == "born_on"
    assert sentence_to_typed_triples("Caroline was born in Texas.", "Caroline", names)[0][1] == "born_in"
    # engaged_to is reachable from a conjoined subject too; adopt + Pet -> has_pet; studies needs a field
    assert sentence_to_typed_triples("Caroline and Sam are engaged.", "Caroline", names)[0][1:] == ("engaged_to", "Sam", "Person")
    assert sentence_to_typed_triples("Caroline adopted a puppy named Max in March 2023.", "Caroline", names)[0][1:] == ("has_pet", "Max", "Pet")
    assert sentence_to_typed_triples("Jolene is studying hard for upcoming finals.", "Jolene", names)[0][1] == "study_for"
    assert sentence_to_typed_triples("Jolene is studying engineering in school.", "Jolene", names)[0][1:3] == ("studies", "engineering")
    # the lemmatiser reads a bare word as a verb
    assert lemma("lives") == "live" and lemma("values") == "value" and lemma("moved") == "move" and lemma("lives", pos="NOUN") == "life", \
        (lemma("lives"), lemma("values"), lemma("moved"), lemma("lives", pos="NOUN"))
    if verbose:
        print(f"self-check passed: {len(functional)} functional sentences as expected; backend {EXTRACTION_BACKEND}; "
              f"fallbacks {FALLBACK_SENTENCES}/{EXTRACTED_SENTENCES} distinct sentences")


if __name__ == "__main__":
    _selfcheck()
