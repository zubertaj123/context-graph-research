"""Scoring: the official LoCoMo QA metric vendored in pure Python, token F1 / exact match, evidence
recall of the retrieved set (hit, frac, MRR), admission precision and recall against gold evidence
turns, and bootstrap / Wilson confidence intervals.

Attribution and provenance of the vendored scorer
-------------------------------------------------
`normalize_answer`, `f1_score` (Porter-stemmed token F1), `f1_multi_answer` (the category-1
comma-split partial F1), `exact_match_official` (token-set equality), the category-3 truncation of
the gold at the first ';', and the category-5 abstention substrings in `OFFICIAL_ABSTAIN` are a
function-by-function port of the official LoCoMo evaluation script

    https://raw.githubusercontent.com/snap-research/locomo/main/task_eval/evaluation.py
    (snap-research/locomo, branch `main`, fetched 2026-10-18;
     SHA-256 of the fetched file 8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd.
     The GitHub commits API was not reachable from the build sandbox, so the file hash stands in
     for the commit id; `task_eval/evaluation.py` has not changed on `main` since the dataset's
     2024 release to our knowledge, but re-check the hash before quoting "official" in the paper.)

The dispatch in `score_answer` mirrors `eval_question_answering` in that script: categories 2, 3
and 4 use plain `f1_score`; category 3 first cuts the gold at the first ';'; category 1 uses
`f1_multi_answer`; category 5 scores 1 iff the lower-cased prediction contains 'no information
available' or 'not mentioned'. The official script reports only F1 for the QA task; `exact_match`
(normalised string equality) is this paper's own additional metric and is kept as it was.

The official `f1_score` stems every token with `nltk.stem.PorterStemmer()` (default
`NLTK_EXTENSIONS` mode). To add no dependency, `PorterStemmer` below is a port of
`nltk/stem/porter.py` (NLTK 3.10.3, Apache License 2.0; Porter, M. "An algorithm for suffix
stripping", Program 14(3), 1980). It was checked token-for-token against nltk 3.10.3 on Martin
Porter's 23,531-word test vocabulary and on every token of every LoCoMo and LongMemEval gold
answer and question (zero differences).

LongMemEval has no official string-match rule (its official metric is an LLM judge), so
`category_id == 0` keeps token F1 / EM against the gold and scores `*_abs` questions by the
lenient abstention list `ABSTAIN`, as before.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np

from .data import QA

# ---------------------------------------------------------------------------
# Abstention phrase lists
# ---------------------------------------------------------------------------

#: Exactly the two substrings the official LoCoMo script checks for category 5 (lower-cased prediction).
OFFICIAL_ABSTAIN = ("no information available", "not mentioned")

#: The pre-existing lenient list. Not official. Used for LongMemEval `*_abs` questions (no official
#: string rule exists there) and reachable through `is_abstention` for diagnostics.
ABSTAIN = ("no information", "not mention", "don't know", "do not know", "cannot answer", "can't answer", "unknown", "no answer",
           "not available", "not provided", "not specified")

#: LoCoMo category ids (the official numbering) and the names the data loader assigns after the A1.1 fix.
LOCOMO_CATEGORY_IDS = {"multi-hop": 1, "temporal": 2, "open-domain": 3, "single-hop": 4, "adversarial": 5}
LOCOMO_CATEGORY_NAMES = {v: k for k, v in LOCOMO_CATEGORY_IDS.items()}


# ---------------------------------------------------------------------------
# Porter stemmer (port of nltk.stem.porter.PorterStemmer, NLTK_EXTENSIONS mode)
# ---------------------------------------------------------------------------

class PorterStemmer:
    """Pure-Python port of `nltk.stem.porter.PorterStemmer` in its default `NLTK_EXTENSIONS` mode,
    which is what the official LoCoMo scorer instantiates as `PorterStemmer()`. Only that mode is
    implemented; see the module docstring for attribution and verification."""

    _VOWELS = frozenset("aeiou")
    _POOL = {"sky": "sky", "skies": "sky", "dying": "die", "lying": "lie", "tying": "tie", "news": "news",
             "innings": "inning", "inning": "inning", "outings": "outing", "outing": "outing",
             "cannings": "canning", "canning": "canning", "howe": "howe", "proceed": "proceed",
             "exceed": "exceed", "succeed": "succeed"}

    @classmethod
    def _flags(cls, word: str) -> list[bool]:
        """True at position i iff word[i] is a consonant: not a vowel, and a 'y' only when it starts the
        word or follows a non-consonant."""
        flags: list[bool] = []
        for i, ch in enumerate(word):
            if ch in cls._VOWELS:
                flags.append(False)
            elif ch == "y":
                flags.append(True if i == 0 else not flags[i - 1])
            else:
                flags.append(True)
        return flags

    def _is_consonant(self, word: str, i: int) -> bool:
        return self._flags(word[: i + 1])[i]

    def _measure(self, stem: str) -> int:
        return "".join("c" if f else "v" for f in self._flags(stem)).count("vc")

    def _m_gt0(self, stem: str) -> bool:
        return self._measure(stem) > 0

    def _m_gt1(self, stem: str) -> bool:
        return self._measure(stem) > 1

    def _contains_vowel(self, stem: str) -> bool:
        return not all(self._flags(stem))

    def _ends_double_consonant(self, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and self._is_consonant(word, len(word) - 1)

    def _ends_cvc(self, word: str) -> bool:
        return (len(word) >= 3 and self._is_consonant(word, len(word) - 3) and not self._is_consonant(word, len(word) - 2)
                and self._is_consonant(word, len(word) - 1) and word[-1] not in ("w", "x", "y")) or (
            len(word) == 2 and not self._is_consonant(word, 0) and self._is_consonant(word, 1))

    def _apply(self, word: str, rules: list[tuple]) -> str:
        """First applicable rule wins; a matching suffix whose condition fails stops the search."""
        for suffix, replacement, condition in rules:
            if suffix == "*d" and self._ends_double_consonant(word):
                stem = word[:-2]
                return stem + replacement if condition is None or condition(stem) else word
            if word.endswith(suffix):
                stem = word[: len(word) - len(suffix)]
                return stem + replacement if condition is None or condition(stem) else word
        return word

    def _step1a(self, word: str) -> str:
        if word.endswith("ies") and len(word) == 4:
            return word[:-3] + "ie"
        return self._apply(word, [("sses", "ss", None), ("ies", "i", None), ("ss", "ss", None), ("s", "", None)])

    def _step1b(self, word: str) -> str:
        if word.endswith("ied"):
            return word[:-3] + ("ie" if len(word) == 4 else "i")
        if word.endswith("eed"):
            stem = word[:-3]
            return stem + "ee" if self._measure(stem) > 0 else word
        for suffix in ("ed", "ing"):
            if word.endswith(suffix):
                inter = word[: -len(suffix)]
                if self._contains_vowel(inter):
                    break
        else:
            return word
        last = inter[-1]
        return self._apply(inter, [("at", "ate", None), ("bl", "ble", None), ("iz", "ize", None),
                                   ("*d", last, lambda stem: last not in ("l", "s", "z")),
                                   ("", "e", lambda stem: self._measure(stem) == 1 and self._ends_cvc(stem))])

    def _step1c(self, word: str) -> str:
        return self._apply(word, [("y", "i", lambda stem: len(stem) > 1 and self._is_consonant(stem, len(stem) - 1))])

    def _step2(self, word: str) -> str:
        if word.endswith("alli") and self._m_gt0(word[:-4]):
            return self._step2(word[:-4] + "al")
        m = self._m_gt0
        return self._apply(word, [("ational", "ate", m), ("tional", "tion", m), ("enci", "ence", m), ("anci", "ance", m),
                                  ("izer", "ize", m), ("bli", "ble", m), ("alli", "al", m), ("entli", "ent", m), ("eli", "e", m),
                                  ("ousli", "ous", m), ("ization", "ize", m), ("ation", "ate", m), ("ator", "ate", m),
                                  ("alism", "al", m), ("iveness", "ive", m), ("fulness", "ful", m), ("ousness", "ous", m),
                                  ("aliti", "al", m), ("iviti", "ive", m), ("biliti", "ble", m), ("fulli", "ful", m),
                                  ("logi", "log", lambda stem: m(word[:-3]))])

    def _step3(self, word: str) -> str:
        m = self._m_gt0
        return self._apply(word, [("icate", "ic", m), ("ative", "", m), ("alize", "al", m), ("iciti", "ic", m), ("ical", "ic", m),
                                  ("ful", "", m), ("ness", "", m)])

    def _step4(self, word: str) -> str:
        m = self._m_gt1
        return self._apply(word, [("al", "", m), ("ance", "", m), ("ence", "", m), ("er", "", m), ("ic", "", m), ("able", "", m),
                                  ("ible", "", m), ("ant", "", m), ("ement", "", m), ("ment", "", m), ("ent", "", m),
                                  ("ion", "", lambda stem: self._measure(stem) > 1 and stem[-1] in ("s", "t")),
                                  ("ou", "", m), ("ism", "", m), ("ate", "", m), ("iti", "", m), ("ous", "", m), ("ive", "", m),
                                  ("ize", "", m)])

    def _step5a(self, word: str) -> str:
        if word.endswith("e"):
            stem = word[:-1]
            if self._measure(stem) > 1:
                return stem
            if self._measure(stem) == 1 and not self._ends_cvc(stem):
                return stem
        return word

    def _step5b(self, word: str) -> str:
        return self._apply(word, [("ll", "l", lambda stem: self._measure(word[:-1]) > 1)])

    def stem(self, word: str) -> str:
        w = word.lower()
        if w in self._POOL:
            return self._POOL[w]
        if len(word) <= 2:
            return w
        for step in (self._step1a, self._step1b, self._step1c, self._step2, self._step3, self._step4, self._step5a, self._step5b):
            w = step(w)
        return w


_STEMMER = PorterStemmer()


def stem(word: str) -> str:
    """Porter stem of one token, as the official LoCoMo scorer computes it."""
    return _STEMMER.stem(word)


# ---------------------------------------------------------------------------
# Normalisation, F1, EM (official forms + the pre-fix forms kept reachable)
# ---------------------------------------------------------------------------

_PUNCT = set(string.punctuation)
_ARTICLES_AND = re.compile(r"\b(a|an|the|and)\b")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """Official LoCoMo `normalize_answer`: drop commas, lower-case, strip ASCII punctuation, blank out the
    tokens a / an / the / and, collapse whitespace (in that order)."""
    s = str(s).replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES_AND.sub(" ", s)
    return " ".join(s.split())


def normalize_answer_legacy(s: str) -> str:
    """The pre-fix normalisation (no 'and' removal). Kept only so earlier numbers can be reproduced."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _token_f1(p: list[str], g: list[str]) -> float:
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_score(pred: str, gold: str) -> float:
    """Official LoCoMo `f1_score`: token F1 over Porter-stemmed tokens of the normalised strings."""
    p = [stem(w) for w in normalize_answer(pred).split()]
    g = [stem(w) for w in normalize_answer(gold).split()]
    return _token_f1(p, g)


def f1_score_legacy(pred: str, gold: str) -> float:
    """The pre-fix token F1 (legacy normalisation, no stemming). Kept only for reproducing earlier numbers."""
    return _token_f1(normalize_answer_legacy(pred).split(), normalize_answer_legacy(gold).split())


def f1_multi_answer(pred: str, gold: str) -> float:
    """Official LoCoMo `f1` (used for category 1, multi-hop): split prediction and gold on commas before
    normalisation; for each gold part take the best `f1_score` over prediction parts; average over gold
    parts. Empty parts (e.g. from a trailing comma) are kept, as in the official script."""
    preds = [p.strip() for p in str(pred).split(",")]
    golds = [g.strip() for g in str(gold).split(",")]
    return float(np.mean([max(f1_score(p, g) for p in preds) for g in golds]))


def exact_match(pred: str, gold: str) -> float:
    """This paper's exact match: equality of the officially normalised strings. (The official script's
    `exact_match_score` is `exact_match_official`; it is defined there but unused by its QA evaluation.)"""
    return float(normalize_answer(pred) == normalize_answer(gold))


def exact_match_official(pred: str, gold: str) -> float:
    """Official LoCoMo `exact_match_score`: equality of the *sets* of normalised tokens."""
    return float(set(normalize_answer(pred).split()) == set(normalize_answer(gold).split()))


def is_abstention(pred: str) -> bool:
    """Lenient abstention check (the pre-fix list). Used for LongMemEval `*_abs` questions and diagnostics;
    not the official LoCoMo rule, which is `is_abstention_official`."""
    p = str(pred).lower()
    return any(a in p for a in ABSTAIN)


def is_abstention_official(pred: str) -> bool:
    """Official LoCoMo category-5 rule: the lower-cased prediction contains one of `OFFICIAL_ABSTAIN`."""
    p = str(pred).lower()
    return any(a in p for a in OFFICIAL_ABSTAIN)


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------

def locomo_category_id(qa) -> int:
    """LoCoMo category id 1..5 of a QA; 0 for LongMemEval. `qa.category_id` in 1..5 (int-coercible) is
    authoritative. Otherwise (attribute missing, 0 — the dataclass default and the LongMemEval code — or
    invalid) the post-fix category name decides: multi-hop=1, temporal=2, open-domain=3, single-hop=4,
    adversarial=5; any other name (the LongMemEval question types) is 0. The loader gives every LoCoMo QA a
    code in 1..5, so for loaded data this is simply `qa.category_id`; the name route only matters for
    hand-built QAs and guards against a LoCoMo question silently falling under the LongMemEval rules."""
    cid = getattr(qa, "category_id", None)
    try:
        cid = int(cid) if cid is not None else 0
    except (TypeError, ValueError):
        cid = 0
    if 1 <= cid <= 5:
        return cid
    return LOCOMO_CATEGORY_IDS.get(str(getattr(qa, "category", "")).strip().lower(), 0)


def official_locomo_score(pred: str, gold: str, category_id: int) -> float:
    """The number `eval_question_answering` in the official script appends for one question, given its
    integer category: 1 -> comma-split partial F1; 2, 4 -> plain F1; 3 -> plain F1 against the gold cut at the
    first ';'; 5 -> 1.0 iff the prediction contains an official abstention substring (the gold is ignored)."""
    pred, gold = str(pred), str(gold)
    if category_id == 5:
        return float(is_abstention_official(pred))
    if category_id == 3:
        gold = gold.split(";")[0].strip()
    if category_id in (2, 3, 4):
        return f1_score(pred, gold)
    if category_id == 1:
        return f1_multi_answer(pred, gold)
    raise ValueError(f"not a LoCoMo category id: {category_id!r}")


def score_answer(pred: str, qa: QA) -> dict:
    """Score one prediction. LoCoMo (category_id 1..5): the official rule for `f1` (see
    `official_locomo_score`), `em` = `exact_match` against the same gold string the official rule uses
    (cut at the first ';' for category 3), `abstained` by the official substrings; for category 5 all three
    equal the official abstention score. LongMemEval (category_id 0): token F1 / EM against the gold, and
    `*_abs` questions (or `qa.adversarial`) scored by the lenient `is_abstention`."""
    pred = "" if pred is None else str(pred)
    cid = locomo_category_id(qa)
    if cid == 0:
        if getattr(qa, "adversarial", False) or str(getattr(qa, "id", "")).endswith("_abs"):
            ok = float(is_abstention(pred))
            return {"f1": ok, "em": ok, "abstained": ok}
        return {"f1": f1_score(pred, qa.answer), "em": exact_match(pred, qa.answer), "abstained": float(is_abstention(pred))}
    if cid == 5:
        ok = float(is_abstention_official(pred))
        return {"f1": ok, "em": ok, "abstained": ok}
    gold = str(qa.answer)
    if cid == 3:
        gold = gold.split(";")[0].strip()
    return {"f1": official_locomo_score(pred, gold, cid), "em": exact_match(pred, gold), "abstained": float(is_abstention_official(pred))}


# ---------------------------------------------------------------------------
# Retrieval and admission diagnostics
# ---------------------------------------------------------------------------

def evidence_recall(retrieved_edges: Iterable, qa: QA) -> dict:
    """Evidence recall of an ordered retrieved set. `hit`: any retrieved fact has a gold evidence turn among
    its `source_turns`; `frac`: share of gold evidence turns covered; `mrr`: reciprocal rank (1-based, in the
    order given) of the first retrieved fact whose `source_turns` intersect the gold evidence, 0.0 if none;
    `first_rank`: that rank, 0 if none. All three are NaN when the question has no gold evidence.
    `n_retrieved` is the number of distinct source turns (unchanged); `n_facts` the number of facts."""
    edges = list(retrieved_edges)
    ev = set(qa.evidence_turns)
    got: set = set()
    first_rank = 0
    for rank, e in enumerate(edges, 1):
        st = set(e.props.get("source_turns", []))
        got |= st
        if first_rank == 0 and st & ev:
            first_rank = rank
    inter = got & ev
    nan = float("nan")
    return {"hit": float(bool(inter)) if ev else nan,
            "frac": (len(inter) / len(ev)) if ev else nan,
            "mrr": (1.0 / first_rank if first_rank else 0.0) if ev else nan,
            "first_rank": first_rank,
            "n_retrieved": len(got),
            "n_facts": len(edges)}


def admission_pr(admitted_turns: set[str], candidate_turns: set[str], evidence_turns: set[str]) -> dict:
    tp = len(admitted_turns & evidence_turns)
    reachable = evidence_turns & candidate_turns
    return {"precision": tp / len(admitted_turns) if admitted_turns else float("nan"),
            "recall": tp / len(evidence_turns) if evidence_turns else float("nan"),
            "recall_reachable": len(admitted_turns & reachable) / len(reachable) if reachable else float("nan"),
            "admit_rate": len(admitted_turns) / len(candidate_turns) if candidate_turns else float("nan"),
            "evidence_coverage": len(reachable) / len(evidence_turns) if evidence_turns else float("nan")}


# ---------------------------------------------------------------------------
# Intervals and summaries
# ---------------------------------------------------------------------------

def wilson(successes: float, n: float, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%). Returns (nan, nan) when n <= 0."""
    if n is None or n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1 - p)) / n + z2 / (4 * n * n)) / denom
    # at p = 0 / p = 1 the bound is exactly 0 / 1; the formula lands a rounding error inside it
    lo = 0.0 if successes <= 0 else max(0.0, centre - half)
    hi = 1.0 if successes >= n else min(1.0, centre + half)
    return lo, hi


def bootstrap_ci(values: Sequence[float], stat: Callable = np.nanmean, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval of `stat` over `values` (NaNs dropped); 2,000 resamples as pre-registered (§10)."""
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    bs = np.array([stat(v[i]) for i in idx])
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def _is_binary(vals: Sequence[float]) -> bool:
    return all(x in (0.0, 1.0) for x in vals)


def summarise(qa_records: list[dict], keys: Sequence[str] = ("hit", "frac", "mrr", "f1", "em")) -> dict:
    """Mean per metric overall and per category, with a 95% bootstrap interval for the overall mean and, for
    0/1-valued metrics (hit, em, abstained, ...), a Wilson 95% interval overall and per category."""
    out: dict = {"n": len(qa_records), "overall": {}, "by_category": defaultdict(dict)}
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in qa_records:
        by_cat[r.get("category", "?")].append(r)
    for k in keys:
        vals = [r[k] for r in qa_records if k in r and r[k] == r[k]]
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        entry = {"mean": float(np.mean(vals)), "ci95": [lo, hi], "n": len(vals)}
        if _is_binary(vals):
            entry["wilson95"] = list(wilson(sum(vals), len(vals)))
        out["overall"][k] = entry
        for c, rs in by_cat.items():
            cv = [r[k] for r in rs if k in r and r[k] == r[k]]
            if cv:
                cell = {"mean": float(np.mean(cv)), "n": len(cv)}
                if _is_binary(cv):
                    cell["wilson95"] = list(wilson(sum(cv), len(cv)))
                out["by_category"][c][k] = cell
    out["by_category"] = dict(out["by_category"])
    return out
