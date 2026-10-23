"""Loaders for LoCoMo (locomo10.json) and LongMemEval (cleaned splits).

Both files are untrusted downloads: they are parsed as JSON only, never executed.

LoCoMo question categories. The official integer codes, verified against locomo10.json (question
counts and mean number of gold evidence turns per question; see docs/02-design-critique.md A1.1):

    code  name          n    evidence turns  shape
    1     multi-hop     282  3.13            aggregates several turns ("What exercises has John done?");
                                             the official scorer uses comma-split partial F1
    2     temporal      321  1.17            "When did/was ..." with a date answer
    3     open-domain    96  2.08            needs world knowledge; gold carries a ';'-separated
                                             rationale, the official scorer keeps the text before ';'
    4     single-hop    841  1.07            one evidence turn, plain token F1
    5     adversarial   446  1.03            unanswerable; the gold is abstention. 444 questions carry
                                             only `adversarial_answer`, 2 carry both keys

`QA.category_id` carries the official code (0 for LongMemEval) so scorers key on the code rather
than on a name; `QA.category` is the human-readable name. An earlier version of this module had
codes 1-4 mapped as single-hop / multi-hop / temporal / open-domain, which is wrong.

Evidence ids: nine of LoCoMo's evidence entries match no `dia_id` (a bare 'D', two ids whose turn
does not exist, a typo 'D:11:26', a zero-padded 'D30:05' and four ';'/space-joined compounds).
`load_locomo` drops entries that match no turn so that recall denominators are exact, counts them
in `DROPPED_EVIDENCE_IDS` / `DROPPED_EVIDENCE` and warns once per load; `repair_evidence=True`
first tries the unambiguous fixes (split compounds, strip the stray ':' and leading zeros) and
keeps the result only when every part is a real turn.

LongMemEval: the pre-registered dataset is LongMemEval-S (`longmemeval_s_cleaned.json`, ~48
sessions per question including distractors), which is the default. The oracle file (answer
sessions only, ~1.9 per question) stays reachable with `variant="oracle"` or an explicit path.
"""
from __future__ import annotations

import ast
import json
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"

# Official LoCoMo category codes -> names (see the module docstring for the verification).
LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
LOCOMO_CATEGORY_IDS = {name: code for code, name in LOCOMO_CATEGORIES.items()}
LOCOMO_ADVERSARIAL = 5

# LongMemEval questions have a `question_type` string, no integer code; QA.category_id is 0 for them.
LONGMEMEVAL_CATEGORY_ID = 0
LONGMEMEVAL_FILES = {"s": "longmemeval_s_cleaned.json", "oracle": "longmemeval_oracle.json"}
LONGMEMEVAL_DEFAULT_VARIANT = "s"


class DataWarning(UserWarning):
    """Raised (as a warning) when a loader has to drop or repair malformed benchmark entries."""


# Bookkeeping for the last load_locomo() call (reset at the start of each call).
DROPPED_EVIDENCE_IDS: int = 0                       # evidence entries dropped because they match no turn
DROPPED_EVIDENCE: list[tuple[str, str, str]] = []   # (conversation id, qa id, raw evidence entry)
REPAIRED_EVIDENCE_IDS: int = 0                      # entries rescued by repair_evidence=True


@dataclass
class Turn:
    id: str                 # LoCoMo dia_id (D3:7) or LongMemEval "<session_id>#<i>"
    session: str
    speaker: str            # LoCoMo speaker name; LongMemEval "user" / "assistant"
    text: str
    at: datetime | None
    has_answer: bool = False


@dataclass
class Session:
    id: str
    at: datetime | None
    turns: list[Turn]


@dataclass
class Observation:
    sentence: str
    speaker: str
    turn_ids: list[str]
    session: str


@dataclass
class QA:
    id: str
    question: str
    answer: str                     # gold text; for LoCoMo category 5 the `adversarial_answer` when `answer` is absent
    evidence_turns: list[str]       # gold evidence turn ids; every id matches a turn of the conversation
    evidence_sessions: list[str]
    category: str                   # human-readable: LOCOMO_CATEGORIES name, or the LongMemEval question_type
    adversarial: bool = False       # LoCoMo category 5; LongMemEval question ids ending in "_abs"
    at: datetime | None = None
    category_id: int = 0            # official LoCoMo code 1..5; 0 for LongMemEval


@dataclass
class Conversation:
    id: str
    speakers: list[str]
    sessions: list[Session]
    observations: list[Observation]
    qa: list[QA]
    source: str                      # "locomo" | "longmemeval"

    def turns(self) -> Iterator[Turn]:
        for s in self.sessions:
            yield from s.turns

    def turn(self, tid: str) -> Turn | None:
        for t in self.turns():
            if t.id == tid:
                return t
        return None


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------
def _parse_locomo_dt(s: str | None) -> datetime | None:
    # "1:56 pm on 8 May, 2023"
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")
    except ValueError:
        return None


_EVIDENCE_ID = re.compile(r"^D:?(\d+):(\d+)$")


def _canonical_evidence_id(token: str) -> str:
    """'D:11:26' -> 'D11:26', 'D30:05' -> 'D30:5'; anything else is returned unchanged."""
    m = _EVIDENCE_ID.match(token)
    return f"D{int(m.group(1))}:{int(m.group(2))}" if m else token


def _clean_evidence(raw_ids: list[Any], turn_ids: set[str], *, repair: bool) -> tuple[list[str], list[str], int]:
    """Keep evidence entries that name a real turn.

    Returns (kept ids in original order, dropped raw entries, number of entries repaired). With
    `repair`, an entry that is not a turn id is split on ';', ',' and whitespace and each part is
    canonicalised; the entry is kept only if every part then names a real turn, otherwise it is
    dropped whole (all-or-nothing, so a partly valid compound is still visible in DROPPED_EVIDENCE).
    """
    kept: list[str] = []
    dropped: list[str] = []
    n_repaired = 0
    for raw in raw_ids:
        s = str(raw).strip()
        if s in turn_ids:
            kept.append(s)
            continue
        if repair:
            parts = [_canonical_evidence_id(p) for p in re.split(r"[;,\s]+", s) if p]
            if parts and all(p in turn_ids for p in parts):
                kept.extend(parts)
                n_repaired += 1
                continue
        dropped.append(s)
    return kept, dropped, n_repaired


def load_locomo(path: Path | str | None = None, *, repair_evidence: bool = False, warn: bool = True) -> list[Conversation]:
    """Load locomo10.json as one Conversation per sample.

    Evidence ids that match no turn are dropped (counted in DROPPED_EVIDENCE_IDS / DROPPED_EVIDENCE;
    a DataWarning summarises them unless `warn=False`). `repair_evidence=True` first applies the
    unambiguous fixes described in the module docstring (REPAIRED_EVIDENCE_IDS counts the rescues).
    `QA.adversarial` is `category == 5`; `QA.answer` is the `answer` field, falling back to
    `adversarial_answer` only when `answer` is absent.
    """
    global DROPPED_EVIDENCE_IDS, DROPPED_EVIDENCE, REPAIRED_EVIDENCE_IDS
    p = Path(path) if path else DATA / "locomo" / "locomo10.json"
    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{p}: expected a JSON list of samples")
    dropped_all: list[tuple[str, str, str]] = []
    n_repaired_all = 0
    out: list[Conversation] = []
    for c in raw:
        conv = c["conversation"]
        speakers = [conv["speaker_a"], conv["speaker_b"]]
        sessions: list[Session] = []
        keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            at = _parse_locomo_dt(conv.get(k + "_date_time"))
            turns = [Turn(id=str(t["dia_id"]), session=k, speaker=t["speaker"], text=t.get("text", ""), at=at) for t in conv[k]]
            sessions.append(Session(id=k, at=at, turns=turns))
        obs: list[Observation] = []
        for sk, by_speaker in (c.get("observation") or {}).items():
            session = sk.replace("_observation", "")
            for spk, items in by_speaker.items():
                for sent, did in items:
                    ids = did if isinstance(did, list) else [did]
                    obs.append(Observation(sentence=sent, speaker=spk, turn_ids=[str(x) for x in ids], session=session))
        turn_session = {t.id: t.session for s in sessions for t in s.turns}
        turn_ids = set(turn_session)
        qa: list[QA] = []
        for i, q in enumerate(c["qa"]):
            qid = f"{c['sample_id']}:q{i}"
            ev, dropped, n_rep = _clean_evidence(q.get("evidence", []) or [], turn_ids, repair=repair_evidence)
            dropped_all.extend((c["sample_id"], qid, d) for d in dropped)
            n_repaired_all += n_rep
            try:
                cid = int(q.get("category"))
            except (TypeError, ValueError):
                cid = 0
            answer = q["answer"] if q.get("answer") is not None else q.get("adversarial_answer", "")
            qa.append(QA(id=qid, question=q["question"], answer=str(answer),
                         evidence_turns=ev, evidence_sessions=sorted({turn_session[e] for e in ev}),
                         category=LOCOMO_CATEGORIES.get(cid, str(q.get("category"))), category_id=cid,
                         adversarial=(cid == LOCOMO_ADVERSARIAL)))
        out.append(Conversation(id=c["sample_id"], speakers=speakers, sessions=sessions, observations=obs, qa=qa, source="locomo"))
    DROPPED_EVIDENCE_IDS = len(dropped_all)
    DROPPED_EVIDENCE = dropped_all
    REPAIRED_EVIDENCE_IDS = n_repaired_all
    if warn and dropped_all:
        detail = ", ".join(f"{conv}/{qid}:{d!r}" for conv, qid, d in dropped_all)
        warnings.warn(f"LoCoMo: dropped {len(dropped_all)} evidence id(s) matching no turn ({detail})"
                      + (f"; repaired {n_repaired_all}" if n_repaired_all else "")
                      + ("" if repair_evidence else "; pass repair_evidence=True to rescue compound / mistyped ids"),
                      DataWarning, stacklevel=2)
    return out


# ---------------------------------------------------------------------------
# LongMemEval (cleaned). Each question carries its own haystack; list fields may be
# stringified Python lists in some releases, hence ast.literal_eval on strings.
# ---------------------------------------------------------------------------
def _lit(x: Any) -> Any:
    if isinstance(x, str) and x[:1] in "[(":
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


def _parse_lme_dt(s: str | None) -> datetime | None:
    # "2023/04/10 (Mon) 23:07"
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%Y/%m/%d (%a) %H:%M")
    except ValueError:
        return None


def _has_answer(t: dict) -> bool:
    v = t.get("has_answer")
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def _lme_conversation(q: dict) -> Conversation:
    """Build one Conversation from one LongMemEval question record (its haystack is the memory stream)."""
    sids = [str(x) for x in _lit(q["haystack_session_ids"])]
    dates = _lit(q["haystack_dates"])
    sess_raw = _lit(q["haystack_sessions"])
    if not (len(sids) == len(dates) == len(sess_raw)):
        raise ValueError(f"{q.get('question_id')}: haystack ids / dates / sessions have different lengths")
    sessions: list[Session] = []
    seen: dict[str, int] = {}
    for sid, d, turns in zip(sids, dates, sess_raw):
        # LongMemEval-S samples some filler sessions twice (identical content, different dates); keep
        # both in the stream but give the repeat a distinct id so turn ids stay unique.
        n = seen.get(sid, 0) + 1
        seen[sid] = n
        uid = sid if n == 1 else f"{sid}~{n}"
        at = _parse_lme_dt(d)
        ts = [Turn(id=f"{uid}#{i}", session=uid, speaker=t.get("role", "user"), text=t.get("content", "") or "", at=at,
                   has_answer=_has_answer(t)) for i, t in enumerate(turns)]
        sessions.append(Session(id=uid, at=at, turns=ts))
    # chronological order, as a memory system would ingest them (stable: ties keep file order)
    sessions.sort(key=lambda s: (s.at or datetime.min))
    ans_sessions = [str(x) for x in _lit(q["answer_session_ids"])]
    ans_set = set(ans_sessions)
    ev_turns = [t.id for s in sessions if s.id in ans_set for t in s.turns if t.has_answer]
    qid = str(q["question_id"])
    qa = [QA(id=qid, question=q["question"], answer=str(q["answer"]), evidence_turns=ev_turns,
             evidence_sessions=ans_sessions, category=str(q["question_type"]), category_id=LONGMEMEVAL_CATEGORY_ID,
             adversarial=qid.endswith("_abs"), at=_parse_lme_dt(q.get("question_date")))]
    return Conversation(id=qid, speakers=["user", "assistant"], sessions=sessions, observations=[], qa=qa, source="longmemeval")


def load_longmemeval(path: Path | str | None = None, limit: int | None = None, *, variant: str = LONGMEMEVAL_DEFAULT_VARIANT) -> list[Conversation]:
    """One Conversation per question (its haystack is the memory stream; one QA).

    variant: 's' (LongMemEval-S, the pre-registered dataset: ~48 sessions per question including
    distractors; default) or 'oracle' (answer sessions only, ~1.9 per question; a debugging aid).
    An explicit `path` overrides the variant's file. The S file is 277 MB: it is parsed once with
    json.load (about 18 s, ~2.4 GB peak) and only the first `limit` questions are built into
    Conversations, so the build step is cheap for small limits.
    """
    if variant not in LONGMEMEVAL_FILES:
        raise ValueError(f"variant must be one of {sorted(LONGMEMEVAL_FILES)}, got {variant!r}")
    p = Path(path) if path else DATA / "longmemeval" / LONGMEMEVAL_FILES[variant]
    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{p}: expected a JSON list of questions")
    items = raw[:limit] if limit else raw
    return [_lme_conversation(q) for q in items]
