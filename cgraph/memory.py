"""Agent memory and governance: record what an agent read, serve recall as evidence,
graduate agent-written knowledge into canonical instructions, audit everything.

The state machine is a review intake: ChangeSet -> validate -> tier -> review -> promote,
with the rules that never bend: proposer != approver; agent confidence is an input, never
the decider; nothing is deleted, promotion sets valid_to.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .model import ContextGraph, envelope
from .ontology import load_ontology
from .retrieval import tokens
from .temporal import supersede

KIND_FLOOR = {"evidence": "T0", "alias": "T1", "reword": "T1", "branch": "T2", "new_intent": "T2", "deprecate": "T3"}
TIER_ORDER = ["T0", "T1", "T2", "T3"]


class Ledger:
    """Append-only JSONL audit: every tool call, every state read, every graduation event."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **payload: Any) -> str:
        rec = {"kind": kind, **payload}
        line = json.dumps(rec, default=str, sort_keys=True)
        rec_id = hashlib.sha1(line.encode()).hexdigest()[:10]
        with open(self.path, "a") as f:
            f.write(json.dumps({"id": rec_id, **rec}, default=str, sort_keys=True) + "\n")
        return rec_id

    def read(self, kind: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
        return [r for r in rows if kind is None or r["kind"] == kind]


# ---------------------------------------------------------------------------
# recording a run
# ---------------------------------------------------------------------------
def record_trace(g: ContextGraph, ledger: Ledger, *, conversation_id: str, task: str, flow_id: str,
                 steps: list[dict[str, Any]], outcome: str, success: bool, prompt_version: str = "pv:supervisor:3",
                 at: datetime) -> str:
    """steps: [{action, tool, status, touched: [node ids], output_sha}]  (what the copilot's record() writes)."""
    n = sum(1 for _ in g.by_label("ReasoningTrace")) + 1
    tr = f"trace:{n:05d}"
    g.add_node("ReasoningTrace", tr, "memory", task=task, outcome=outcome, success=success,
               **envelope("agent", f"copilot:trace/{n}", authority="evidence", valid_from=at, content=tr))
    g.add_edge(conversation_id, "HAS_TRACE", tr)
    g.add_edge(tr, "ABOUT", flow_id)
    g.add_edge(tr, "RAN_UNDER", prompt_version)
    for order, s in enumerate(steps, start=1):
        rs, tc = f"{tr}:s{order}", f"{tr}:s{order}:tc"
        g.add_node("ReasoningStep", rs, "memory", action=s["action"], order=order,
                   **envelope("agent", f"{tr}#{order}", authority="evidence", valid_from=at, content=rs))
        g.add_edge(tr, "HAS_REASONING_STEP", rs)
        g.add_node("ToolCall", tc, "memory", tool=s["tool"], call_status=s.get("status", "ok"), output_sha=s.get("output_sha"),
                   **envelope("agent", f"{tr}#{order}/tool", authority="evidence", valid_from=at, content=tc))
        g.add_edge(rs, "USES_TOOL", tc)
        for nid in s.get("touched", [])[:25]:          # cap: <= 25 TOUCHED per step
            if nid in g.nodes:                          # TOUCHED is MATCH-only: never creates domain nodes
                g.add_edge(rs, "TOUCHED", nid)
        ledger.write("tool_call", trace=tr, step=order, tool=s["tool"], output_sha=s.get("output_sha"),
                     graph_version=g.meta["graph_version"], as_of=at)
    return tr


def recall_similar(g: ContextGraph, task: str, k: int = 3) -> list[dict[str, Any]]:
    """recall_similar_sessions: served as 'reference only', never as the canonical path."""
    qt = set(tokens(task))
    scored = []
    for t in g.by_label("ReasoningTrace"):
        overlap = len(qt & set(tokens(t["task"])))
        if overlap:
            scored.append((overlap, t))
    scored.sort(key=lambda x: -x[0])
    return [{"trace_id": t.id, "task": t["task"], "outcome": t["outcome"], "success": t["success"],
             "label": "reference only"} for _, t in scored[:k]]


def learned_rules(g: ContextGraph, flow_id: str) -> list[dict[str, Any]]:
    """learned_rules: Memory ABOUT flow, SUGGESTS action, with its graduation status from any ChangeSet DERIVED_FROM it."""
    rows = []
    for m in g.sources(flow_id, "ABOUT"):
        if m.label != "Memory":
            continue
        actions = [a.id for a in g.targets(m.id, "SUGGESTS")]
        cs = [c for c in g.sources(m.id, "DERIVED_FROM") if c.label == "ChangeSet"]
        rows.append({"memory_id": m.id, "signature": m["signature"], "success_rate": m["success_rate"], "support": m["support"],
                     "suggests": actions, "graduation": cs[0]["graduation"] if cs else "not_proposed", "legible": m["legible"]})
    return rows


# ---------------------------------------------------------------------------
# graduation
# ---------------------------------------------------------------------------
@dataclass
class Proposal:
    kind: str
    proposer: str
    proposes_step: str
    derived_from: list[str]
    payload: dict[str, Any]              # e.g. {"insert_before": step_id, "new_step": {...}}
    changeset_id: str = ""
    checks: dict[str, Any] = field(default_factory=dict)


class GraduationError(Exception):
    pass


def propose(g: ContextGraph, ledger: Ledger, p: Proposal, at: datetime) -> str:
    n = sum(1 for _ in g.by_label("ChangeSet")) + 1
    cs = f"cs:{p.kind}:{n:04d}"
    p.changeset_id = cs
    g.add_node("ChangeSet", cs, "lifecycle", kind=p.kind, graduation="candidate", tier=KIND_FLOOR[p.kind], proposer=p.proposer,
               payload_json=json.dumps(p.payload, default=str),
               **envelope("agent", f"intake/{cs}", status="candidate", authority="candidate", attested_by=None, valid_from=at, content=p.payload))
    g.add_edge(cs, "PROPOSES", p.proposes_step)
    for d in p.derived_from:
        g.add_edge(cs, "DERIVED_FROM", d)
    ledger.write("graduation", changeset=cs, event="proposed", by=p.proposer, at=at)
    return cs


def validate(g: ContextGraph, ledger: Ledger, p: Proposal, replay: Callable[[dict[str, Any]], dict[str, float]] | None = None) -> dict[str, Any]:
    """Schema check (human-readable fields are mandatory even for agent-authored nodes) + held-out replay."""
    checks: dict[str, Any] = {}
    new_step = p.payload.get("new_step", {})
    missing = [f for f in ("title", "human_summary", "rationale", "customer_effort") if f not in new_step]
    checks["schema_ok"] = not missing
    checks["schema_missing"] = missing
    if replay is not None:
        r = replay(p.payload)
        checks["replay"] = r
        checks["replay_ok"] = r.get("lift", 0.0) >= 0.10 and r.get("regression", 0.0) <= 0.01
    else:
        checks["replay_ok"] = p.kind in ("evidence", "alias", "reword")  # atomic kinds need no replay
    p.checks = checks
    ledger.write("graduation", changeset=p.changeset_id, event="validated", checks=checks)
    return checks


def tier(g: ContextGraph, p: Proposal) -> str:
    """tier = max(kind floor, risk tier of every governance tag on the actions touched). Never chosen by the proposer."""
    onto = load_ontology()
    tags = onto["governance_tags"]
    tiers = [KIND_FLOOR[p.kind]]
    action = p.payload.get("new_step", {}).get("action")
    if action:
        for t in g.targets(action, "TAGGED_WITH"):
            tiers.append(tags[t["key"]])
    for st in [p.proposes_step]:
        for a in g.targets(st, "PERFORMS"):
            for t in g.targets(a.id, "TAGGED_WITH"):
                tiers.append(tags[t["key"]])
    best = max(tiers, key=TIER_ORDER.index)
    g.nodes[p.changeset_id].props["tier"] = best
    return best


def review(g: ContextGraph, ledger: Ledger, p: Proposal, approver: str | None, decision: str, at: datetime) -> str:
    cs = g.nodes[p.changeset_id]
    if not p.checks.get("schema_ok") or not p.checks.get("replay_ok"):
        decision = "rejected"
    t = cs["tier"]
    if t != "T0":
        if approver is None:
            raise GraduationError(f"tier {t} needs a named approver")
        if approver == p.proposer:
            raise GraduationError("proposer != approver: an agent or person may not approve its own change")
    cs.props["graduation"] = "approved" if decision == "approved" else "rejected"
    cs.props["approver"] = approver or "auto:T0"
    cs.props["status"] = "approved" if decision == "approved" else "rejected"
    cs.props["attested_by"] = approver or "auto:T0"
    ledger.write("graduation", changeset=p.changeset_id, event=cs["graduation"], by=cs["approver"], tier=t, at=at)
    return cs["graduation"]


def promote(g: ContextGraph, ledger: Ledger, p: Proposal, at: datetime) -> dict[str, Any]:
    """Apply an approved 'branch' change: insert a new step before `insert_before`. Old edges get valid_to."""
    cs = g.nodes[p.changeset_id]
    if cs["graduation"] != "approved":
        raise GraduationError("only approved change sets can be promoted")
    before = p.payload["insert_before"]
    ns = p.payload["new_step"]
    sid = ns["id"]
    g.add_node("Step", sid, "process", title=ns["title"], human_summary=ns["human_summary"], rationale=ns["rationale"],
               customer_effort=ns["customer_effort"], order=g.nodes[before]["order"],
               **envelope("agent", f"intake/{p.changeset_id}", status="approved", authority="canonical", risk_tier=cs["tier"],
                          valid_from=at, authored_by=p.proposer, attested_by=cs["approver"], content=sid))
    g.add_edge(sid, "PERFORMS", ns["action"])
    flow = g.sources(before, "HAS_STEP")[0]
    g.add_edge(flow.id, "HAS_STEP", sid, **envelope("agent", f"intake/{p.changeset_id}", valid_from=at, attested_by=cs["approver"], content=(flow.id, sid)))
    superseded = []
    for e in g.inc(before, "NEXT", live_at=at) + g.inc(before, "STARTS_AT", live_at=at):
        superseded.append(e.key)
        supersede(g, e, sid, at, p.changeset_id)
    new_next = g.add_edge(sid, "NEXT", before, **envelope("agent", f"intake/{p.changeset_id}", valid_from=at, attested_by=cs["approver"], content=(sid, before)))
    new_next.props["changeset"] = p.changeset_id
    cs.props["graduation"] = "promoted"
    g.meta["promoted_changesets"] += 1
    g.meta["graph_version"] = f"ontol0.1+cs{g.meta['promoted_changesets']}@{at.date().isoformat()}"
    ledger.write("graduation", changeset=p.changeset_id, event="promoted", superseded=superseded, graph_version=g.meta["graph_version"], at=at)
    return {"new_step": sid, "superseded": superseded, "graph_version": g.meta["graph_version"]}


def why(ledger: Ledger, trace_id: str) -> list[dict[str, Any]]:
    """Audit trail: every call behind an answer, in order."""
    return [r for r in ledger.read("tool_call") if r.get("trace") == trace_id]
