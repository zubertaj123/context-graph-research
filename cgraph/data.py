"""Synthetic six-layer context graph generated from a hidden answer key.

Everything the labs measure (retrieval precision, as-of correctness, graduation
outcomes) is scored against `AnswerKey`, so no number is an opinion.
Scale 1 is ~3k nodes; pass scale=5 for ~15k.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .model import ContextGraph, T0, envelope
from .ontology import load_ontology

T_SUPERSEDE = datetime(2026, 6, 1, tzinfo=timezone.utc)   # when a canonical NEXT edge is replaced

PRODUCT_LINES = ["Latitude", "XPS", "OptiPlex"]
MODEL_NUMBERS = {"Latitude": [5440, 7440, 7450, 9450], "XPS": [13, 15, 16, 17], "OptiPlex": [3000, 5000, 7010, 7020]}

SYMPTOMS = {
    "no_post_led":            "no POST, amber LED blink pattern",
    "black_screen_after_sleep": "black screen after waking from sleep since the BIOS update",
    "battery_not_charging":   "battery not charging, plugged in",
    "wifi_drops":             "Wi-Fi drops every few minutes",
    "overheating":            "fan loud and system hot under light load",
}

# canonical actions: id -> (name, kind, governance tags)
ACTIONS = {
    "A_CHECK_LED":         ("Read the LED blink pattern", "diagnostic", []),
    "A_RESEAT_MEMORY":     ("Reseat the memory modules", "remedial", ["parts"]),
    "A_RUN_EPSA":          ("Run ePSA / SupportAssist diagnostics", "diagnostic", []),
    "A_SUSPEND_BITLOCKER": ("Suspend BitLocker", "precondition", ["data_loss"]),
    "A_UPDATE_BIOS":       ("Update the BIOS to the latest release", "remedial", ["firmware"]),
    "A_RESET_BIOS":        ("Reset BIOS to factory defaults", "remedial", ["settings_reset"]),
    "A_HARD_RESET":        ("Perform a hard reset (power drain)", "remedial", []),
    "A_UPDATE_GRAPHICS":   ("Update the graphics driver", "remedial", []),
    "A_CHECK_ADAPTER":     ("Verify the AC adapter wattage and LED", "diagnostic", []),
    "A_REPLACE_BATTERY":   ("Dispatch a replacement battery", "dispatch", ["dispatch", "parts"]),
    "A_UPDATE_WIFI":       ("Update the wireless driver", "remedial", []),
    "A_RESET_NETWORK":     ("Reset the network stack", "remedial", ["settings_reset"]),
    "A_CLEAN_VENTS":       ("Clean the vents and check fan speed", "remedial", []),
    "A_THERMAL_PASTE":     ("Onsite thermal module replacement", "dispatch", ["dispatch", "onsite"]),
}

# flow blueprint per symptom: ordered (action, decision) where decision may branch
FLOW_BLUEPRINT: dict[str, list[dict[str, Any]]] = {
    "no_post_led": [
        {"action": "A_CHECK_LED", "effort": "low", "ask": "led_pattern"},
        {"action": "A_RESEAT_MEMORY", "effort": "medium"},
        {"action": "A_RUN_EPSA", "effort": "low"},
        {"action": "A_RESET_BIOS", "effort": "medium"},
    ],
    "black_screen_after_sleep": [
        {"action": "A_RUN_EPSA", "effort": "low"},
        {"action": "A_SUSPEND_BITLOCKER", "effort": "medium", "decision": {"var": "bitlocker_enabled"}},
        {"action": "A_UPDATE_BIOS", "effort": "medium", "decision": {"var": "bios_version"}},
        {"action": "A_UPDATE_GRAPHICS", "effort": "low"},
        {"action": "A_HARD_RESET", "effort": "low"},
    ],
    "battery_not_charging": [
        {"action": "A_CHECK_ADAPTER", "effort": "low"},
        {"action": "A_RUN_EPSA", "effort": "low", "decision": {"var": "battery_health"}},
        {"action": "A_RESET_BIOS", "effort": "medium"},
        {"action": "A_REPLACE_BATTERY", "effort": "high", "decision": {"var": "warranty_tier"}},
    ],
    "wifi_drops": [
        {"action": "A_UPDATE_WIFI", "effort": "low"},
        {"action": "A_RESET_NETWORK", "effort": "medium"},
        {"action": "A_RUN_EPSA", "effort": "low"},
    ],
    "overheating": [
        {"action": "A_CLEAN_VENTS", "effort": "medium"},
        {"action": "A_UPDATE_BIOS", "effort": "medium", "decision": {"var": "bios_version"}},
        {"action": "A_RUN_EPSA", "effort": "low"},
        {"action": "A_THERMAL_PASTE", "effort": "high", "decision": {"var": "warranty_tier"}},
    ],
}

# decision variables: name -> (resolution, preferred columns in trust order, business term)
VARIABLES = {
    "bios_version":      ("system", ["supportassist.readings.bios_version", "asset_db.assets.bios_version_recorded"], "The firmware release currently running on the device"),
    "bitlocker_enabled": ("system", ["supportassist.readings.bitlocker_enabled"], "Whether drive encryption is on; a BIOS update without suspending it can lock the drive"),
    "battery_health":    ("system", ["supportassist.readings.battery_health"], "Battery wear level reported by telemetry, percent of design capacity"),
    "warranty_tier":     ("system", ["entitlement.contracts.tier"], "Support contract level, decides whether parts and onsite are covered"),
    "country":           ("system", ["asset_db.assets.country"], "Where the device is registered; scopes which agents may read it"),
    "led_pattern":       ("ask_customer", [], "The amber/white blink pattern on the power LED"),
    "noise_present":     ("ask_customer", [], "Whether the customer hears clicking or grinding"),
}

CATALOG = {
    "asset_db":      {"platform": "postgres", "schemas": {"assets": {"assets": ["service_tag", "model", "bios_version_recorded", "country", "ship_date"]}}},
    "supportassist": {"platform": "postgres", "schemas": {"readings": {"readings": ["service_tag", "bios_version", "bitlocker_enabled", "battery_health", "cpu_temp", "observed_at"]}}},
    "entitlement":   {"platform": "postgres", "schemas": {"contracts": {"contracts": ["service_tag", "tier", "expires", "onsite"]}}},
    "crm":           {"platform": "salesforce", "schemas": {"cases": {"cases": ["case_id", "service_tag", "subject", "contact_email"]}}},
}
COLUMN_DESCRIPTIONS = {
    "bios_version_recorded": "BIOS version recorded at ship time; stale unless SupportAssist reports the current version.",
    "bios_version": "BIOS version reported by the latest SupportAssist telemetry (certified, live).",
    "bitlocker_enabled": "Drive encryption state from telemetry.",
    "battery_health": "Battery capacity as percent of design capacity.",
    "tier": "Contract tier: basic, prosupport, prosupport_plus.",
    "country": "ISO country of registration.",
    "contact_email": "Customer contact; PII.",
    "subject": "Case subject line as typed by the customer; restricted.",
}
COLUMN_TAGS = {"contact_email": ["pii"], "subject": ["restricted"], "country": ["pii"]}

DISTRACTORS = [
    "Specifications: {model} ships with up to 64 GB of memory and two Thunderbolt ports.",
    "Warranty terms for {model}: standard limited hardware warranty for one year from ship date.",
    "Safety: before servicing {model}, disconnect power and remove the battery where applicable.",
    "The {model} keyboard backlight can be toggled with Fn+F5.",
    "Regulatory notice for {model}: this device complies with Part 15 of the FCC rules.",
    "To change the display language on {model}, open Settings and choose Time and language.",
]


@dataclass
class AnswerKey:
    flow_for: dict[tuple[str, str], str] = field(default_factory=dict)        # (product_line, symptom) -> flow id
    step_evidence: dict[str, list[str]] = field(default_factory=dict)         # step id -> true chunk ids (latest, en-us)
    resolving_action: dict[tuple[str, str], str] = field(default_factory=dict)  # (model id, symptom) -> action id
    superseded_edges: list[tuple[str, str, str]] = field(default_factory=list)  # edges replaced at T_SUPERSEDE
    replacement_edges: list[tuple[str, str, str]] = field(default_factory=list)
    systems_of_record: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)  # table -> service_tag -> row


def _sid(s: str) -> str:
    return s.lower().replace(" ", "_").replace("/", "_")


def build(scale: int = 1, seed: int = 7) -> tuple[ContextGraph, AnswerKey]:
    rnd = random.Random(seed)
    g = ContextGraph()
    key = AnswerKey()
    onto = load_ontology()

    # ---------------- taxonomy ----------------
    for tag, tier in onto["governance_tags"].items():
        g.add_node("GovernanceTag", f"tag:{tag}", "taxonomy", key=tag, tier=tier,
                   **envelope("taxonomy", f"governance.yaml#{tag}", content=tag))
    models: dict[str, list[str]] = {}
    for pl in PRODUCT_LINES:
        g.add_node("ProductLine", f"line:{_sid(pl)}", "taxonomy", name=pl, **envelope("taxonomy", f"taxonomy.csv#{pl}", content=pl))
        models[pl] = []
        for n in MODEL_NUMBERS[pl][: max(1, min(4, 2 + scale))]:
            mid = f"model:{_sid(pl)}_{n}"
            g.add_node("Model", mid, "taxonomy", name=f"{pl} {n}", **envelope("taxonomy", f"taxonomy.csv#{pl}-{n}", content=mid))
            g.add_edge(mid, "IN_LINE", f"line:{_sid(pl)}")
            models[pl].append(mid)
    for var, (_res, _cols, definition) in VARIABLES.items():
        g.add_node("BusinessTerm", f"term:var:{var}", "taxonomy", term=var, definition=definition,
                   **envelope("taxonomy", f"glossary.csv#{var}", content=var))

    # ---------------- catalog (where values live; never the value) ----------------
    col_ids: dict[str, str] = {}
    for db, spec in CATALOG.items():
        g.add_node("Database", f"db:{db}", "catalog", name=db, platform=spec["platform"], **envelope("catalog", f"pg_catalog#{db}", content=db))
        for sch, tables in spec["schemas"].items():
            g.add_node("Schema", f"schema:{db}.{sch}", "catalog", name=sch, **envelope("catalog", f"pg_namespace#{db}.{sch}", content=sch))
            g.add_edge(f"db:{db}", "HAS_SCHEMA", f"schema:{db}.{sch}")
            for tbl, cols in tables.items():
                tid = f"table:{db}.{sch}.{tbl}"
                g.add_node("Table", tid, "catalog", name=tbl, **envelope("catalog", f"pg_class#{db}.{sch}.{tbl}", content=tid))
                g.add_edge(f"schema:{db}.{sch}", "HAS_TABLE", tid)
                for c in cols:
                    cid = f"column:{db}.{sch}.{tbl}.{c}"
                    col_ids[f"{db}.{sch}.{c}"] = cid
                    ctype = "text" if c not in ("battery_health", "cpu_temp") else "numeric"
                    g.add_node("Column", cid, "catalog", name=c, type=ctype,
                               description=COLUMN_DESCRIPTIONS.get(c, f"{c} (undocumented; machine-drafted description pending steward sign-off)"),
                               machine_drafted=c not in COLUMN_DESCRIPTIONS,
                               **envelope("catalog", f"pg_attribute#{cid}", content=cid))
                    g.add_edge(tid, "HAS_COLUMN", cid)
                    for t in COLUMN_TAGS.get(c, []):
                        g.add_edge(cid, "TAGGED_WITH", f"tag:{t}")
    # declared foreign keys: every service_tag joins to the asset spine
    for ref in ("supportassist.readings.service_tag", "entitlement.contracts.service_tag", "crm.cases.service_tag"):
        g.add_edge(col_ids[ref], "REFERENCES", col_ids["asset_db.assets.service_tag"], criteria="service_tag = service_tag")

    # ---------------- process (canonical instructions) ----------------
    for aid, (name, kind, tags) in ACTIONS.items():
        g.add_node("Action", aid, "process", name=name, kind=kind, **envelope("boomi", f"boomi:action/{aid}@1", content=aid))
        for t in tags:
            g.add_edge(aid, "TAGGED_WITH", f"tag:{t}")
    for var, (res, cols, _definition) in VARIABLES.items():
        vid = f"var:{var}"
        g.add_node("Variable", vid, "process", name=var, resolution=res, **envelope("boomi", f"boomi:var/{var}", content=var))
        g.add_edge(vid, "DEFINED_BY", f"term:var:{var}")
        for rank, colref in enumerate(cols):
            db, sch, col = colref.split(".")
            g.add_edge(vid, "RESOLVED_BY", col_ids[f"{db}.{sch}.{col}"], rank=rank,
                       **envelope("catalog", "catalog.yaml#" + var, content=colref))
    for sym, text in SYMPTOMS.items():
        g.add_node("Symptom", f"symptom:{sym}", "evidence", name=text, **envelope("caselog", f"symptom/{sym}", authority="evidence", content=sym))

    step_action: dict[str, str] = {}
    for pl in PRODUCT_LINES:
        for sym, blueprint in FLOW_BLUEPRINT.items():
            fid = f"flow:{_sid(pl)}:{sym}"
            key.flow_for[(pl, sym)] = fid
            g.add_node("Flow", fid, "process", title=f"{pl}: {SYMPTOMS[sym]}", kind="troubleshooting", product_line=pl,
                       human_summary=f"Canonical path for '{SYMPTOMS[sym]}' on {pl}.",
                       **envelope("boomi", f"boomi:flow/{fid}@3", risk_tier="T1", content=fid))
            g.add_edge(fid, "HAS_SYMPTOM", f"symptom:{sym}")
            prev: str | None = None
            for i, st in enumerate(blueprint):
                sid = f"step:{_sid(pl)}:{sym}:{i+1}"
                aname, _kind, tags = ACTIONS[st["action"]]
                tier = max((onto["governance_tags"][t] for t in tags), default="T0")
                g.add_node("Step", sid, "process", title=f"{i+1}. {aname}",
                           human_summary=f"{aname} on {pl} for {SYMPTOMS[sym]}.",
                           rationale=f"Step {i+1} of the {pl} path: {aname.lower()} rules out the most likely cause before costlier steps.",
                           customer_effort=st["effort"], order=i + 1,
                           **envelope("boomi", f"boomi:step/{sid}@3", risk_tier=tier, content=sid))
                step_action[sid] = st["action"]
                g.add_edge(fid, "HAS_STEP", sid, **envelope("boomi", f"boomi:flow/{fid}@3", content=(fid, sid)))
                g.add_edge(sid, "PERFORMS", st["action"])
                if prev is None:
                    g.add_edge(fid, "STARTS_AT", sid, **envelope("boomi", f"boomi:flow/{fid}@3", content=(fid, "start")))
                else:
                    g.add_edge(prev, "NEXT", sid, **envelope("boomi", f"boomi:flow/{fid}@3", content=(prev, sid)))
                if "decision" in st:
                    var = st["decision"]["var"]
                    dtid, rid = f"dt:{sid}", f"rule:{sid}"
                    g.add_node("DecisionTable", dtid, "process", hit_policy="FIRST", **envelope("boomi", f"boomi:dt/{dtid}", content=dtid))
                    g.add_node("Rule", rid, "process", expr=f'{{"==": [{{"var": "{var}"}}, "<needs>"]}}', priority=1, **envelope("boomi", f"boomi:rule/{rid}", content=rid))
                    g.add_edge(sid, "EVALUATES", dtid)
                    g.add_edge(dtid, "HAS_RULE", rid)
                    g.add_edge(rid, "USES", f"var:{var}")
                    g.add_edge(rid, "ROUTES_TO", sid)
                if "ask" in st:
                    g.add_node("DecisionTable", f"dt:{sid}", "process", hit_policy="FIRST", **envelope("boomi", f"boomi:dt/{sid}", content=sid))
                    g.add_node("Rule", f"rule:{sid}", "process", expr='{"var": "%s"}' % st["ask"], priority=1, **envelope("boomi", f"boomi:rule/{sid}", content=sid))
                    g.add_edge(sid, "EVALUATES", f"dt:{sid}")
                    g.add_edge(f"dt:{sid}", "HAS_RULE", f"rule:{sid}")
                    g.add_edge(f"rule:{sid}", "USES", f"var:{st['ask']}")
                prev = sid
            # hidden truth: which action actually resolves each model's symptom (varies by model)
            for mid in models[pl]:
                key.resolving_action[(mid, sym)] = rnd.choice([b["action"] for b in blueprint[1:]])

    # a superseded canonical edge: the Latitude black-screen path once went EPSA -> BIOS update
    # directly; a ChangeSet inserted 'Suspend BitLocker' first. Old edge keeps valid_to, new edge valid_from.
    old_src, old_dst = "step:latitude:black_screen_after_sleep:1", "step:latitude:black_screen_after_sleep:3"
    g.add_edge(old_src, "NEXT", old_dst, **envelope("boomi", "boomi:flow/flow:latitude:black_screen_after_sleep@2",
                                                      status="deprecated", valid_from=T0 - timedelta(days=200), valid_to=T_SUPERSEDE, content="old"))
    key.superseded_edges.append((old_src, "NEXT", old_dst))
    # the generated chain edge 1->2 is the replacement: stamp its validity to start at T_SUPERSEDE
    for e in g.out(old_src, "NEXT"):
        if e.dst == "step:latitude:black_screen_after_sleep:2":
            e.props["valid_from"] = T_SUPERSEDE
            e.props["changeset"] = "cs:branch:0001"
            key.replacement_edges.append(e.key)

    # ---------------- evidence (documents, chunks, entities, cases) ----------------
    entity_ids: dict[tuple[str, str], str] = {}

    def entity(name: str, etype: str, canonical: str | None, conf: float) -> str:
        k = (name, etype)
        if k in entity_ids:
            return entity_ids[k]
        eid = f"entity:{_sid(etype)}:{_sid(name)}"
        upper = "OBJECT" if etype in ("Model", "Part") else "EVENT"
        g.add_node("Entity", eid, "evidence", name=name, type=etype, upper_type=upper,
                   **envelope("doc", "lane_b.extract", authority="evidence", confidence=conf, content=eid))
        if canonical and conf >= 0.8:
            g.add_edge(eid, "SAME_AS", canonical, confidence=conf, method="ontology_dictionary")
        entity_ids[k] = eid
        return eid

    chunk_counter = 0
    for pl in PRODUCT_LINES:
        for mid in models[pl]:
            mname = g.nodes[mid]["name"]
            for doc_type, revisions, locales in (("owners-manual", [1, 2], ["en-us", "de-de"]), ("service-manual", [1], ["en-us"]), ("kb", [1], ["en-us"])):
                for locale in locales:
                    for rev in revisions:
                        did = f"doc:{_sid(mname)}:{doc_type}:r{rev}:{locale}"
                        is_latest = rev == max(revisions)
                        g.add_node("Document", did, "evidence", title=f"{mname} {doc_type} rev{rev} ({locale})", doc_type=doc_type,
                                   model=mid, revision=rev, locale=locale, is_latest=is_latest,
                                   **envelope("doc", f"{did}.pdf", authority="evidence", version=rev, content=did))
                        g.add_edge(did, "ABOUT_PRODUCT", mid)
                        page, prev_chunk = 0, None
                        for sym, blueprint in FLOW_BLUEPRINT.items():
                            if doc_type == "kb" and sym not in ("black_screen_after_sleep", "no_post_led"):
                                continue
                            for i, st in enumerate(blueprint):
                                page += 1
                                sid = f"step:{_sid(pl)}:{sym}:{i+1}"
                                aname = ACTIONS[st["action"]][0]
                                prefix = "DE: " if locale == "de-de" else ""
                                body = f"{prefix}{mname}: when you see '{SYMPTOMS[sym]}', {aname.lower()}. " \
                                       f"{g.nodes[sid]['rationale']} (rev {rev})"
                                if doc_type == "kb":
                                    body = f"{prefix}KB article for {mname}: known issue '{SYMPTOMS[sym]}'. Fix: {aname.lower()}. See release notes {rev}."
                                chunk_counter += 1
                                cid = f"chunk:{did}#p{page}"
                                g.add_node("Chunk", cid, "evidence", page=page, text=body, is_latest=is_latest, locale=locale,
                                           doc_type=doc_type, model=mid, **envelope("doc", f"{did}.pdf#p{page}", authority="evidence", content=cid))
                                g.add_edge(did, "HAS_CHUNK", cid)
                                if prev_chunk:
                                    g.add_edge(prev_chunk, "NEXT_CHUNK", cid)
                                prev_chunk = cid
                                g.add_edge(cid, "MENTIONS", entity(mname, "Model", mid, 0.97), confidence=0.97)
                                g.add_edge(cid, "MENTIONS", entity(aname, "Action", st["action"], 0.95), confidence=0.95)
                                if sym == "no_post_led":
                                    g.add_edge(cid, "MENTIONS", entity("led:2,3", "ErrorCode", None, 0.9), confidence=0.9)
                                if is_latest and locale == "en-us":
                                    key.step_evidence.setdefault(sid, []).append(cid)
                        for d in rnd.sample(DISTRACTORS, 3):
                            page += 1
                            cid = f"chunk:{did}#p{page}"
                            g.add_node("Chunk", cid, "evidence", page=page, text=d.format(model=mname), is_latest=is_latest, locale=locale,
                                       doc_type=doc_type, model=mid, **envelope("doc", f"{did}.pdf#p{page}", authority="evidence", content=cid))
                            g.add_edge(did, "HAS_CHUNK", cid)
                            g.add_edge(cid, "MENTIONS", entity(mname, "Model", mid, 0.97), confidence=0.97)
    # anchor: EVIDENCED_BY from each step to its top-3 true chunks (KB weighted 2x)
    for sid, chunks in key.step_evidence.items():
        scored = sorted(chunks, key=lambda c: (2.0 if g.nodes[c]["doc_type"] == "kb" else 1.0) * 0.95, reverse=True)[:3]
        for c in scored:
            g.add_edge(sid, "EVIDENCED_BY", c, score=(2.0 if g.nodes[c]["doc_type"] == "kb" else 1.0) * 0.95, via="action")

    # cases: observed outcomes per (model, symptom)
    case_n = 0
    for pl in PRODUCT_LINES:
        for mid in models[pl]:
            for sym in SYMPTOMS:
                truth = key.resolving_action[(mid, sym)]
                for _ in range(6 * scale):
                    case_n += 1
                    resolved = rnd.random() < 0.8
                    outcome = "resolved" if resolved else rnd.choice(["dispatched", "reopened"])
                    g.add_node("Case", f"case:{case_n:06d}", "evidence", symptom=sym, outcome=outcome,
                               action_taken=truth if resolved else rnd.choice(list(ACTIONS)),
                               **envelope("caselog", f"crm:case/{case_n}", authority="evidence", content=case_n))
                    g.add_edge(f"case:{case_n:06d}", "CASE_MODEL", mid)
                    g.add_edge(f"case:{case_n:06d}", "CASE_SYMPTOM", f"symptom:{sym}")

    # ---------------- memory (what agents did and learned) ----------------
    g.add_node("PromptVersion", "pv:supervisor:3", "memory", surface="supervisor", **envelope("agent", "rsi/pv3", version=3, content="pv3"))
    for u in range(3):
        uid = f"user:engineer:{u}"
        g.add_node("User", uid, "memory", channel="engineer", **envelope("nba", f"nba:user/{u}", content=uid))
    trace_n = 0
    for pl in PRODUCT_LINES:
        for sym in SYMPTOMS:
            fid = key.flow_for[(pl, sym)]
            steps = [e.dst for e in g.out(fid, "HAS_STEP")]
            for _ in range(2 * scale):
                trace_n += 1
                uid = f"user:engineer:{rnd.randrange(3)}"
                conv, tr = f"conv:{trace_n:05d}", f"trace:{trace_n:05d}"
                g.add_node("Conversation", conv, "memory", session_id=f"s{trace_n}", **envelope("nba", f"nba:session/{trace_n}", authority="evidence", content=conv))
                g.add_edge(uid, "HAS_CONVERSATION", conv)
                mid = rnd.choice(models[pl])
                g.add_node("Message", f"{conv}:m1", "memory", role="user", content=f"{g.nodes[mid]['name']}: {SYMPTOMS[sym]}", **envelope("nba", f"{conv}#1", authority="evidence", content=1))
                g.add_edge(conv, "HAS_MESSAGE", f"{conv}:m1")
                success = rnd.random() < 0.85
                g.add_node("ReasoningTrace", tr, "memory", task=f"{SYMPTOMS[sym]} on {g.nodes[mid]['name']}", outcome="resolved" if success else "reopened",
                           success=success, **envelope("agent", f"copilot:trace/{trace_n}", authority="evidence", content=tr))
                g.add_edge(conv, "HAS_TRACE", tr)
                g.add_edge(tr, "ABOUT", fid)
                g.add_edge(tr, "RAN_UNDER", "pv:supervisor:3")
                for order, sid in enumerate(steps[: rnd.randint(2, len(steps))], start=1):
                    rs = f"{tr}:s{order}"
                    g.add_node("ReasoningStep", rs, "memory", action="next_step" if order == 1 else "search_docs", order=order,
                               **envelope("agent", f"{tr}#{order}", authority="evidence", content=rs))
                    g.add_edge(tr, "HAS_REASONING_STEP", rs)
                    tc = f"{rs}:tc"
                    g.add_node("ToolCall", tc, "memory", tool="next_step" if order == 1 else "search_docs", call_status="ok", duration_ms=rnd.randint(5, 400),
                               **envelope("agent", f"{tr}#{order}/tool", authority="evidence", content=tc))
                    g.add_edge(rs, "USES_TOOL", tc)
                    g.add_edge(rs, "TOUCHED", sid)
                    for c in g.targets(sid, "EVIDENCED_BY")[:2]:
                        g.add_edge(rs, "TOUCHED", c.id)
                    if order == 1:
                        g.add_edge(rs, "TOUCHED", col_ids["supportassist.readings.bios_version"])
            # a learned rule for this flow (AI-native memory: legible=false, authority evidence)
            mem = f"memory:{_sid(pl)}:{sym}"
            best = rnd.choice([b["action"] for b in FLOW_BLUEPRINT[sym][1:]])
            g.add_node("Memory", mem, "memory", signature=f"{sym} on {pl} -> {best}", success_rate=round(rnd.uniform(0.7, 0.95), 2),
                       support=rnd.randint(8, 60), legible=False, **envelope("agent", f"nba:learned/{mem}", status="candidate", authority="evidence", attested_by=None, content=mem))
            g.add_edge(mem, "ABOUT", fid)
            g.add_edge(mem, "SUGGESTS", best)
            for t in [f"trace:{n:05d}" for n in range(max(1, trace_n - 2 * scale + 1), trace_n + 1)]:
                g.add_edge(mem, "LEARNED_FROM", t)

    # ---------------- lifecycle ----------------
    g.add_node("ChangeSet", "cs:branch:0001", "lifecycle", kind="branch", graduation="promoted", tier="T3", proposer="agent:copilot", approver="steward:alice",
               **envelope("agent", "intake/cs:branch:0001", status="approved", risk_tier="T3", content="cs1"))
    g.add_edge("cs:branch:0001", "PROPOSES", "step:latitude:black_screen_after_sleep:2")
    g.add_edge("cs:branch:0001", "DERIVED_FROM", "memory:latitude:black_screen_after_sleep")
    g.add_node("ChangeSet", "cs:evidence:0002", "lifecycle", kind="evidence", graduation="candidate", tier="T0", proposer="agent:copilot",
               **envelope("agent", "intake/cs:evidence:0002", status="candidate", authority="candidate", attested_by=None, content="cs2"))
    g.add_edge("cs:evidence:0002", "PROPOSES", "step:xps:wifi_drops:2")
    g.add_edge("cs:evidence:0002", "DERIVED_FROM", "trace:00001")
    g.add_node("ChangeSet", "cs:branch:0003", "lifecycle", kind="branch", graduation="rejected", tier="T3", proposer="agent:copilot", approver="steward:bob",
               **envelope("agent", "intake/cs:branch:0003", status="rejected", authority="candidate", attested_by=None, content="cs3"))
    g.add_edge("cs:branch:0003", "PROPOSES", "step:optiplex:overheating:2")
    g.add_edge("cs:branch:0003", "CONFLICTS_WITH", "step:optiplex:overheating:4")
    g.meta["graph_version"] = "ontol0.1+cs1@2026-06-01"
    g.meta["promoted_changesets"] = 1

    # ---------------- simulated systems of record (live state stays OUT of the graph) ----------------
    tags = [f"SIM{n:07d}" for n in range(1, 6 * scale + 1)]
    key.systems_of_record = {"asset_db.assets": {}, "supportassist.readings": {}, "entitlement.contracts": {}, "crm.cases": {}}
    all_models = [m for ms in models.values() for m in ms]
    for i, st in enumerate(tags):
        mid = all_models[i % len(all_models)]
        key.systems_of_record["asset_db.assets"][st] = {"service_tag": st, "model": mid, "bios_version_recorded": "1.12.0", "country": rnd.choice(["US", "DE", "IN"]), "ship_date": "2025-03-01"}
        key.systems_of_record["supportassist.readings"][st] = {"service_tag": st, "bios_version": rnd.choice(["1.18.0", "1.19.2"]), "bitlocker_enabled": rnd.random() < 0.6,
                                                               "battery_health": rnd.randint(40, 100), "cpu_temp": rnd.randint(45, 95), "observed_at": "2026-10-01T09:00:00Z"}
        key.systems_of_record["entitlement.contracts"][st] = {"service_tag": st, "tier": rnd.choice(["basic", "prosupport", "prosupport_plus"]), "expires": "2027-01-01", "onsite": rnd.random() < 0.5}
        key.systems_of_record["crm.cases"][st] = {"case_id": f"C{i}", "service_tag": st, "subject": "restricted text", "contact_email": "pii@example.com"}
    return g, key
