from __future__ import annotations

"""Cross-record invariants for the planning SSOT, including restored snapshots."""

from collections.abc import Mapping
from typing import Any


def _records(state: Mapping[str, Any], key: str, id_key: str) -> dict[str, Mapping[str, Any]]:
    rows = state.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"PROMPT_STATE_SHAPE: {key} must contain objects")
    result = {str(row.get(id_key) or ""): row for row in rows}
    if "" in result or len(result) != len(rows):
        raise ValueError(f"PROMPT_STATE_IDS: {key} IDs must be nonempty and unique")
    return result


def validate_state_links(state: Mapping[str, Any]) -> None:
    from .planning_state_contract import ROUTE_SOURCES, _sha, _validated_route

    original = state["original_prompt"]
    known = _records(state, "known", "known_id")
    for row in [state.get("goal"), *known.values(), *state.get("references", [])]:
        source = row.get("source") if isinstance(row, Mapping) else None
        if not isinstance(source, Mapping):
            raise ValueError("PROMPT_STATE_SOURCE: authored record requires a source span")
        start, end = source.get("char_start"), source.get("char_end")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(original)
                or source.get("text") != original[start:end]
                or source.get("text_sha256") != _sha(original[start:end])):
            raise ValueError("PROMPT_STATE_SOURCE: authored span is stale")
    unknowns = _records(state, "unresolved", "unresolved_id")
    research = _records(state, "research_queue", "research_id")
    evidence = _records(state, "evidence", "evidence_id")
    decisions = _records(state, "decisions", "decision_id")
    for uid, row in unknowns.items():
        _validated_route(str(row.get("reason")), str(row.get("resolution_route")))
        if row.get("status") not in {"open", "resolved", "blocked"}:
            raise ValueError("PROMPT_STATE_STATUS: invalid unresolved status")
        if not str(row.get("information_needed") or "").strip():
            raise ValueError("PROMPT_STATE_INFORMATION: unknown needs information_needed")
        route = row.get("resolution_route")
        if set(row.get("source_kinds", [])) != set(ROUTE_SOURCES[route]):
            raise ValueError("PROMPT_STATE_ROUTE: sources violate the host-owned route")
        ref = row.get("research_ref")
        if route in {"user_only", "default_policy"}:
            if ref:
                raise ValueError("PROMPT_STATE_ROUTE: policy/user decisions cannot enter research")
        elif ref not in research or uid not in research[ref].get("resolves", []):
            raise ValueError("PROMPT_STATE_RESEARCH: unknown has no matching research item")
    for row in research.values():
        if not str(row.get("information_needed") or "").strip() or not str(row.get("objective") or "").strip():
            raise ValueError("PROMPT_STATE_INFORMATION: information_needed must precede queries")
        if row.get("status") not in {"pending", "complete", "blocked"}:
            raise ValueError("PROMPT_STATE_STATUS: invalid research status")
        if not isinstance(row.get("queries"), list):
            raise ValueError("PROMPT_STATE_QUERY: queries must be an array")
        for uid in row.get("resolves", []):
            unknown = unknowns[uid]
            if set(row.get("source_kinds", [])) != set(unknown.get("source_kinds", [])):
                raise ValueError("PROMPT_STATE_ROUTE: research sources disagree with unknown")
    refs = {ref for row in evidence.values() if row.get("sufficient") is True for ref in row.get("evidence_refs", [])}
    for row in evidence.values():
        if row.get("sufficient") is True and (not row.get("claims") or not row.get("evidence_refs")):
            raise ValueError("PROMPT_STATE_EVIDENCE: sufficient evidence requires cited claims")
    resolved = state.get("resolved")
    if not isinstance(resolved, list):
        raise ValueError("PROMPT_STATE_RESOLVED: resolution records are required")
    resolved_ids = set()
    for row in resolved:
        uid = row.get("unresolved_id") if isinstance(row, Mapping) else None
        if uid not in unknowns or uid in resolved_ids or unknowns[uid].get("status") != "resolved":
            raise ValueError("PROMPT_STATE_RESOLVED: invalid resolution link")
        resolved_ids.add(uid)
        if row.get("basis") == "host_default_policy":
            if unknowns[uid].get("reason") != "scope":
                raise ValueError("PROMPT_STATE_RESOLVED: default policy resolves scope only")
        elif row.get("basis") == "grounded_research":
            own_refs = {ref for e in evidence.values() if e.get("research_ref") == unknowns[uid].get("research_ref") and e.get("sufficient") is True for ref in e.get("evidence_refs", [])}
            if not row.get("evidence_refs") or not set(row["evidence_refs"]).issubset(own_refs):
                raise ValueError("PROMPT_STATE_RESOLVED: resolution lacks its own research evidence")
        else:
            raise ValueError("PROMPT_STATE_RESOLVED: unsupported resolution basis")
    if any(row.get("status") == "resolved" and uid not in resolved_ids for uid, row in unknowns.items()):
        raise ValueError("PROMPT_STATE_RESOLVED: resolved status requires a resolution record")
    requirements = {}
    details = {}
    for did, row in decisions.items():
        kind = row.get("decision_type")
        if kind == "requirement":
            quote = row.get("prompt_quote")
            cited = row.get("evidence_refs", [])
            if (not quote and not cited) or (quote and quote not in original) or not set(cited).issubset(refs):
                raise ValueError("PROMPT_STATE_DECISION: requirement lacks real provenance")
            rid = row.get("requirement_id")
            if not rid or rid in requirements:
                raise ValueError("PROMPT_STATE_DECISION: duplicate or missing requirement ID")
            requirements[rid] = row
        elif kind == "detailed_implementation_plan":
            rid = row.get("requirement_ref")
            if rid in details:
                raise ValueError("PROMPT_STATE_DECISION: duplicate implementation detail")
            details[rid] = row
            from .planning_detail_template import validate_worksheet
            validate_worksheet(row.get("engineering_worksheet"), refs)
            for field in ("implementation_capabilities", "implementation_obligations"):
                if not row.get(field):
                    raise ValueError("PROMPT_STATE_DECISION: implementation detail is empty")
                for obligation in row[field]:
                    cited = obligation.get("evidence_refs", [])
                    if not cited or not set(cited).issubset(refs):
                        raise ValueError("PROMPT_STATE_DECISION: implementation lacks real evidence")
        else:
            raise ValueError("PROMPT_STATE_DECISION: unsupported decision type")
    if set(details) - set(requirements):
        raise ValueError("PROMPT_STATE_COVERAGE: implementation references unknown requirement")
    if state.get("plan_ready"):
        if any(row.get("status") != "resolved" for row in unknowns.values()):
            raise ValueError("PROMPT_STATE_READY: unresolved user intent or research blocks planning")
        if state.get("blockers"):
            raise ValueError("PROMPT_STATE_READY: active blockers remain")
        coverage = state.get("coverage", [])
        if len(coverage) != len(requirements) or not requirements or set(details) != set(requirements):
            raise ValueError("PROMPT_STATE_COVERAGE: every requirement needs implementation and coverage")
        covered = set()
        for row in coverage:
            rid = row.get("requirement_ref")
            if rid not in details or rid in covered or row.get("status") != "covered" or row.get("detailed_plan_ref") != details[rid]["decision_id"]:
                raise ValueError("PROMPT_STATE_COVERAGE: invalid coverage claim")
            covered.add(rid)
