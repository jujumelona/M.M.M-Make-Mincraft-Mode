from __future__ import annotations

"""Cross-record structural invariants for the canonical planning-state SSOT.

These checks protect topology and provenance links. They deliberately do not re-interpret
prompt language or require literal quote equality.
"""

from collections.abc import Mapping
from typing import Any


def _records(
    state: Mapping[str, Any],
    key: str,
    id_key: str,
) -> dict[str, Mapping[str, Any]]:
    rows = state.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"PROMPT_STATE_SHAPE: {key} must contain objects")
    result = {str(row.get(id_key) or ""): row for row in rows}
    if "" in result or len(result) != len(rows):
        raise ValueError(f"PROMPT_STATE_IDS: {key} IDs must be nonempty and unique")
    return result


def _optional_records(
    state: Mapping[str, Any],
    key: str,
    id_key: str,
) -> dict[str, Mapping[str, Any]]:
    if key not in state:
        return {}
    return _records(state, key, id_key)


def validate_state_links(state: Mapping[str, Any]) -> None:
    """Reject broken task-state links while leaving semantic interpretation to its owner."""
    from .planning_state_contract import ROUTE_SOURCES, _validated_route

    known = _records(state, "known", "known_id")
    references = state.get("references")
    goal = state.get("goal")
    if not isinstance(goal, Mapping) or not str(goal.get("statement") or "").strip():
        raise ValueError("PROMPT_STATE_GOAL: goal must contain a statement")
    if not isinstance(references, list) or any(not isinstance(row, Mapping) for row in references):
        raise ValueError("PROMPT_STATE_SHAPE: references must contain objects")
    if any(not str(row.get("statement") or "").strip() for row in known.values()):
        raise ValueError("PROMPT_STATE_KNOWN: known statements must not be empty")

    unknowns = _records(state, "unresolved", "unresolved_id")
    research = _records(state, "research_queue", "research_id")
    evidence = _records(state, "evidence", "evidence_id")
    decisions = _records(state, "decisions", "decision_id")
    candidates = _optional_records(
        state,
        "implementation_candidates",
        "candidate_id",
    )

    for uid, row in unknowns.items():
        reason = str(row.get("reason") or "")
        route = _validated_route(reason, str(row.get("resolution_route") or ""))
        if row.get("status") not in {"open", "resolved", "blocked"}:
            raise ValueError("PROMPT_STATE_STATUS: invalid unresolved status")
        if not str(row.get("information_needed") or "").strip():
            raise ValueError("PROMPT_STATE_INFORMATION: unknown needs information_needed")
        if set(row.get("source_kinds", [])) != set(ROUTE_SOURCES[route]):
            raise ValueError("PROMPT_STATE_ROUTE: sources violate the host-owned route")
        research_ref = str(row.get("research_ref") or "")
        if route in {"user_only", "default_policy"}:
            if research_ref:
                raise ValueError(
                    "PROMPT_STATE_ROUTE: policy/user decisions cannot enter research"
                )
        elif research_ref not in research or uid not in research[research_ref].get("resolves", []):
            raise ValueError(
                "PROMPT_STATE_RESEARCH: unknown has no matching research item"
            )

    for row in research.values():
        if not str(row.get("information_needed") or "").strip() or not str(row.get("objective") or "").strip():
            raise ValueError(
                "PROMPT_STATE_INFORMATION: information_needed must precede queries"
            )
        if row.get("status") not in {"pending", "complete", "blocked"}:
            raise ValueError("PROMPT_STATE_STATUS: invalid research status")
        if not isinstance(row.get("queries"), list):
            raise ValueError("PROMPT_STATE_QUERY: queries must be an array")
        resolves = row.get("resolves")
        if not isinstance(resolves, list) or not resolves:
            raise ValueError("PROMPT_STATE_RESEARCH: research must resolve an unknown")
        for uid in resolves:
            if uid not in unknowns:
                raise ValueError("PROMPT_STATE_RESEARCH: research cites unknown unresolved ID")
            if set(row.get("source_kinds", [])) != set(unknowns[uid].get("source_kinds", [])):
                raise ValueError(
                    "PROMPT_STATE_ROUTE: research sources disagree with unknown"
                )

    sufficient_refs = {
        ref
        for row in evidence.values()
        if row.get("sufficient") is True
        for ref in row.get("evidence_refs", [])
    }
    for row in evidence.values():
        if str(row.get("research_ref") or "") not in research:
            raise ValueError("PROMPT_STATE_EVIDENCE: evidence has no research owner")
        if row.get("sufficient") is True and (
            not row.get("claims") or not row.get("evidence_refs")
        ):
            raise ValueError(
                "PROMPT_STATE_EVIDENCE: sufficient evidence requires cited claims"
            )

    resolved = state.get("resolved")
    if not isinstance(resolved, list):
        raise ValueError("PROMPT_STATE_RESOLVED: resolution records are required")
    resolved_ids: set[str] = set()
    for row in resolved:
        uid = row.get("unresolved_id") if isinstance(row, Mapping) else None
        if uid not in unknowns or uid in resolved_ids or unknowns[uid].get("status") != "resolved":
            raise ValueError("PROMPT_STATE_RESOLVED: invalid resolution link")
        resolved_ids.add(str(uid))
        basis = row.get("basis")
        if basis == "host_default_policy":
            if unknowns[uid].get("reason") != "scope":
                raise ValueError(
                    "PROMPT_STATE_RESOLVED: default policy resolves scope only"
                )
        elif basis == "grounded_research":
            own_refs = {
                ref
                for evidence_row in evidence.values()
                if evidence_row.get("research_ref") == unknowns[uid].get("research_ref")
                and evidence_row.get("sufficient") is True
                for ref in evidence_row.get("evidence_refs", [])
            }
            cited = row.get("evidence_refs")
            if not isinstance(cited, list) or not cited or not set(cited).issubset(own_refs):
                raise ValueError(
                    "PROMPT_STATE_RESOLVED: resolution lacks its own research evidence"
                )
        else:
            raise ValueError("PROMPT_STATE_RESOLVED: unsupported resolution basis")
    if any(
        row.get("status") == "resolved" and uid not in resolved_ids
        for uid, row in unknowns.items()
    ):
        raise ValueError(
            "PROMPT_STATE_RESOLVED: resolved status requires a resolution record"
        )

    prompt_refs_allowed = {"goal", *known.keys()}
    requirements: dict[str, Mapping[str, Any]] = {}
    details: dict[str, Mapping[str, Any]] = {}
    for decision_id, row in decisions.items():
        kind = row.get("decision_type")
        if kind == "requirement":
            prompt_refs = row.get("prompt_refs", [])
            evidence_refs = row.get("evidence_refs", [])
            if not isinstance(prompt_refs, list) or not isinstance(evidence_refs, list):
                raise ValueError(
                    "PROMPT_STATE_DECISION: requirement provenance refs must be arrays"
                )
            if not prompt_refs and not evidence_refs:
                raise ValueError("PROMPT_STATE_DECISION: requirement lacks provenance")
            if not set(prompt_refs).issubset(prompt_refs_allowed):
                raise ValueError("PROMPT_STATE_DECISION: requirement cites unknown prompt refs")
            if not set(evidence_refs).issubset(sufficient_refs):
                raise ValueError("PROMPT_STATE_DECISION: requirement cites unknown evidence")
            requirement_id = str(row.get("requirement_id") or "")
            if not requirement_id or requirement_id in requirements:
                raise ValueError(
                    "PROMPT_STATE_DECISION: duplicate or missing requirement ID"
                )
            requirements[requirement_id] = row
        elif kind == "detailed_implementation_plan":
            requirement_id = str(row.get("requirement_ref") or "")
            if not requirement_id or requirement_id in details:
                raise ValueError(
                    "PROMPT_STATE_DECISION: duplicate or missing implementation detail"
                )
            details[requirement_id] = row
            from .planning_detail_template import normalize_required_sections, validate_worksheet

            raw_required_sections = row.get("required_detail_sections")
            if raw_required_sections is not None and not isinstance(raw_required_sections, list):
                raise ValueError(
                    "PROMPT_STATE_DECISION: required_detail_sections must be a host-authored array"
                )
            required_sections = normalize_required_sections(raw_required_sections)
            if raw_required_sections is not None and list(required_sections) != raw_required_sections:
                raise ValueError(
                    "PROMPT_STATE_DECISION: required_detail_sections must use canonical order"
                )
            validate_worksheet(
                row.get("engineering_worksheet"),
                sufficient_refs,
                required_sections,
            )
            for field in ("implementation_capabilities", "implementation_obligations"):
                obligations = row.get(field)
                if not isinstance(obligations, list) or not obligations:
                    raise ValueError(
                        "PROMPT_STATE_DECISION: implementation detail is empty"
                    )
                for obligation in obligations:
                    if not isinstance(obligation, Mapping):
                        raise ValueError(
                            "PROMPT_STATE_DECISION: implementation obligation must be an object"
                        )
                    cited = obligation.get("evidence_refs", [])
                    if not cited or not set(cited).issubset(sufficient_refs):
                        raise ValueError(
                            "PROMPT_STATE_DECISION: implementation lacks real evidence"
                        )
        else:
            raise ValueError(
                f"PROMPT_STATE_DECISION: unsupported decision type for {decision_id}"
            )

    allowed_verdicts = {"reuse", "adapt", "new", "reject", "blocked"}
    for row in candidates.values():
        requirement_ref = str(row.get("requirement_ref") or "")
        if requirement_ref and requirement_ref not in requirements:
            raise ValueError(
                "PROMPT_STATE_CANDIDATE: candidate references unknown requirement"
            )
        if row.get("verdict") not in allowed_verdicts:
            raise ValueError("PROMPT_STATE_CANDIDATE: invalid reuse verdict")
        cited = row.get("evidence", [])
        missing = row.get("missing", [])
        if not isinstance(cited, list) or not isinstance(missing, list):
            raise ValueError(
                "PROMPT_STATE_CANDIDATE: evidence and missing must be arrays"
            )
        if not set(cited).issubset(sufficient_refs):
            raise ValueError("PROMPT_STATE_CANDIDATE: candidate cites unknown evidence")

    blockers = state.get("blockers")
    if not isinstance(blockers, list) or any(not isinstance(row, Mapping) for row in blockers):
        raise ValueError("PROMPT_STATE_BLOCKER: blockers must contain objects")
    for row in blockers:
        caused_by = row.get("caused_by")
        if caused_by is not None:
            if not isinstance(caused_by, list) or any(str(uid) not in unknowns for uid in caused_by):
                raise ValueError("PROMPT_STATE_BLOCKER: blocker cites unknown unresolved ID")
        single = str(row.get("unresolved_id") or "")
        if single and single not in unknowns:
            raise ValueError("PROMPT_STATE_BLOCKER: blocker cites unknown unresolved ID")

    if set(details) - set(requirements):
        raise ValueError(
            "PROMPT_STATE_COVERAGE: implementation references unknown requirement"
        )
    if state.get("plan_ready"):
        if any(row.get("status") != "resolved" for row in unknowns.values()):
            raise ValueError(
                "PROMPT_STATE_READY: unresolved user intent or research blocks planning"
            )
        if blockers:
            raise ValueError("PROMPT_STATE_READY: active blockers remain")
        coverage = state.get("coverage", [])
        if not isinstance(coverage, list):
            raise ValueError("PROMPT_STATE_COVERAGE: coverage must be an array")
        if len(coverage) != len(requirements) or not requirements or set(details) != set(requirements):
            raise ValueError(
                "PROMPT_STATE_COVERAGE: every requirement needs implementation and coverage"
            )
        covered: set[str] = set()
        for row in coverage:
            if not isinstance(row, Mapping):
                raise ValueError("PROMPT_STATE_COVERAGE: coverage rows must be objects")
            requirement_id = str(row.get("requirement_ref") or "")
            if (
                requirement_id not in details
                or requirement_id in covered
                or row.get("status") != "covered"
                or row.get("detailed_plan_ref") != details[requirement_id]["decision_id"]
            ):
                raise ValueError("PROMPT_STATE_COVERAGE: invalid coverage claim")
            covered.add(requirement_id)


__all__ = ["validate_state_links"]
