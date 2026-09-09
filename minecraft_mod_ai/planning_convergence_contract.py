from __future__ import annotations

"""Research-backed convergence contract for the prompt-first planner.

The planner terminates by finite-frontier exhaustion or semantic fixed point, never by an
arbitrary round budget.  This mirrors monotone worklist/fixed-point algorithms: research
may only move host-owned obligations forward inside a frozen universe; only the explicit
requirement-compilation boundary may add the finite implementation-obligation set.
"""

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .planning_state_contract import _hash_without, validate_planning_state
from .planning_state_research import collect_planning_state_research
from .planning_state_resolution import compile_researched_requirements
from .root_cause_trace import emit_root_cause

_RESEARCH_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"pending", "complete", "blocked"}),
    "complete": frozenset({"complete"}),
    "blocked": frozenset({"blocked"}),
}
_UNRESOLVED_TRANSITIONS: dict[str, frozenset[str]] = {
    "open": frozenset({"open", "resolved", "blocked"}),
    "resolved": frozenset({"resolved"}),
    "blocked": frozenset({"blocked"}),
}


class PlanningConvergenceError(RuntimeError):
    """A host-owned planning convergence invariant was violated."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _id_map(state: Mapping[str, Any], collection: str, field: str) -> dict[str, Mapping[str, Any]]:
    rows = state.get(collection)
    if not isinstance(rows, list):
        raise PlanningConvergenceError(
            f"PLANNING_CONVERGENCE_SHAPE: {collection} must be an array"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PlanningConvergenceError(
                f"PLANNING_CONVERGENCE_SHAPE: {collection} rows must be objects"
            )
        identifier = str(row.get(field) or "")
        if not identifier or identifier in result:
            raise PlanningConvergenceError(
                f"PLANNING_CONVERGENCE_IDS: {collection} IDs must be non-empty and unique"
            )
        result[identifier] = row
    return result


def _requirement_map(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = state.get("decisions")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping) or row.get("decision_type") != "requirement":
            continue
        requirement_id = str(row.get("requirement_id") or "")
        if requirement_id:
            result[requirement_id] = row
    return result


def _evidence_semantic_key(row: Mapping[str, Any]) -> str:
    return _canonical(
        {
            "research_ref": str(row.get("research_ref") or ""),
            "claims": row.get("claims") or [],
            "evidence_refs": sorted(str(item) for item in (row.get("evidence_refs") or [])),
            "sufficient": row.get("sufficient") is True,
            "source": str(row.get("source") or ""),
        }
    )


def planning_progress_fingerprint(state: Mapping[str, Any]) -> str:
    """Fingerprint semantic planner progress while ignoring volatile diagnostics/order."""
    unresolved = _id_map(state, "unresolved", "unresolved_id")
    research = _id_map(state, "research_queue", "research_id")
    requirements = _requirement_map(state)
    evidence_rows = state.get("evidence")
    resolved_rows = state.get("resolved")
    coverage_rows = state.get("coverage")
    payload = {
        "unresolved": sorted(
            (
                uid,
                str(row.get("status") or ""),
                str(row.get("reason") or ""),
                str(row.get("research_ref") or ""),
            )
            for uid, row in unresolved.items()
        ),
        "research": sorted(
            (
                rid,
                str(row.get("status") or ""),
                tuple(sorted(str(item) for item in (row.get("resolves") or []))),
            )
            for rid, row in research.items()
        ),
        "evidence": sorted(
            _evidence_semantic_key(row)
            for row in evidence_rows if isinstance(evidence_rows, list) and isinstance(row, Mapping)
        ),
        "resolved": sorted(
            (
                str(row.get("unresolved_id") or ""),
                str(row.get("basis") or ""),
                _canonical(row.get("evidence_refs") or []),
            )
            for row in resolved_rows if isinstance(resolved_rows, list) and isinstance(row, Mapping)
        ),
        "requirements": sorted(
            (rid, str(row.get("statement") or "")) for rid, row in requirements.items()
        ),
        "coverage": sorted(
            _canonical(row)
            for row in coverage_rows if isinstance(coverage_rows, list) and isinstance(row, Mapping)
        ),
        "plan_ready": state.get("plan_ready") is True,
    }
    return "sha256:" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _dedupe_semantic_rows(state: dict[str, Any]) -> None:
    evidence = state.get("evidence")
    if isinstance(evidence, list):
        seen: set[str] = set()
        deduped: list[Any] = []
        for row in evidence:
            if not isinstance(row, Mapping):
                deduped.append(row)
                continue
            key = _evidence_semantic_key(row)
            if key not in seen:
                seen.add(key)
                deduped.append(row)
        state["evidence"] = deduped

    resolved = state.get("resolved")
    if isinstance(resolved, list):
        seen_ids: set[str] = set()
        deduped_resolved: list[Any] = []
        for row in resolved:
            uid = str(row.get("unresolved_id") or "") if isinstance(row, Mapping) else ""
            if uid and uid in seen_ids:
                continue
            if uid:
                seen_ids.add(uid)
            deduped_resolved.append(row)
        state["resolved"] = deduped_resolved

    blockers = state.get("blockers")
    if isinstance(blockers, list):
        seen_blockers: set[str] = set()
        deduped_blockers: list[Any] = []
        for row in blockers:
            if isinstance(row, Mapping):
                key = _canonical(
                    {
                        "unresolved_id": str(row.get("unresolved_id") or ""),
                        "stage": str(row.get("stage") or ""),
                        "statement": str(row.get("statement") or ""),
                        "caused_by": sorted(str(item) for item in (row.get("caused_by") or [])),
                    }
                )
            else:
                key = _canonical(row)
            if key not in seen_blockers:
                seen_blockers.add(key)
                deduped_blockers.append(row)
        state["blockers"] = deduped_blockers


def _emit_convergence(
    reason: str,
    state: Mapping[str, Any],
    *,
    operation: str,
    before_fingerprint: str | None = None,
) -> None:
    unresolved = _id_map(state, "unresolved", "unresolved_id")
    research = _id_map(state, "research_queue", "research_id")
    after_fingerprint = planning_progress_fingerprint(state)
    emit_root_cause(
        "planning_convergence",
        stage="planning_state",
        operation=operation,
        result="STOP" if reason != "PLANNING_PROGRESS_DELTA" else "PROGRESS",
        reason=reason,
        details={
            "termination_reason": reason,
            "fingerprint_before": before_fingerprint,
            "fingerprint_after": after_fingerprint,
            "obligation_universe_size": len(unresolved),
            "research_universe_size": len(research),
            "unresolved_open": sum(
                row.get("status") != "resolved" for row in unresolved.values()
            ),
            "research_pending": sum(
                row.get("status") == "pending" for row in research.values()
            ),
            "research_complete": sum(
                row.get("status") == "complete" for row in research.values()
            ),
            "research_blocked": sum(
                row.get("status") == "blocked" for row in research.values()
            ),
        },
    )


def assert_research_transition_monotone(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """Research may advance statuses/evidence but may not change its frozen universe."""
    before_unknowns = _id_map(before, "unresolved", "unresolved_id")
    after_unknowns = _id_map(after, "unresolved", "unresolved_id")
    before_research = _id_map(before, "research_queue", "research_id")
    after_research = _id_map(after, "research_queue", "research_id")

    if set(before_unknowns) != set(after_unknowns) or set(before_research) != set(after_research):
        raise PlanningConvergenceError(
            "PLANNING_OUT_OF_UNIVERSE_OBLIGATION: research changed the frozen obligation universe"
        )

    for uid, old in before_unknowns.items():
        old_status = str(old.get("status") or "")
        new_status = str(after_unknowns[uid].get("status") or "")
        if new_status not in _UNRESOLVED_TRANSITIONS.get(old_status, frozenset()):
            raise PlanningConvergenceError(
                "PLANNING_NON_MONOTONE_TRANSITION: "
                f"unresolved {uid} moved {old_status!r}->{new_status!r}"
            )

    for rid, old in before_research.items():
        old_status = str(old.get("status") or "")
        new_status = str(after_research[rid].get("status") or "")
        if new_status not in _RESEARCH_TRANSITIONS.get(old_status, frozenset()):
            raise PlanningConvergenceError(
                "PLANNING_NON_MONOTONE_TRANSITION: "
                f"research {rid} moved {old_status!r}->{new_status!r}"
            )

    before_evidence = {
        _evidence_semantic_key(row)
        for row in before.get("evidence", [])
        if isinstance(row, Mapping)
    }
    after_evidence = {
        _evidence_semantic_key(row)
        for row in after.get("evidence", [])
        if isinstance(row, Mapping)
    }
    if not before_evidence.issubset(after_evidence):
        raise PlanningConvergenceError(
            "PLANNING_NON_MONOTONE_TRANSITION: research discarded previously known evidence"
        )

    before_resolved = {
        str(row.get("unresolved_id") or "")
        for row in before.get("resolved", [])
        if isinstance(row, Mapping)
    }
    after_resolved = {
        str(row.get("unresolved_id") or "")
        for row in after.get("resolved", [])
        if isinstance(row, Mapping)
    }
    if not before_resolved.issubset(after_resolved):
        raise PlanningConvergenceError(
            "PLANNING_NON_MONOTONE_TRANSITION: research discarded a resolved obligation"
        )


def collect_planning_state_research_convergent(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
    *,
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one semi-naive research delta over the frozen pending frontier.

    Terminal blocked rows are deliberately masked as complete for the legacy collector so
    its historical blocked->pending reset cannot reopen them.  The original blocked status
    is restored before validation.  If no pending frontier exists, no model/retrieval call
    is made at all.
    """
    validate_planning_state(state, prompt=prompt)
    before = deepcopy(dict(state))
    before_fingerprint = planning_progress_fingerprint(before)
    research_before = _id_map(before, "research_queue", "research_id")
    pending_ids = {
        rid for rid, row in research_before.items() if row.get("status") == "pending"
    }

    if not pending_ids:
        active_unknowns = any(
            row.get("status") != "resolved"
            for row in _id_map(before, "unresolved", "unresolved_id").values()
        )
        _emit_convergence(
            "PLANNING_FRONTIER_EXHAUSTED" if active_unknowns else "PLANNING_RESEARCH_GOAL_SATISFIED",
            before,
            operation="collect_planning_state_research_convergent",
            before_fingerprint=before_fingerprint,
        )
        return before

    protected = deepcopy(before)
    protected_research = _id_map(protected, "research_queue", "research_id")
    terminal_blocked = {
        rid for rid, row in protected_research.items() if row.get("status") == "blocked"
    }
    for rid in terminal_blocked:
        row = protected_research[rid]
        if isinstance(row, dict):
            row["status"] = "complete"
    _rehash(protected)
    validate_planning_state(protected, prompt=prompt)

    value = collect_planning_state_research(
        router,
        prompt,
        protected,
        trace_metadata=trace_metadata,
    )
    value = deepcopy(dict(value))
    research_after = _id_map(value, "research_queue", "research_id")
    for rid in terminal_blocked:
        row = research_after.get(rid)
        if isinstance(row, dict):
            row["status"] = "blocked"

    _dedupe_semantic_rows(value)
    _rehash(value)
    validate_planning_state(value, prompt=prompt)
    assert_research_transition_monotone(before, value)

    after_fingerprint = planning_progress_fingerprint(value)
    pending_after = sum(
        row.get("status") == "pending"
        for row in _id_map(value, "research_queue", "research_id").values()
    )
    active_unknowns = any(
        row.get("status") != "resolved"
        for row in _id_map(value, "unresolved", "unresolved_id").values()
    )
    if after_fingerprint == before_fingerprint:
        reason = "PLANNING_FIXED_POINT_UNRESOLVED" if active_unknowns else "PLANNING_RESEARCH_GOAL_SATISFIED"
    elif pending_after == 0 and active_unknowns:
        reason = "PLANNING_FRONTIER_EXHAUSTED"
    elif pending_after == 0:
        reason = "PLANNING_RESEARCH_GOAL_SATISFIED"
    else:
        reason = "PLANNING_PROGRESS_DELTA"
    _emit_convergence(
        reason,
        value,
        operation="collect_planning_state_research_convergent",
        before_fingerprint=before_fingerprint,
    )
    return value


def compile_researched_requirements_convergent(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand the universe exactly once at the explicit requirement boundary, then freeze it."""
    validate_planning_state(state, prompt=prompt)
    if _requirement_map(state):
        _emit_convergence(
            "PLANNING_FIXED_POINT_REQUIREMENTS_ALREADY_FROZEN",
            state,
            operation="compile_researched_requirements_convergent",
            before_fingerprint=planning_progress_fingerprint(state),
        )
        return deepcopy(dict(state))

    before_unknowns = _id_map(state, "unresolved", "unresolved_id")
    before_research = _id_map(state, "research_queue", "research_id")
    before_fingerprint = planning_progress_fingerprint(state)
    value = compile_researched_requirements(router, prompt, state)
    validate_planning_state(value, prompt=prompt)

    requirements = _requirement_map(value)
    after_unknowns = _id_map(value, "unresolved", "unresolved_id")
    after_research = _id_map(value, "research_queue", "research_id")
    if not set(before_unknowns).issubset(after_unknowns) or not set(before_research).issubset(after_research):
        raise PlanningConvergenceError(
            "PLANNING_NON_MONOTONE_TRANSITION: requirement compilation removed existing obligations"
        )

    added_unknown_ids = set(after_unknowns) - set(before_unknowns)
    added_research_ids = set(after_research) - set(before_research)
    if requirements:
        if len(added_unknown_ids) != len(requirements) or len(added_research_ids) != len(requirements):
            raise PlanningConvergenceError(
                "PLANNING_OUT_OF_UNIVERSE_OBLIGATION: requirement boundary did not create exactly one finite implementation obligation per requirement"
            )
        seen_requirement_refs: set[str] = set()
        for uid in added_unknown_ids:
            row = after_unknowns[uid]
            requirement_ref = str(row.get("requirement_ref") or "")
            research_ref = str(row.get("research_ref") or "")
            if (
                row.get("reason") != "implementation_method"
                or row.get("status") != "open"
                or requirement_ref not in requirements
                or requirement_ref in seen_requirement_refs
                or research_ref not in added_research_ids
            ):
                raise PlanningConvergenceError(
                    "PLANNING_OUT_OF_UNIVERSE_OBLIGATION: invalid implementation obligation expansion"
                )
            seen_requirement_refs.add(requirement_ref)
        for rid in added_research_ids:
            row = after_research[rid]
            if row.get("status") != "pending" or str(row.get("requirement_ref") or "") not in requirements:
                raise PlanningConvergenceError(
                    "PLANNING_OUT_OF_UNIVERSE_OBLIGATION: invalid implementation research expansion"
                )

    emit_root_cause(
        "planning_obligation_universe_frozen",
        stage="planning_state",
        operation="compile_researched_requirements_convergent",
        result="PASS" if requirements else "STOP",
        reason=(
            "PLANNING_OBLIGATION_UNIVERSE_FROZEN"
            if requirements
            else "PLANNING_FRONTIER_EXHAUSTED"
        ),
        details={
            "fingerprint_before": before_fingerprint,
            "fingerprint_after": planning_progress_fingerprint(value),
            "requirements": sorted(requirements),
            "added_unresolved": sorted(added_unknown_ids),
            "added_research": sorted(added_research_ids),
            "obligation_universe_size": len(after_unknowns),
            "research_universe_size": len(after_research),
        },
    )
    return value


def emit_planning_goal_satisfied(state: Mapping[str, Any]) -> None:
    """Record the only successful terminal condition for the whole planning pipeline."""
    _emit_convergence(
        "PLANNING_GOAL_SATISFIED",
        state,
        operation="prepare_planning_state",
        before_fingerprint=None,
    )


__all__ = [
    "PlanningConvergenceError",
    "assert_research_transition_monotone",
    "collect_planning_state_research_convergent",
    "compile_researched_requirements_convergent",
    "emit_planning_goal_satisfied",
    "planning_progress_fingerprint",
]
