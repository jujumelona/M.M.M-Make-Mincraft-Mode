from __future__ import annotations

"""Finite monotone convergence for the prompt-first planner.

Normal termination is semantic, never an arbitrary round budget: research works over a
host-owned finite obligation universe; terminal states never reopen; and fixed-point or
frontier exhaustion is terminal. Only requirement compilation may add the finite set of
implementation obligations.
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

_RESEARCH_TRANSITIONS = {
    "pending": frozenset({"pending", "complete", "blocked"}),
    "complete": frozenset({"complete"}),
    "blocked": frozenset({"blocked"}),
}
_UNRESOLVED_TRANSITIONS = {
    "open": frozenset({"open", "resolved", "blocked"}),
    "resolved": frozenset({"resolved"}),
    "blocked": frozenset({"blocked"}),
}
_IMPLEMENTATION_STAGE = "implementation_plan"


class PlanningConvergenceError(RuntimeError):
    """A host-owned convergence invariant was violated."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _id_map(
    state: Mapping[str, Any], collection: str, field: str
) -> dict[str, Mapping[str, Any]]:
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


def _requirements(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = state.get("decisions")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, Mapping) and row.get("decision_type") == "requirement":
            requirement_id = str(row.get("requirement_id") or "")
            if requirement_id:
                result[requirement_id] = row
    return result


def _evidence_key(row: Mapping[str, Any]) -> str:
    # Claim prose is deliberately excluded: wording churn is not semantic progress.
    return _canonical(
        {
            "research_ref": str(row.get("research_ref") or ""),
            "evidence_refs": sorted(str(x) for x in (row.get("evidence_refs") or [])),
            "sufficient": row.get("sufficient") is True,
            "source": str(row.get("source") or ""),
        }
    )


def planning_progress_fingerprint(state: Mapping[str, Any]) -> str:
    """Canonical semantic state, excluding diagnostics/order/prose-only churn."""
    unresolved = _id_map(state, "unresolved", "unresolved_id")
    research = _id_map(state, "research_queue", "research_id")
    requirements = _requirements(state)
    evidence = state.get("evidence")
    resolved = state.get("resolved")
    coverage = state.get("coverage")
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
                tuple(sorted(str(x) for x in (row.get("resolves") or []))),
            )
            for rid, row in research.items()
        ),
        "evidence": sorted(
            _evidence_key(row)
            for row in evidence
            if isinstance(evidence, list) and isinstance(row, Mapping)
        ),
        "resolved": sorted(
            (
                str(row.get("unresolved_id") or ""),
                str(row.get("basis") or ""),
                tuple(sorted(str(x) for x in (row.get("evidence_refs") or []))),
            )
            for row in resolved
            if isinstance(resolved, list) and isinstance(row, Mapping)
        ),
        "requirements": sorted(
            (rid, str(row.get("statement") or ""))
            for rid, row in requirements.items()
        ),
        "coverage": sorted(
            _canonical(row)
            for row in coverage
            if isinstance(coverage, list) and isinstance(row, Mapping)
        ),
        "plan_ready": state.get("plan_ready") is True,
    }
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return "sha256:" + digest


def _rehash(state: dict[str, Any]) -> None:
    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")


def _dedupe(state: dict[str, Any]) -> None:
    evidence = state.get("evidence")
    if isinstance(evidence, list):
        seen: set[str] = set()
        kept: list[Any] = []
        for row in evidence:
            key = _evidence_key(row) if isinstance(row, Mapping) else _canonical(row)
            if key not in seen:
                seen.add(key)
                kept.append(row)
        state["evidence"] = kept

    resolved = state.get("resolved")
    if isinstance(resolved, list):
        seen_ids: set[str] = set()
        kept_resolved: list[Any] = []
        for row in resolved:
            uid = str(row.get("unresolved_id") or "") if isinstance(row, Mapping) else ""
            if uid and uid in seen_ids:
                continue
            if uid:
                seen_ids.add(uid)
            kept_resolved.append(row)
        state["resolved"] = kept_resolved


def _ensure_implementation_blocks(state: dict[str, Any]) -> bool:
    """Migrate old checkpoints and make implementation evidence a real plan gate."""
    rows = state.get("unresolved")
    if not isinstance(rows, list):
        return False
    changed = False
    for row in rows:
        if not isinstance(row, dict) or row.get("reason") != "implementation_method":
            continue
        blocks = [str(x) for x in (row.get("blocks") or []) if str(x)]
        if _IMPLEMENTATION_STAGE not in blocks:
            row["blocks"] = [*blocks, _IMPLEMENTATION_STAGE]
            changed = True
    return changed


def _terminalize_research_frontier(state: dict[str, Any]) -> bool:
    """Pending after one exhaustive pass and already-blocked research are terminal."""
    research = _id_map(state, "research_queue", "research_id")
    unresolved = _id_map(state, "unresolved", "unresolved_id")
    terminal_refs: set[str] = set()
    changed = False
    for rid, row in research.items():
        status = row.get("status")
        if status == "pending":
            if not isinstance(row, dict):
                raise PlanningConvergenceError(
                    "PLANNING_CONVERGENCE_SHAPE: research rows must be mutable objects"
                )
            row["status"] = "blocked"
            status = "blocked"
            changed = True
        if status == "blocked":
            terminal_refs.add(rid)
    for row in unresolved.values():
        if (
            row.get("status") == "open"
            and str(row.get("research_ref") or "") in terminal_refs
        ):
            if not isinstance(row, dict):
                raise PlanningConvergenceError(
                    "PLANNING_CONVERGENCE_SHAPE: unresolved rows must be mutable objects"
                )
            row["status"] = "blocked"
            changed = True
    return changed


def _emit(
    reason: str,
    state: Mapping[str, Any],
    *,
    operation: str,
    before_fingerprint: str | None,
) -> None:
    unresolved = _id_map(state, "unresolved", "unresolved_id")
    research = _id_map(state, "research_queue", "research_id")
    emit_root_cause(
        "planning_convergence",
        stage="planning_state",
        operation=operation,
        result="STOP" if reason != "PLANNING_PROGRESS_DELTA" else "PROGRESS",
        reason=reason,
        details={
            "termination_reason": reason,
            "fingerprint_before": before_fingerprint,
            "fingerprint_after": planning_progress_fingerprint(state),
            "obligation_universe_size": len(unresolved),
            "research_universe_size": len(research),
            "unresolved_open": sum(r.get("status") == "open" for r in unresolved.values()),
            "unresolved_blocked": sum(r.get("status") == "blocked" for r in unresolved.values()),
            "unresolved_resolved": sum(r.get("status") == "resolved" for r in unresolved.values()),
            "research_pending": sum(r.get("status") == "pending" for r in research.values()),
            "research_complete": sum(r.get("status") == "complete" for r in research.values()),
            "research_blocked": sum(r.get("status") == "blocked" for r in research.values()),
        },
    )


def assert_research_transition_monotone(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    """Research may advance facts/statuses but cannot alter its frozen ID universe."""
    old_unknowns = _id_map(before, "unresolved", "unresolved_id")
    new_unknowns = _id_map(after, "unresolved", "unresolved_id")
    old_research = _id_map(before, "research_queue", "research_id")
    new_research = _id_map(after, "research_queue", "research_id")
    if set(old_unknowns) != set(new_unknowns) or set(old_research) != set(new_research):
        raise PlanningConvergenceError(
            "PLANNING_OUT_OF_UNIVERSE_OBLIGATION: research changed the frozen obligation universe"
        )

    for uid, old in old_unknowns.items():
        old_status = str(old.get("status") or "")
        new_status = str(new_unknowns[uid].get("status") or "")
        if new_status not in _UNRESOLVED_TRANSITIONS.get(old_status, frozenset()):
            raise PlanningConvergenceError(
                f"PLANNING_NON_MONOTONE_TRANSITION: unresolved {uid} moved "
                f"{old_status!r}->{new_status!r}"
            )
    for rid, old in old_research.items():
        old_status = str(old.get("status") or "")
        new_status = str(new_research[rid].get("status") or "")
        if new_status not in _RESEARCH_TRANSITIONS.get(old_status, frozenset()):
            raise PlanningConvergenceError(
                f"PLANNING_NON_MONOTONE_TRANSITION: research {rid} moved "
                f"{old_status!r}->{new_status!r}"
            )

    old_evidence = {
        _evidence_key(row)
        for row in before.get("evidence", [])
        if isinstance(row, Mapping)
    }
    new_evidence = {
        _evidence_key(row)
        for row in after.get("evidence", [])
        if isinstance(row, Mapping)
    }
    if not old_evidence.issubset(new_evidence):
        raise PlanningConvergenceError(
            "PLANNING_NON_MONOTONE_TRANSITION: research discarded known evidence"
        )
    old_resolved = {
        str(row.get("unresolved_id") or "")
        for row in before.get("resolved", [])
        if isinstance(row, Mapping)
    }
    new_resolved = {
        str(row.get("unresolved_id") or "")
        for row in after.get("resolved", [])
        if isinstance(row, Mapping)
    }
    if not old_resolved.issubset(new_resolved):
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
    """Process one finite research delta and make exhaustion/fixed-point terminal."""
    validate_planning_state(state, prompt=prompt)
    before = deepcopy(dict(state))
    if _ensure_implementation_blocks(before):
        _rehash(before)
        validate_planning_state(before, prompt=prompt)
    before_fp = planning_progress_fingerprint(before)
    pending = any(
        row.get("status") == "pending"
        for row in _id_map(before, "research_queue", "research_id").values()
    )
    policy_open = any(
        row.get("status") == "open" and row.get("resolution_route") == "default_policy"
        for row in _id_map(before, "unresolved", "unresolved_id").values()
    )
    if not pending and not policy_open:
        active = any(
            row.get("status") != "resolved"
            for row in _id_map(before, "unresolved", "unresolved_id").values()
        )
        _emit(
            "PLANNING_FRONTIER_EXHAUSTED" if active else "PLANNING_RESEARCH_GOAL_SATISFIED",
            before,
            operation="collect_planning_state_research_convergent",
            before_fingerprint=before_fp,
        )
        return before

    value = deepcopy(
        dict(
            collect_planning_state_research(
                router, prompt, before, trace_metadata=trace_metadata
            )
        )
    )
    _ensure_implementation_blocks(value)
    _dedupe(value)
    _rehash(value)
    validate_planning_state(value, prompt=prompt)
    assert_research_transition_monotone(before, value)
    raw_after_fp = planning_progress_fingerprint(value)

    if _terminalize_research_frontier(value):
        _dedupe(value)
        _rehash(value)
        validate_planning_state(value, prompt=prompt)
        assert_research_transition_monotone(before, value)

    active = any(
        row.get("status") != "resolved"
        for row in _id_map(value, "unresolved", "unresolved_id").values()
    )
    if raw_after_fp == before_fp and active:
        reason = "PLANNING_FIXED_POINT_UNRESOLVED"
    elif active:
        reason = "PLANNING_FRONTIER_EXHAUSTED"
    else:
        reason = "PLANNING_RESEARCH_GOAL_SATISFIED"
    _emit(
        reason,
        value,
        operation="collect_planning_state_research_convergent",
        before_fingerprint=before_fp,
    )
    return value


def compile_researched_requirements_convergent(
    router: Any, prompt: str, state: Mapping[str, Any]
) -> dict[str, Any]:
    """Allow the finite implementation-universe expansion once, then freeze it."""
    validate_planning_state(state, prompt=prompt)
    if _requirements(state):
        value = deepcopy(dict(state))
        if _ensure_implementation_blocks(value):
            _rehash(value)
            validate_planning_state(value, prompt=prompt)
        fp = planning_progress_fingerprint(value)
        _emit(
            "PLANNING_FIXED_POINT_REQUIREMENTS_ALREADY_FROZEN",
            value,
            operation="compile_researched_requirements_convergent",
            before_fingerprint=fp,
        )
        return value

    old_unknowns = _id_map(state, "unresolved", "unresolved_id")
    old_research = _id_map(state, "research_queue", "research_id")
    before_fp = planning_progress_fingerprint(state)
    value = deepcopy(dict(compile_researched_requirements(router, prompt, state)))
    _ensure_implementation_blocks(value)
    _rehash(value)
    validate_planning_state(value, prompt=prompt)

    requirements = _requirements(value)
    new_unknowns = _id_map(value, "unresolved", "unresolved_id")
    new_research = _id_map(value, "research_queue", "research_id")
    if not set(old_unknowns).issubset(new_unknowns) or not set(old_research).issubset(new_research):
        raise PlanningConvergenceError(
            "PLANNING_NON_MONOTONE_TRANSITION: requirement compilation removed obligations"
        )
    added_unknowns = set(new_unknowns) - set(old_unknowns)
    added_research = set(new_research) - set(old_research)

    if requirements:
        if len(added_unknowns) != len(requirements) or len(added_research) != len(requirements):
            raise PlanningConvergenceError(
                "PLANNING_OUT_OF_UNIVERSE_OBLIGATION: expected exactly one implementation "
                "obligation and research item per requirement"
            )
        seen_refs: set[str] = set()
        for uid in added_unknowns:
            row = new_unknowns[uid]
            requirement_ref = str(row.get("requirement_ref") or "")
            research_ref = str(row.get("research_ref") or "")
            valid = (
                row.get("reason") == "implementation_method"
                and row.get("status") == "open"
                and requirement_ref in requirements
                and requirement_ref not in seen_refs
                and research_ref in added_research
                and _IMPLEMENTATION_STAGE in (row.get("blocks") or [])
            )
            if not valid:
                raise PlanningConvergenceError(
                    "PLANNING_OUT_OF_UNIVERSE_OBLIGATION: invalid implementation obligation expansion"
                )
            seen_refs.add(requirement_ref)
        for rid in added_research:
            row = new_research[rid]
            if (
                row.get("status") != "pending"
                or str(row.get("requirement_ref") or "") not in requirements
            ):
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
            "fingerprint_before": before_fp,
            "fingerprint_after": planning_progress_fingerprint(value),
            "requirements": sorted(requirements),
            "added_unresolved": sorted(added_unknowns),
            "added_research": sorted(added_research),
            "obligation_universe_size": len(new_unknowns),
            "research_universe_size": len(new_research),
        },
    )
    return value


def emit_planning_goal_satisfied(state: Mapping[str, Any]) -> None:
    _emit(
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
