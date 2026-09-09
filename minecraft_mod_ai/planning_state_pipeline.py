from __future__ import annotations

"""Single prompt-first planning state machine with durable transition snapshots.

No requirement catalog, implementation plan, or retrieval query exists before the
preceding state is available. Every transition persists its complete input/output state
as a trace artifact, so a downstream failure cannot erase the state that caused it.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, TypeVar

from .planner_trace_artifacts import repository_revision
from .planning_convergence_contract import (
    collect_planning_state_research_convergent,
    compile_researched_requirements_convergent,
    emit_planning_goal_satisfied,
)
from .planning_detail_applicability import (
    apply_host_detail_section_applicability,
    ensure_host_detail_section_applicability,
    required_sections_by_requirement,
)
from .planning_state_adaptive_implementation import compile_progress_monotone_detailed_plans
from .planning_state_contract import build_initial_planning_state, validate_planning_state
from .root_cause_trace import emit_root_cause, traced_callable

_T = TypeVar("_T")
DetailSectionApplicabilityResolver = Callable[
    [tuple[str, ...]],
    Mapping[str, Mapping[str, str]],
]
_STATE_COLLECTIONS = (
    "known",
    "references",
    "unresolved",
    "research_queue",
    "evidence",
    "resolved",
    "decisions",
    "implementation_candidates",
    "coverage",
    "blockers",
)


def _state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state_sha256": str(state.get("state_sha256") or ""),
        "plan_ready": state.get("plan_ready"),
        "counts": {
            key: len(value) if isinstance((value := state.get(key)), list) else None
            for key in _STATE_COLLECTIONS
        },
    }


def _trace_state_snapshot(
    event: str,
    operation: str,
    state: Mapping[str, Any],
    *,
    result: str = "SNAPSHOT",
    reason: str = "",
) -> None:
    """Persist the complete state in an unabridged trace artifact.

    ``emit_root_cause`` keeps the console/journal representation bounded but writes its
    original ``details`` object through ``save_trace_artifact`` before bounding it. That
    gives every transition a durable full-state artifact without flooding stderr.
    """
    emit_root_cause(
        event,
        stage="planning_state",
        operation=operation,
        result=result,
        reason=reason,
        details={
            **_state_summary(state),
            "state": deepcopy(dict(state)),
        },
    )


def _transition(
    operation: str,
    callback: Callable[[], _T],
    *,
    input_state: Mapping[str, Any] | None = None,
) -> _T:
    if input_state is not None:
        _trace_state_snapshot(
            "planning_state_transition_input",
            operation,
            input_state,
        )
    value = traced_callable(
        callback,
        stage="planning_state",
        operation=operation,
    )()
    if isinstance(value, Mapping):
        _trace_state_snapshot(
            "planning_state_transition_output",
            operation,
            value,
        )
    else:
        emit_root_cause(
            "planning_state_transition_output",
            stage="planning_state",
            operation=operation,
            result="SNAPSHOT",
            details={"value": value},
        )
    return value


def _requirements_exist(state: Mapping[str, Any]) -> bool:
    decisions = state.get("decisions")
    return any(
        isinstance(item, Mapping) and item.get("decision_type") == "requirement"
        for item in decisions if isinstance(decisions, list)
    )


def _requirement_ids(state: Mapping[str, Any]) -> tuple[str, ...]:
    decisions = state.get("decisions")
    if not isinstance(decisions, list):
        return ()
    return tuple(
        str(item.get("requirement_id") or "")
        for item in decisions
        if isinstance(item, Mapping) and item.get("decision_type") == "requirement"
    )


def _stage_unknowns(state: Mapping[str, Any], stage: str) -> list[Mapping[str, Any]]:
    rows = state.get("unresolved")
    if not isinstance(rows, list):
        return []
    return [
        item
        for item in rows
        if isinstance(item, Mapping)
        and item.get("status") != "resolved"
        and isinstance(item.get("blocks"), list)
        and stage in item.get("blocks", [])
    ]


def _block_summary(state: Mapping[str, Any], *, stage: str) -> str:
    stage_unknowns = _stage_unknowns(state, stage)
    unresolved = [
        (
            str(item.get("unresolved_id") or "?"),
            str(item.get("reason") or "unknown"),
            str(item.get("resolution_route") or "unknown"),
            str(item.get("status") or "unknown"),
        )
        for item in stage_unknowns
    ]
    relevant_research_refs = {
        str(item.get("research_ref") or "") for item in stage_unknowns
    }
    research = [
        (
            str(item.get("research_id") or "?"),
            str(item.get("status") or "unknown"),
            str(item.get("requirement_ref") or ""),
        )
        for item in state.get("research_queue", [])
        if isinstance(item, Mapping)
        and item.get("status") != "complete"
        and (
            not relevant_research_refs
            or str(item.get("research_id") or "") in relevant_research_refs
        )
    ]
    blockers = [
        str(item.get("statement") or item.get("stage") or item.get("blocker_id") or "unknown")
        for item in state.get("blockers", [])
        if isinstance(item, Mapping)
        and (
            not stage_unknowns
            or not item.get("unresolved_id")
            or str(item.get("unresolved_id") or "")
            in {str(row.get("unresolved_id") or "") for row in stage_unknowns}
        )
    ]
    diagnostics = [
        (
            str(item.get("research_ref") or "?"),
            dict(item.get("diagnostics") or {}),
        )
        for item in state.get("evidence", [])
        if isinstance(item, Mapping)
        and item.get("sufficient") is not True
        and (
            not relevant_research_refs
            or str(item.get("research_ref") or "") in relevant_research_refs
        )
    ]
    return (
        f"stage={stage}; unresolved={unresolved}; research={research}; "
        f"blockers={blockers}; diagnostics={diagnostics}"
    )


def _raise_blocked(state: Mapping[str, Any], *, operation: str, message: str) -> None:
    _trace_state_snapshot(
        "planning_state_blocked",
        operation,
        state,
        result="FAIL",
        reason=message,
    )
    raise ValueError(message)


def prepare_planning_state(
    router: Any,
    prompt: str,
    *,
    trace_metadata: Mapping[str, Any] | None = None,
    existing_state: Mapping[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    detail_section_applicability_resolver: DetailSectionApplicabilityResolver | None = None,
) -> dict[str, Any]:
    """Resolve prompt meaning, reference scope, implementation evidence, and plan detail.

    The optional applicability resolver is a trusted host boundary. It receives only
    opaque requirement IDs, never prompt text, requirement prose, model output, or
    evidence. Any omitted requirement/facet therefore remains unknown and keeps the
    full fail-safe worksheet branch.
    """

    emit_root_cause(
        "planning_state_runtime_identity",
        stage="planning_state",
        operation="prepare_planning_state",
        result="START",
        details={
            **repository_revision(),
            "module_file": __file__,
            "restored_state": existing_state is not None,
            "trace_metadata": dict(trace_metadata or {}),
        },
    )

    state = _transition(
        "bootstrap_or_restore",
        lambda: (
            deepcopy(dict(existing_state))
            if existing_state is not None
            else build_initial_planning_state(router, prompt)
        ),
        input_state=existing_state,
    )
    # Fresh states are already validated inside build_initial_planning_state(). Only a
    # restored checkpoint needs another entry-boundary validation here.
    if existing_state is not None:
        from .planning_detail_checkpoint import refresh_worksheet_checkpoint

        state = _transition(
            "refresh_worksheet_checkpoint",
            lambda: refresh_worksheet_checkpoint(state),
            input_state=state,
        )
        _transition(
            "validate_restored_state",
            lambda: validate_planning_state(state, prompt=prompt),
            input_state=state,
        )
    if state.get("plan_ready") is True:
        emit_planning_goal_satisfied(state)
        return state
    if checkpoint is not None:
        checkpoint(deepcopy(state))

    if not _requirements_exist(state):
        if state.get("research_queue"):
            state = _transition(
                "collect_prompt_research",
                lambda: collect_planning_state_research_convergent(
                    router,
                    prompt,
                    state,
                    trace_metadata=trace_metadata,
                ),
                input_state=state,
            )
            if checkpoint is not None:
                checkpoint(deepcopy(state))

        state = _transition(
            "compile_researched_requirements",
            lambda: compile_researched_requirements_convergent(router, prompt, state),
            input_state=state,
        )
        if checkpoint is not None:
            checkpoint(deepcopy(state))

        if not _requirements_exist(state):
            _raise_blocked(
                state,
                operation="requirement_selection",
                message=(
                    "PLANNING_REQUIREMENT_SELECTION_BLOCKED: "
                    + _block_summary(state, stage="requirement_selection")
                ),
            )

    if detail_section_applicability_resolver is None:
        state = _transition(
            "normalize_detail_section_applicability",
            lambda: ensure_host_detail_section_applicability(state),
            input_state=state,
        )
    else:
        applicability_by_requirement = _transition(
            "resolve_detail_section_applicability",
            lambda: detail_section_applicability_resolver(_requirement_ids(state)),
            input_state=state,
        )
        state = _transition(
            "apply_detail_section_applicability",
            lambda: apply_host_detail_section_applicability(
                state,
                applicability_by_requirement,
            ),
            input_state=state,
        )
    if checkpoint is not None:
        checkpoint(deepcopy(state))

    state = _transition(
        "collect_implementation_research",
        lambda: collect_planning_state_research_convergent(
            router,
            prompt,
            state,
            trace_metadata=trace_metadata,
        ),
        input_state=state,
    )
    if checkpoint is not None:
        checkpoint(deepcopy(state))

    # Stop on the actual research blocker and persist the complete blocked state before
    # any detailed-plan invariant can replace the first cause.
    if _stage_unknowns(state, "implementation_plan"):
        _raise_blocked(
            state,
            operation="implementation_plan",
            message=(
                "PLANNING_IMPLEMENTATION_RESEARCH_BLOCKED: "
                + _block_summary(state, stage="implementation_plan")
            ),
        )

    section_selection = _transition(
        "select_detail_sections",
        lambda: required_sections_by_requirement(state),
        input_state=state,
    )
    state = _transition(
        "compile_progress_monotone_detailed_plans",
        lambda: compile_progress_monotone_detailed_plans(
            router,
            prompt,
            state,
            required_sections_by_requirement=section_selection,
            checkpoint=checkpoint,
        ),
        input_state=state,
    )
    _transition(
        "validate_final",
        lambda: validate_planning_state(state, prompt=prompt),
        input_state=state,
    )
    if state.get("plan_ready") is not True:
        _raise_blocked(
            state,
            operation="final_readiness",
            message="PLANNING_STATE_NOT_READY: planning state did not reach code-ready coverage",
        )
    emit_planning_goal_satisfied(state)
    if checkpoint is not None:
        checkpoint(deepcopy(state))
    return state


__all__ = ["DetailSectionApplicabilityResolver", "prepare_planning_state"]
