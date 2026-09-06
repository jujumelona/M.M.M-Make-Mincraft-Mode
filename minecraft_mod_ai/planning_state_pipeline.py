from __future__ import annotations

"""Single prompt-first planning state machine.

No requirement catalog, implementation plan, or retrieval query exists before the
preceding state is available. Every blocked stage is surfaced before a downstream stage
can overwrite its root cause with a secondary invariant failure.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, TypeVar

from .planning_detail_applicability import (
    ensure_host_detail_section_applicability,
    required_sections_by_requirement,
)
from .planning_state_contract import build_initial_planning_state, validate_planning_state
from .planning_state_implementation import compile_detailed_implementation_plans
from .planning_state_research import collect_planning_state_research
from .planning_state_resolution import compile_researched_requirements
from .root_cause_trace import traced_callable

_T = TypeVar("_T")


def _transition(operation: str, callback: Callable[[], _T]) -> _T:
    return traced_callable(
        callback,
        stage="planning_state",
        operation=operation,
    )()


def _requirements_exist(state: Mapping[str, Any]) -> bool:
    decisions = state.get("decisions")
    return any(
        isinstance(item, Mapping) and item.get("decision_type") == "requirement"
        for item in decisions if isinstance(decisions, list)
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


def prepare_planning_state(
    router: Any,
    prompt: str,
    *,
    trace_metadata: Mapping[str, Any] | None = None,
    existing_state: Mapping[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Resolve prompt meaning, reference scope, implementation evidence, and plan detail."""

    state = _transition(
        "bootstrap_or_restore",
        lambda: (
            deepcopy(dict(existing_state))
            if existing_state is not None
            else build_initial_planning_state(router, prompt)
        ),
    )
    _transition("validate_initial", lambda: validate_planning_state(state, prompt=prompt))
    if state.get("plan_ready") is True:
        return state
    if checkpoint is not None:
        checkpoint(deepcopy(state))

    if not _requirements_exist(state):
        if state.get("research_queue"):
            state = _transition(
                "collect_prompt_research",
                lambda: collect_planning_state_research(
                    router,
                    prompt,
                    state,
                    trace_metadata=trace_metadata,
                ),
            )
            if checkpoint is not None:
                checkpoint(deepcopy(state))

        state = _transition(
            "compile_researched_requirements",
            lambda: compile_researched_requirements(router, prompt, state),
        )
        if checkpoint is not None:
            checkpoint(deepcopy(state))

        if not _requirements_exist(state):
            raise ValueError(
                "PLANNING_REQUIREMENT_SELECTION_BLOCKED: "
                + _block_summary(state, stage="requirement_selection")
            )

    state = _transition(
        "normalize_detail_section_applicability",
        lambda: ensure_host_detail_section_applicability(state),
    )
    if checkpoint is not None:
        checkpoint(deepcopy(state))

    state = _transition(
        "collect_implementation_research",
        lambda: collect_planning_state_research(
            router,
            prompt,
            state,
            trace_metadata=trace_metadata,
        ),
    )
    if checkpoint is not None:
        checkpoint(deepcopy(state))

    # Do not descend into detailed planning while implementation evidence is blocked.
    # That would replace provider/source diagnostics with a secondary DETAILED_PLAN_*
    # invariant and recreate the original failure pattern one stage later.
    if _stage_unknowns(state, "implementation_plan"):
        raise ValueError(
            "PLANNING_IMPLEMENTATION_RESEARCH_BLOCKED: "
            + _block_summary(state, stage="implementation_plan")
        )

    section_selection = _transition(
        "select_detail_sections",
        lambda: required_sections_by_requirement(state),
    )
    state = _transition(
        "compile_detailed_implementation_plans",
        lambda: compile_detailed_implementation_plans(
            router,
            prompt,
            state,
            required_sections_by_requirement=section_selection,
        ),
    )
    _transition("validate_final", lambda: validate_planning_state(state, prompt=prompt))
    if state.get("plan_ready") is not True:
        raise ValueError(
            "PLANNING_STATE_NOT_READY: planning state did not reach code-ready coverage"
        )
    if checkpoint is not None:
        checkpoint(deepcopy(state))
    return state


__all__ = ["prepare_planning_state"]
