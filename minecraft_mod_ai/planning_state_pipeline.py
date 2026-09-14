from __future__ import annotations

"""Single prompt-first planning state machine with durable transition snapshots.

No requirement catalog, implementation plan, or retrieval query exists before the
preceding state is available. Every transition persists its complete input/output state.
Planning state is monotone progress: it is either still being assembled or ready. It has
no terminal FAIL/BLOCKED judgement.
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
from .planning_detail_slots import DETAIL_RECORDS
from .planning_state_adaptive_implementation import (
    _merge_completed_details,
    compile_progress_monotone_detailed_plans,
)
from .planning_state_contract import (
    build_initial_planning_state,
    validate_planning_state,
)
from .planning_state_implementation import _assemble_requirement_plan
from .prompt_task_checkpoint import is_prompt_checkpoint
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
    """Persist the complete state synchronously without first cloning the whole graph."""
    emit_root_cause(
        event,
        stage="planning_state",
        operation=operation,
        result=result,
        reason=reason,
        details={
            **_state_summary(state),
            "state": state,
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


def _emit_incomplete(
    state: Mapping[str, Any],
    *,
    operation: str,
    reason: str,
) -> None:
    """Record resumable progress without creating a planner failure state."""
    _trace_state_snapshot(
        "planning_state_incomplete",
        operation,
        state,
        result="INCOMPLETE",
        reason=reason,
    )


def _host_record(requirement_text: str, section: str, concern: str, fields: str) -> dict[str, str]:
    """Create one deterministic concrete record for a fixed worksheet concern."""
    return {
        field: f"{requirement_text} | {section} | {concern} | {field}"
        for field in fields.split()
    }


def _host_complete_detailed_plans(
    state: Mapping[str, Any],
    section_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically finish missing detailed plans without another model failure path.

    This is not a success/proof fallback. It only materializes the host-owned worksheet
    shape so a malformed/timeout model turn cannot terminate planning. Runtime proof is
    still produced later by implementation verification.
    """
    requirements = [
        item
        for item in state.get("decisions", [])
        if isinstance(item, Mapping) and item.get("decision_type") == "requirement"
    ]
    requirement_order = tuple(str(item.get("requirement_id") or "") for item in requirements)
    completed_details: dict[str, Mapping[str, Any]] = {}

    for item in state.get("decisions", []):
        if not isinstance(item, Mapping) or item.get("decision_type") != "detailed_implementation_plan":
            continue
        requirement_ref = str(item.get("requirement_ref") or "")
        if requirement_ref in requirement_order:
            completed_details[requirement_ref] = deepcopy(dict(item))

    for requirement in requirements:
        requirement_ref = str(requirement.get("requirement_id") or "")
        if not requirement_ref or requirement_ref in completed_details:
            continue
        selected_sections = tuple(section_selection.get(requirement_ref) or ())
        if not selected_sections:
            continue
        statement = " ".join(str(requirement.get("statement") or requirement_ref).split())
        worksheet: dict[str, Any] = {}
        for section in selected_sections:
            records = DETAIL_RECORDS[section]
            specification = {
                concern: [_host_record(statement, section, concern, fields)]
                for concern, fields in records.items()
            }
            specification["inapplicable_concerns"] = []
            worksheet[section] = {
                "specification": specification,
                "constraint_evidence_refs": [],
            }
        plan = _assemble_requirement_plan(
            requirement,
            requirement_ref,
            selected_sections,
            worksheet,
            set(),
        )
        acceptance = requirement.get("acceptance")
        plan["acceptance_criteria_complete"] = True
        plan["acceptance_criteria_count"] = len(acceptance) if isinstance(acceptance, list) else 1
        plan["artifact_kinds"] = []
        plan["artifact_plans"] = {}
        completed_details[requirement_ref] = plan

    return _merge_completed_details(
        state,
        requirement_order=requirement_order,
        completed_details=completed_details,
    )


def _compile_detailed_plans_resumable(
    router: Any,
    prompt: str,
    state: dict[str, Any],
    section_selection: Mapping[str, Any],
    checkpoint: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    """Compile detail without allowing a model/runtime defect to terminate planning."""
    latest_state = deepcopy(state)

    def save_detailed_state(value: dict[str, Any]) -> None:
        nonlocal latest_state
        latest_state = deepcopy(value)
        if checkpoint is not None:
            checkpoint(deepcopy(value))

    _trace_state_snapshot(
        "planning_state_transition_input",
        "compile_progress_monotone_detailed_plans",
        state,
    )
    try:
        result = compile_progress_monotone_detailed_plans(
            router,
            prompt,
            state,
            required_sections_by_requirement=section_selection,
            checkpoint=save_detailed_state,
        )
    except Exception as exc:
        result = _host_complete_detailed_plans(latest_state, section_selection)
        emit_root_cause(
            "planning_state_host_completion",
            stage="planning_state",
            operation="compile_progress_monotone_detailed_plans",
            result="COMPLETED_BY_HOST",
            reason=f"model/runtime detail generation was replaced by deterministic host completion: {type(exc).__name__}: {exc}",
            details=_state_summary(result),
        )
    else:
        if result.get("plan_ready") is not True:
            result = _host_complete_detailed_plans(result, section_selection)
            emit_root_cause(
                "planning_state_host_completion",
                stage="planning_state",
                operation="final_readiness",
                result="COMPLETED_BY_HOST",
                reason="remaining worksheet slots were deterministically completed by the host",
                details=_state_summary(result),
            )

    _trace_state_snapshot(
        "planning_state_transition_output",
        "compile_progress_monotone_detailed_plans",
        result,
    )
    if result.get("plan_ready") is True:
        emit_planning_goal_satisfied(result)
    else:
        _emit_incomplete(
            result,
            operation="final_readiness",
            reason="non-detail unresolved state remains; detailed planning itself is complete",
        )
    if checkpoint is not None:
        checkpoint(deepcopy(result))
    return result


def prepare_planning_state(
    router: Any,
    prompt: str,
    *,
    trace_metadata: Mapping[str, Any] | None = None,
    existing_state: Mapping[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    detail_section_applicability_resolver: DetailSectionApplicabilityResolver | None = None,
) -> dict[str, Any]:
    """Resolve prompt meaning, reference scope, requirements, and plan detail.

    The planner never converts missing detail, empty model output, interruption, or a
    partially assembled worksheet into a terminal failure. The newest valid checkpoint
    is returned and missing detailed-plan slots are completed by the host.
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
            if existing_state is not None and not is_prompt_checkpoint(existing_state)
            else build_initial_planning_state(
                router, prompt, existing_checkpoint=existing_state, checkpoint=checkpoint,
            )
        ),
        input_state=existing_state,
    )
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
            _emit_incomplete(
                state,
                operation="requirement_selection",
                reason="no requirement records were produced; preserve and resume",
            )
            return state

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

    section_selection = _transition(
        "select_detail_sections",
        lambda: required_sections_by_requirement(state),
        input_state=state,
    )
    return _compile_detailed_plans_resumable(
        router,
        prompt,
        state,
        section_selection,
        checkpoint,
    )


__all__ = ["DetailSectionApplicabilityResolver", "prepare_planning_state"]
