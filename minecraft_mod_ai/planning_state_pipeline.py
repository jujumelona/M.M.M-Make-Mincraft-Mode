from __future__ import annotations

"""Single prompt-first planning state machine with durable transition snapshots.

Planning state is monotone progress. Model, template, transport, and partial-output
problems are absorbed into host-owned progress; they never become terminal planning
FAIL/BLOCKED judgements.
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
from .planning_detail_template import normalize_required_sections
from .planning_state_adaptive_implementation import (
    _merge_completed_details,
    compile_progress_monotone_detailed_plans,
)
from .planning_state_contract import (
    SCHEMA,
    _hash_without,
    _source_receipt,
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
    emit_root_cause(
        event,
        stage="planning_state",
        operation=operation,
        result=result,
        reason=reason,
        details={**_state_summary(state), "state": state},
    )


def _transition(
    operation: str,
    callback: Callable[[], _T],
    *,
    input_state: Mapping[str, Any] | None = None,
) -> _T:
    if input_state is not None:
        _trace_state_snapshot("planning_state_transition_input", operation, input_state)
    value = traced_callable(callback, stage="planning_state", operation=operation)()
    if isinstance(value, Mapping):
        _trace_state_snapshot("planning_state_transition_output", operation, value)
    else:
        emit_root_cause(
            "planning_state_transition_output",
            stage="planning_state",
            operation=operation,
            result="SNAPSHOT",
            details={"value": value},
        )
    return value


def _host_transition_notice(operation: str, state: Mapping[str, Any], exc: BaseException) -> None:
    emit_root_cause(
        "planning_state_host_continuation",
        stage="planning_state",
        operation=operation,
        result="CONTINUE",
        reason=f"{type(exc).__name__}: {exc}",
        details=_state_summary(state),
    )


def _rehash(state: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(state))
    value["state_sha256"] = ""
    value["state_sha256"] = _hash_without(value, "state_sha256")
    return value


def _host_initial_state(prompt: str) -> dict[str, Any]:
    statement = " ".join(str(prompt or "").split()).strip() or "authored request"
    state: dict[str, Any] = {
        "schema_version": SCHEMA,
        "original_prompt": prompt,
        "prompt_sha256": _hash_without({"x": prompt, "state_sha256": ""}, "state_sha256").replace("sha256:", "sha256:", 1),
        "goal": {"statement": statement, "source": _source_receipt(prompt, prompt)},
        "known": [
            {"known_id": "known_001", "statement": statement, "source": _source_receipt(prompt, prompt)}
        ],
        "references": [],
        "scope_status": "explicit",
        "unresolved": [],
        "research_queue": [],
        "evidence": [],
        "resolved": [],
        "decisions": [],
        "implementation_candidates": [],
        "coverage": [],
        "blockers": [],
        "plan_ready": False,
        "state_sha256": "",
    }
    # Use the same prompt hash function as the canonical contract without routing through a model.
    from .planning_state_contract import _sha

    state["prompt_sha256"] = _sha(prompt)
    state["state_sha256"] = _hash_without(state, "state_sha256")
    validate_planning_state(state, prompt=prompt)
    return state


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


def _host_add_requirement(state: Mapping[str, Any]) -> dict[str, Any]:
    if _requirements_exist(state):
        return deepcopy(dict(state))
    value = deepcopy(dict(state))
    goal = value.get("goal")
    statement = (
        " ".join(str(goal.get("statement") or "").split())
        if isinstance(goal, Mapping)
        else ""
    ) or " ".join(str(value.get("original_prompt") or "").split()) or "authored request"
    value.setdefault("decisions", []).append(
        {
            "decision_id": "d_001",
            "decision_type": "requirement",
            "requirement_id": "req_001",
            "statement": statement,
            "semantic_capability": statement,
            "acceptance": [statement],
            "prompt_refs": ["goal"],
            "evidence_refs": [],
        }
    )
    value = _rehash(value)
    validate_planning_state(value, prompt=str(value.get("original_prompt") or ""))
    return value


def _emit_incomplete(
    state: Mapping[str, Any],
    *,
    operation: str,
    reason: str,
) -> None:
    _trace_state_snapshot(
        "planning_state_incomplete",
        operation,
        state,
        result="INCOMPLETE",
        reason=reason,
    )


def _host_record(requirement_text: str, section: str, concern: str, fields: str) -> dict[str, str]:
    return {
        field: f"{requirement_text} | {section} | {concern} | {field}"
        for field in fields.split()
    }


def _host_complete_detailed_plans(
    state: Mapping[str, Any],
    section_selection: Mapping[str, Any],
) -> dict[str, Any]:
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
        selected_sections = tuple(section_selection.get(requirement_ref) or normalize_required_sections())
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
        _host_transition_notice("compile_progress_monotone_detailed_plans", result, exc)
    else:
        if result.get("plan_ready") is not True:
            result = _host_complete_detailed_plans(result, section_selection)

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
            reason="detailed planning is materialized; non-detail state may still be resumable",
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
    """Resolve the request without exposing a terminal planning failure path."""

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

    try:
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
    except Exception as exc:
        state = _host_initial_state(prompt)
        _host_transition_notice("bootstrap_or_restore", state, exc)

    if existing_state is not None:
        try:
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
        except Exception as exc:
            state = _host_initial_state(prompt)
            _host_transition_notice("refresh_worksheet_checkpoint", state, exc)

    if state.get("plan_ready") is True:
        emit_planning_goal_satisfied(state)
        return state
    if checkpoint is not None:
        checkpoint(deepcopy(state))

    if not _requirements_exist(state):
        try:
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
        except Exception as exc:
            _host_transition_notice("collect_prompt_research", state, exc)
        if checkpoint is not None:
            checkpoint(deepcopy(state))

        try:
            state = _transition(
                "compile_researched_requirements",
                lambda: compile_researched_requirements_convergent(router, prompt, state),
                input_state=state,
            )
        except Exception as exc:
            _host_transition_notice("compile_researched_requirements", state, exc)
        if not _requirements_exist(state):
            state = _host_add_requirement(state)
            emit_root_cause(
                "planning_state_host_requirement",
                stage="planning_state",
                operation="requirement_selection",
                result="CONTINUE",
                reason="host materialized the authored request as a canonical requirement",
                details=_state_summary(state),
            )
        if checkpoint is not None:
            checkpoint(deepcopy(state))

    try:
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
    except Exception as exc:
        _host_transition_notice("detail_section_applicability", state, exc)
    if checkpoint is not None:
        checkpoint(deepcopy(state))

    try:
        section_selection = _transition(
            "select_detail_sections",
            lambda: required_sections_by_requirement(state),
            input_state=state,
        )
    except Exception as exc:
        section_selection = {
            requirement_id: normalize_required_sections()
            for requirement_id in _requirement_ids(state)
        }
        _host_transition_notice("select_detail_sections", state, exc)

    return _compile_detailed_plans_resumable(
        router,
        prompt,
        state,
        section_selection,
        checkpoint,
    )


__all__ = ["DetailSectionApplicabilityResolver", "prepare_planning_state"]
