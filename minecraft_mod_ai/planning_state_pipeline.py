from __future__ import annotations

"""Single prompt-first planning state machine with durable transition snapshots.

Planning state advances monotonically through durable checkpoints. Model, template,
transport, and partial-output interruptions may resume only after a new validated
obligation marker is persisted. A detailed-planning attempt that makes no durable
progress, or returns non-ready without progress, fails closed instead of being promoted
or silently returned as a resumable success.
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
from .planning_detail_template import normalize_required_sections
from .planning_state_adaptive_implementation import (
    compile_progress_monotone_detailed_plans,
)
from .planning_state_contract import (
    SCHEMA,
    _hash_without,
    _source_receipt,
    build_initial_planning_state,
    validate_planning_state,
)
from .prompt_task_checkpoint import is_prompt_checkpoint
from .root_cause_trace import emit_root_cause

_T = TypeVar("_T")
DetailSectionApplicabilityResolver = Callable[
    [tuple[str, ...]],
    Mapping[str, Mapping[str, str]],
]
PlanningCheckpoint = Callable[[dict[str, Any]], None]
_DetailProgressPosition = tuple[
    frozenset[str],
    frozenset[tuple[str, int]],
    frozenset[tuple[str, str, str]],
    frozenset[str],
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


def _observe(event: str, **fields: Any) -> None:
    """Emit planning telemetry without allowing observability to affect control flow."""

    try:
        emit_root_cause(event, **fields)
    except Exception:
        pass


def _observe_goal_satisfied(state: Mapping[str, Any]) -> None:
    """Keep goal telemetry best-effort for the same reason as root-cause telemetry."""

    try:
        emit_planning_goal_satisfied(state)
    except Exception:
        pass


def _trace_state_snapshot(
    event: str,
    operation: str,
    state: Mapping[str, Any],
    *,
    result: str = "SNAPSHOT",
    reason: str = "",
) -> None:
    _observe(
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
    value = callback()
    if isinstance(value, Mapping):
        _trace_state_snapshot("planning_state_transition_output", operation, value)
    else:
        _observe(
            "planning_state_transition_output",
            stage="planning_state",
            operation=operation,
            result="SNAPSHOT",
            details={"value": value},
        )
    return value


def _host_transition_notice(operation: str, state: Mapping[str, Any], exc: BaseException) -> None:
    _observe(
        "planning_state_host_continuation",
        stage="planning_state",
        operation=operation,
        result="CONTINUE",
        reason=f"{type(exc).__name__}: {exc}",
        details=_state_summary(state),
    )


def _checkpoint_state(
    checkpoint: PlanningCheckpoint | None,
    state: Mapping[str, Any],
) -> None:
    if checkpoint is not None:
        checkpoint(deepcopy(dict(state)))


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
        "prompt_sha256": _hash_without(
            {"x": prompt, "state_sha256": ""}, "state_sha256"
        ).replace("sha256:", "sha256:", 1),
        "goal": {"statement": statement, "source": _source_receipt(prompt, prompt)},
        "known": [
            {
                "known_id": "known_001",
                "statement": statement,
                "source": _source_receipt(prompt, prompt),
            }
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
    from .planning_state_contract import _sha

    state["prompt_sha256"] = _sha(prompt)
    state["state_sha256"] = _hash_without(state, "state_sha256")
    validate_planning_state(state, prompt=prompt)
    return state


def _requirements_exist(state: Mapping[str, Any]) -> bool:
    decisions = state.get("decisions")
    if not isinstance(decisions, list):
        return False
    return any(
        isinstance(item, Mapping) and item.get("decision_type") == "requirement"
        for item in decisions
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


def _has_open_user_only_unknown(state: Mapping[str, Any]) -> bool:
    """Return True when planning is correctly waiting on information only the user can supply."""

    unresolved = state.get("unresolved")
    if not isinstance(unresolved, list):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("status") == "open"
        and item.get("resolution_route") == "user_only"
        for item in unresolved
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


def _research_stage_needed(state: Mapping[str, Any]) -> bool:
    """Enter research only when an explicit unresolved obligation needs it."""

    queue = state.get("research_queue")
    if isinstance(queue, list) and any(
        isinstance(row, Mapping) and row.get("status") == "pending"
        for row in queue
    ):
        return True

    unresolved = state.get("unresolved")
    return isinstance(unresolved, list) and any(
        isinstance(row, Mapping)
        and row.get("status") == "open"
        and row.get("resolution_route") == "default_policy"
        for row in unresolved
    )


def _collect_research_if_needed(
    router: Any,
    prompt: str,
    state: dict[str, Any],
    *,
    trace_metadata: Mapping[str, Any] | None,
    checkpoint: PlanningCheckpoint | None,
) -> dict[str, Any]:
    if not _research_stage_needed(state):
        return state
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
    _checkpoint_state(checkpoint, state)
    return state


def _resolve_requirements_or_wait(
    router: Any,
    prompt: str,
    state: dict[str, Any],
    *,
    trace_metadata: Mapping[str, Any] | None,
    checkpoint: PlanningCheckpoint | None,
) -> tuple[dict[str, Any], bool]:
    """Resolve authored requirements; return ``waiting=True`` only for user-only unknowns."""

    if _requirements_exist(state):
        return state, False

    state = _collect_research_if_needed(
        router,
        prompt,
        state,
        trace_metadata=trace_metadata,
        checkpoint=checkpoint,
    )

    try:
        state = _transition(
            "compile_researched_requirements",
            lambda: compile_researched_requirements_convergent(router, prompt, state),
            input_state=state,
        )
    except Exception as exc:
        _host_transition_notice("compile_researched_requirements", state, exc)

    if _requirements_exist(state):
        _checkpoint_state(checkpoint, state)
        return state, False

    if _has_open_user_only_unknown(state):
        _observe(
            "planning_state_waiting_for_user_input",
            stage="planning_state",
            operation="requirement_selection",
            result="RESUMABLE",
            reason="an open user-only unknown must be supplied by the user before requirement selection",
            details=_state_summary(state),
        )
        _checkpoint_state(checkpoint, state)
        return state, True

    state = _host_add_requirement(state)
    _observe(
        "planning_state_host_requirement",
        stage="planning_state",
        operation="requirement_selection",
        result="CONTINUE",
        reason="host materialized the authored request as a canonical requirement",
        details=_state_summary(state),
    )
    _checkpoint_state(checkpoint, state)
    return state, False


def _select_detail_sections(
    state: dict[str, Any],
    *,
    resolver: DetailSectionApplicabilityResolver | None,
    checkpoint: PlanningCheckpoint | None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    try:
        if resolver is None:
            state = _transition(
                "normalize_detail_section_applicability",
                lambda: ensure_host_detail_section_applicability(state),
                input_state=state,
            )
        else:
            applicability_by_requirement = _transition(
                "resolve_detail_section_applicability",
                lambda: resolver(_requirement_ids(state)),
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
    _checkpoint_state(checkpoint, state)

    try:
        selection = _transition(
            "select_detail_sections",
            lambda: required_sections_by_requirement(state),
            input_state=state,
        )
    except Exception as exc:
        selection = {
            requirement_id: normalize_required_sections()
            for requirement_id in _requirement_ids(state)
        }
        _host_transition_notice("select_detail_sections", state, exc)
    return state, selection


def _completed_detail_refs(state: Mapping[str, Any]) -> frozenset[str]:
    decisions = state.get("decisions")
    if not isinstance(decisions, list):
        return frozenset()
    return frozenset(
        str(item.get("requirement_ref") or "")
        for item in decisions
        if isinstance(item, Mapping)
        and item.get("decision_type") == "detailed_implementation_plan"
        and item.get("requirement_ref")
    )


def _criterion_progress_markers(
    state: Mapping[str, Any],
) -> frozenset[tuple[str, int]]:
    rows = state.get("detail_progress")
    if not isinstance(rows, list):
        return frozenset()
    return frozenset(
        (requirement_ref, criterion_index)
        for row in rows
        if isinstance(row, Mapping)
        for requirement_ref in (str(row.get("requirement_ref") or ""),)
        for criterion_index in (row.get("criterion_index"),)
        if requirement_ref and type(criterion_index) is int and criterion_index >= 0
    )


def _artifact_progress_markers(
    state: Mapping[str, Any],
) -> frozenset[tuple[str, str, str]]:
    progress = state.get("artifact_progress")
    if not isinstance(progress, Mapping):
        return frozenset()
    return frozenset(
        (str(requirement_ref), str(artifact_kind), str(step_id))
        for requirement_ref, by_kind in progress.items()
        if isinstance(by_kind, Mapping)
        for artifact_kind, by_step in by_kind.items()
        if isinstance(by_step, Mapping)
        for step_id in by_step
    )


def _template_progress_bindings(state: Mapping[str, Any]) -> frozenset[str]:
    progress = state.get("template_progress")
    if not isinstance(progress, Mapping):
        return frozenset()
    return frozenset(str(binding) for binding in progress)


def _detail_progress_position(state: Mapping[str, Any]) -> _DetailProgressPosition:
    """Return finite durable markers that prove strict resumable planning progress."""

    return (
        _completed_detail_refs(state),
        _criterion_progress_markers(state),
        _artifact_progress_markers(state),
        _template_progress_bindings(state),
    )


def _detail_progress_strictly_advanced(
    before: _DetailProgressPosition,
    after: _DetailProgressPosition,
) -> bool:
    before_details, before_criteria, before_artifacts, before_templates = before
    after_details, after_criteria, after_artifacts, after_templates = after

    if not before_details.issubset(after_details):
        return False
    if after_details != before_details:
        return True

    monotone_pairs = (
        (before_criteria, after_criteria),
        (before_artifacts, after_artifacts),
        (before_templates, after_templates),
    )
    if any(not old.issubset(new) for old, new in monotone_pairs):
        return False
    return any(old != new for old, new in monotone_pairs)


def _handle_nonready_detailed_result(
    result: dict[str, Any],
    progress_before: _DetailProgressPosition,
    checkpoint: PlanningCheckpoint | None,
) -> tuple[dict[str, Any], bool]:
    progress_after = _detail_progress_position(result)
    advanced = _detail_progress_strictly_advanced(progress_before, progress_after)
    if advanced:
        _observe(
            "detailed_planning_resume_after_progress",
            stage="planning_runtime", operation="compile_progress_monotone_detailed_plans",
            result="CONTINUE",
            reason="compiler produced durable progress; requeue remaining obligations",
            details={**_state_summary(result), "policy": "continue_while_durable_progress_advances"},
        )
    else:
        _observe(
            "detailed_planning_pending",
            stage="planning_runtime", operation="compile_progress_monotone_detailed_plans",
            result="RESUMABLE",
            reason="compiler returned a truthful non-ready state without new durable progress",
            details={**_state_summary(result), "policy": "persist_pending_state_without_synthetic_completion"},
        )
    _checkpoint_state(checkpoint, result)
    return (deepcopy(result) if advanced else result), advanced


def _clear_generation_interruption(result: dict[str, Any]) -> dict[str, Any]:
    if "generation_interruption" not in result:
        return result
    cleaned = dict(result)
    cleaned.pop("generation_interruption", None)
    return _rehash(cleaned)


def _compile_detailed_plans_resumable_impl(
    router: Any,
    prompt: str,
    state: dict[str, Any],
    section_selection: Mapping[str, Any],
    checkpoint: PlanningCheckpoint | None,
) -> dict[str, Any]:
    latest_state = deepcopy(state)

    def save_detailed_state(value: dict[str, Any]) -> None:
        nonlocal latest_state
        latest_state = deepcopy(value)
        _checkpoint_state(checkpoint, value)

    _trace_state_snapshot(
        "planning_state_transition_input",
        "compile_progress_monotone_detailed_plans",
        state,
    )

    while True:
        attempt_state = deepcopy(latest_state)
        progress_before = _detail_progress_position(attempt_state)
        try:
            result = compile_progress_monotone_detailed_plans(
                router,
                prompt,
                attempt_state,
                required_sections_by_requirement=section_selection,
                checkpoint=save_detailed_state,
            )
        except Exception as exc:
            progress_after = _detail_progress_position(latest_state)
            if _detail_progress_strictly_advanced(progress_before, progress_after):
                _host_transition_notice(
                    "compile_progress_monotone_detailed_plans",
                    latest_state,
                    exc,
                )
                _observe(
                    "detailed_planning_resume_from_checkpoint",
                    stage="planning_runtime",
                    operation="compile_progress_monotone_detailed_plans",
                    result="CONTINUE",
                    reason=f"{type(exc).__name__}: {exc}",
                    details={
                        **_state_summary(latest_state),
                        "policy": "resume_only_after_strict_durable_progress",
                    },
                )
                continue

            pending_state = deepcopy(latest_state)
            pending_state["plan_ready"] = False
            pending_state["generation_interruption"] = {
                "type": type(exc).__name__,
                "reason": str(exc),
            }
            _observe(
                "detailed_planning_pending",
                stage="planning_runtime",
                operation="compile_progress_monotone_detailed_plans",
                result="RESUMABLE",
                reason=str(exc),
                details={
                    **_state_summary(pending_state),
                    "policy": "persist_recoverable_interruption_without_synthetic_completion",
                },
            )
            _checkpoint_state(checkpoint, pending_state)
            return pending_state

        if result.get("plan_ready") is not True:
            latest_state, advanced = _handle_nonready_detailed_result(
                result, progress_before, checkpoint
            )
            if not advanced:
                return latest_state
            continue
        break

    result = _clear_generation_interruption(result)
    _trace_state_snapshot(
        "planning_state_transition_output",
        "compile_progress_monotone_detailed_plans",
        result,
    )
    _observe_goal_satisfied(result)
    _checkpoint_state(checkpoint, result)
    return result


def _compile_detailed_plans_resumable(
    router: Any,
    prompt: str,
    state: dict[str, Any],
    section_selection: Mapping[str, Any],
    checkpoint: PlanningCheckpoint | None,
) -> dict[str, Any]:
    return _compile_detailed_plans_resumable_impl(
        router, prompt, state, section_selection, checkpoint
    )


def prepare_planning_state(
    router: Any,
    prompt: str,
    *,
    trace_metadata: Mapping[str, Any] | None = None,
    existing_state: Mapping[str, Any] | None = None,
    checkpoint: PlanningCheckpoint | None = None,
    detail_section_applicability_resolver: DetailSectionApplicabilityResolver | None = None,
) -> dict[str, Any]:
    """Resolve the request without promoting an incomplete detailed plan."""

    _observe(
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
                    router,
                    prompt,
                    existing_checkpoint=existing_state,
                    checkpoint=checkpoint,
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
        _observe_goal_satisfied(state)
        return state
    _checkpoint_state(checkpoint, state)

    state, waiting_for_user = _resolve_requirements_or_wait(
        router,
        prompt,
        state,
        trace_metadata=trace_metadata,
        checkpoint=checkpoint,
    )
    if waiting_for_user:
        return state

    state, section_selection = _select_detail_sections(
        state,
        resolver=detail_section_applicability_resolver,
        checkpoint=checkpoint,
    )
    return _compile_detailed_plans_resumable(
        router,
        prompt,
        state,
        section_selection,
        checkpoint,
    )


__all__ = ["DetailSectionApplicabilityResolver", "prepare_planning_state"]
