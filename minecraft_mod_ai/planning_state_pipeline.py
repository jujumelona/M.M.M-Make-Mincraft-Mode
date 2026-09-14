from __future__ import annotations

"""Single prompt-first planning state machine with durable transition snapshots.

Planning state is monotone progress. Model, template, transport, and partial-output
problems are absorbed into host-owned progress; they never become terminal planning
FAIL/BLOCKED judgements. Detailed plans are promoted only from validated authored
checkpoints; the host never fabricates missing engineering detail.
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
from .planning_state_adaptive_implementation import compile_progress_monotone_detailed_plans
from .planning_state_contract import (
    SCHEMA,
    _hash_without,
    _source_receipt,
    build_initial_planning_state,
    validate_planning_state,
)
from .prompt_task_checkpoint import is_prompt_checkpoint
from .root_cause_trace import emit_root_cause, traced_callable

_T = TypeVar("_T")
DetailSectionApplicabilityResolver = Callable[
    [tuple[str, ...]],
    Mapping[str, Mapping[str, str]],
]
PlanningCheckpoint = Callable[[dict[str, Any]], None]
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
    return any(
        isinstance(item, Mapping) and item.get("decision_type") == "requirement"
        for item in decisions
        if isinstance(decisions, list)
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
        emit_root_cause(
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
    emit_root_cause(
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


def _detail_progress_position(state: Mapping[str, Any]) -> tuple[
    frozenset[str],
    frozenset[tuple[str, int]],
    frozenset[tuple[str, str, str]],
    frozenset[str],
]:
    """Return durable completed-obligation markers used to prove resumable progress.

    The marker sets are finite because every key belongs to a host-declared requirement,
    acceptance criterion, artifact responsibility, or fixed-template binding. Content
    changes alone never count as progress.
    """

    completed_details = frozenset(
        str(item.get("requirement_ref") or "")
        for item in state.get("decisions", [])
        if isinstance(item, Mapping)
        and item.get("decision_type") == "detailed_implementation_plan"
        and item.get("requirement_ref")
    )

    criterion_markers: set[tuple[str, int]] = set()
    detail_progress = state.get("detail_progress", []) or []
    if isinstance(detail_progress, list):
        for row in detail_progress:
            if not isinstance(row, Mapping):
                continue
            requirement_ref = str(row.get("requirement_ref") or "")
            criterion_index = row.get("criterion_index")
            if requirement_ref and type(criterion_index) is int and criterion_index >= 0:
                criterion_markers.add((requirement_ref, criterion_index))

    artifact_markers: set[tuple[str, str, str]] = set()
    artifact_progress = state.get("artifact_progress", {}) or {}
    if isinstance(artifact_progress, Mapping):
        for requirement_ref, by_kind in artifact_progress.items():
            if not isinstance(by_kind, Mapping):
                continue
            for artifact_kind, by_step in by_kind.items():
                if not isinstance(by_step, Mapping):
                    continue
                for step_id in by_step:
                    artifact_markers.add(
                        (str(requirement_ref), str(artifact_kind), str(step_id))
                    )

    template_progress = state.get("template_progress", {}) or {}
    template_bindings = frozenset(
        str(binding)
        for binding in template_progress
    ) if isinstance(template_progress, Mapping) else frozenset()

    return (
        completed_details,
        frozenset(criterion_markers),
        frozenset(artifact_markers),
        template_bindings,
    )


def _detail_progress_strictly_advanced(
    before: tuple[
        frozenset[str],
        frozenset[tuple[str, int]],
        frozenset[tuple[str, str, str]],
        frozenset[str],
    ],
    after: tuple[
        frozenset[str],
        frozenset[tuple[str, int]],
        frozenset[tuple[str, str, str]],
        frozenset[str],
    ],
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


def _compile_detailed_plans_resumable(
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
                emit_root_cause(
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

            emit_root_cause(
                "detailed_planning_runtime_stalled",
                stage="planning_runtime",
                operation="compile_progress_monotone_detailed_plans",
                result="ERROR",
                reason=f"{type(exc).__name__}: {exc}",
                details={
                    **_state_summary(latest_state),
                    "policy": "runtime_defect_not_synthetic_plan",
                },
            )
            raise RuntimeError(
                "DETAILED_PLAN_RUNTIME_STALLED: detailed planning raised again without "
                "completing any new durable obligation; refusing to synthesize plan_ready"
            ) from exc

        if result.get("plan_ready") is not True:
            _trace_state_snapshot(
                "detailed_planning_invariant_violation",
                "compile_progress_monotone_detailed_plans",
                result,
                result="ERROR",
                reason="compiler returned without a fully ready validated plan",
            )
            raise RuntimeError(
                "DETAILED_PLAN_NOT_READY: detailed planner returned before every validated "
                "requirement was complete; refusing host-authored placeholder completion"
            )
        break

    _trace_state_snapshot(
        "planning_state_transition_output",
        "compile_progress_monotone_detailed_plans",
        result,
    )
    emit_planning_goal_satisfied(result)
    _checkpoint_state(checkpoint, result)
    return result


def prepare_planning_state(
    router: Any,
    prompt: str,
    *,
    trace_metadata: Mapping[str, Any] | None = None,
    existing_state: Mapping[str, Any] | None = None,
    checkpoint: PlanningCheckpoint | None = None,
    detail_section_applicability_resolver: DetailSectionApplicabilityResolver | None = None,
) -> dict[str, Any]:
    """Resolve the request without exposing a terminal planning failure state."""

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
        emit_planning_goal_satisfied(state)
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
