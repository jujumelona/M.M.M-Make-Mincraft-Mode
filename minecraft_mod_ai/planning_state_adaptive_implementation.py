from __future__ import annotations

"""Acceptance-criterion, progress-monotone detailed planning for small local models.

The host never asks the model to emit a complete worksheet or worksheet section. Each
already-approved public acceptance criterion is one bounded model contract. Criterion
fragments are checkpointed individually, merged deterministically into the canonical
worksheet, and never regenerated after resume. There is no count-driven retry loop.
"""

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from copy import deepcopy
from typing import Any

from .model_concurrency import router_native_model_parallelism
from .planner_operation import planner_operation
from .planning_criterion_fragments import (
    assemble_worksheet_from_fragments,
    clear_requirement_progress,
    generate_criterion_fragment,
    load_requirement_progress,
    requirement_acceptance_criteria,
    store_criterion_progress,
)
from .planning_detail_template import WORKSHEET_SECTIONS, normalize_required_sections
from .planning_state_contract import validate_planning_state
from .planning_state_implementation import (
    _assemble_requirement_plan,
    _requirement_decisions,
    _requirement_grounding,
)
from .root_cause_trace import emit_root_cause


Checkpoint = Callable[[dict[str, Any]], None]
_TERMINAL_DETAIL_STAGE = "detailed_planning"


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _existing_detail_map(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for item in state.get("decisions", []):
        if (
            isinstance(item, Mapping)
            and item.get("decision_type") == "detailed_implementation_plan"
        ):
            requirement_ref = _text(item.get("requirement_ref"))
            if requirement_ref:
                result[requirement_ref] = item
    return result


def _detail_matches_selection(
    detail: Mapping[str, Any],
    selected_sections: tuple[str, ...],
) -> bool:
    raw = detail.get("required_detail_sections")
    return isinstance(raw, list) and tuple(raw) == selected_sections


def _merge_completed_details(
    state: Mapping[str, Any],
    *,
    requirement_order: tuple[str, ...],
    completed_details: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = deepcopy(dict(state))
    non_detail = [
        item
        for item in value.get("decisions", [])
        if not (
            isinstance(item, Mapping)
            and item.get("decision_type") == "detailed_implementation_plan"
        )
    ]
    details: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for index, requirement_ref in enumerate(requirement_order, start=1):
        raw = completed_details.get(requirement_ref)
        if raw is None:
            continue
        detail = deepcopy(dict(raw))
        decision_id = f"detail_{index:03d}"
        detail["decision_id"] = decision_id
        detail["decision_type"] = "detailed_implementation_plan"
        detail["requirement_ref"] = requirement_ref
        details.append(detail)
        coverage.append(
            {
                "requirement_ref": requirement_ref,
                "status": "covered",
                "detailed_plan_ref": decision_id,
            }
        )

    value["decisions"] = non_detail + details
    value["coverage"] = coverage
    active_unknowns = [
        item
        for item in value.get("unresolved", [])
        if isinstance(item, Mapping) and item.get("status") != "resolved"
    ]
    active_blockers = [
        item for item in value.get("blockers", []) if isinstance(item, Mapping)
    ]
    value["plan_ready"] = (
        len(completed_details) == len(requirement_order)
        and not active_unknowns
        and not active_blockers
    )
    result = _rehash(value)
    validate_planning_state(result)
    return result


def _terminal_detail_blockers(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in state.get("blockers", [])
        if isinstance(row, Mapping)
        and row.get("stage") == _TERMINAL_DETAIL_STAGE
        and row.get("terminal") is True
    ]


def _checkpoint_state(
    state: Mapping[str, Any],
    checkpoint: Checkpoint | None,
) -> dict[str, Any]:
    value = _rehash(deepcopy(dict(state)))
    validate_planning_state(value)
    if checkpoint is not None:
        checkpoint(deepcopy(value))
    return value


def _checkpoint_terminal_blocker(
    state: Mapping[str, Any],
    *,
    requirement_ref: str,
    work_unit: str,
    reason: str,
    checkpoint: Checkpoint | None,
) -> dict[str, Any]:
    value = deepcopy(dict(state))
    blockers = value.setdefault("blockers", [])
    blockers.append(
        {
            "blocker_id": f"b_{len(blockers) + 1:03d}",
            "stage": _TERMINAL_DETAIL_STAGE,
            "terminal": True,
            "requirement_ref": requirement_ref,
            "section": work_unit,
            "statement": reason,
        }
    )
    value["plan_ready"] = False
    value = _checkpoint_state(value, checkpoint)
    emit_root_cause(
        "detailed_planning_terminal_blocker",
        stage="planning_state",
        operation="compile_progress_monotone_detailed_plans",
        result="FAIL",
        reason=reason,
        details={
            "requirement_ref": requirement_ref,
            "work_unit": work_unit,
            "terminal": True,
        },
    )
    return value


def _compile_criterion(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    requirement_ref: str,
    criterion_index: int,
    criterion: str,
    selected_sections: tuple[str, ...],
    evidence: list[Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[str, Any]:
    with planner_operation(
        f"detailed_criterion:{requirement_ref}:{criterion_index + 1}"
    ):
        return generate_criterion_fragment(
            router,
            requirement=requirement,
            criterion=criterion,
            selected_sections=selected_sections,
            evidence=evidence,
            allowed_refs=allowed_refs,
        )


def _finish_requirement(
    job: dict[str, Any],
    *,
    working_state: Mapping[str, Any],
    requirement_order: tuple[str, ...],
    completed_details: dict[str, Mapping[str, Any]],
    checkpoint: Checkpoint | None,
) -> dict[str, Any]:
    worksheet = assemble_worksheet_from_fragments(
        job["requirement"],
        selected_sections=job["selected_sections"],
        criteria=job["criteria"],
        fragments=job["fragments"],
        allowed_refs=job["allowed"],
    )
    plan = _assemble_requirement_plan(
        job["requirement"],
        job["requirement_ref"],
        job["selected_sections"],
        worksheet,
        job["allowed"],
    )
    completed_details[job["requirement_ref"]] = plan
    cleared = clear_requirement_progress(working_state, job["requirement_ref"])
    result = _merge_completed_details(
        cleared,
        requirement_order=requirement_order,
        completed_details=completed_details,
    )
    emit_root_cause(
        "detailed_requirement_checkpoint",
        stage="planning_state",
        operation="compile_progress_monotone_detailed_plans",
        result="PASS",
        details={
            "requirement_ref": job["requirement_ref"],
            "completed_acceptance_criteria": len(job["criteria"]),
            "completed_requirements": len(completed_details),
            "total_requirements": len(requirement_order),
            "strategy": "acceptance_criterion_fragments_to_host_worksheet",
        },
    )
    if checkpoint is not None:
        checkpoint(deepcopy(result))
    return result


def compile_progress_monotone_detailed_plans(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
    *,
    required_sections_by_requirement: Mapping[str, Iterable[str]],
    checkpoint: Checkpoint | None = None,
) -> dict[str, Any]:
    """Compile one bounded contract per unfinished public acceptance criterion.

    The ranking function is the number of unfinished canonical acceptance criteria. Every
    successful model call removes exactly one element and checkpoints it. Failed work is
    terminal rather than re-enqueued. Completed criteria and requirements are restored from
    checkpoints without model calls. The only concurrency is among independent unfinished
    criteria, bounded by the router's native model parallelism.
    """

    validate_planning_state(state, prompt=prompt)
    prior_terminal = _terminal_detail_blockers(state)
    if prior_terminal:
        raise RuntimeError(
            "DETAILED_PLAN_BLOCKED: terminal detailed-planning blocker already checkpointed; "
            + "; ".join(_text(row.get("statement")) for row in prior_terminal)
        )

    requirements = _requirement_decisions(state)
    if not requirements:
        raise ValueError("DETAILED_PLAN_REQUIREMENTS: no researched requirements exist")
    if not isinstance(required_sections_by_requirement, Mapping):
        raise ValueError("DETAILED_PLAN_SECTIONS: host selection must be a requirement mapping")

    requirement_order = tuple(_text(row.get("requirement_id")) for row in requirements)
    if any(not requirement_ref for requirement_ref in requirement_order):
        raise ValueError("DETAILED_PLAN_REQUIREMENTS: requirement IDs must be non-empty")
    if len(set(requirement_order)) != len(requirement_order):
        raise ValueError("DETAILED_PLAN_REQUIREMENTS: requirement IDs must be unique")

    unknown_selection = set(str(key) for key in required_sections_by_requirement) - set(
        requirement_order
    )
    if unknown_selection:
        raise ValueError(
            "DETAILED_PLAN_SECTIONS: selection cites unknown requirement(s): "
            + ", ".join(sorted(unknown_selection))
        )

    selections = {
        requirement_ref: normalize_required_sections(
            required_sections_by_requirement.get(requirement_ref)
        )
        for requirement_ref in requirement_order
    }
    existing = _existing_detail_map(state)
    completed_details: dict[str, Mapping[str, Any]] = {
        requirement_ref: detail
        for requirement_ref, detail in existing.items()
        if requirement_ref in selections
        and _detail_matches_selection(detail, selections[requirement_ref])
    }

    working_state: dict[str, Any] = deepcopy(dict(state))
    for requirement_ref in completed_details:
        working_state = clear_requirement_progress(working_state, requirement_ref)
    working_state = _merge_completed_details(
        working_state,
        requirement_order=requirement_order,
        completed_details=completed_details,
    )
    if checkpoint is not None and len(completed_details) != len(existing):
        checkpoint(deepcopy(working_state))
    if working_state.get("plan_ready") is True:
        return working_state

    jobs: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_ref = _text(requirement.get("requirement_id"))
        if requirement_ref in completed_details:
            continue
        evidence, allowed = _requirement_grounding(working_state, requirement_ref)
        selected_sections = selections[requirement_ref]
        criteria = requirement_acceptance_criteria(requirement)
        fragments = load_requirement_progress(
            working_state,
            requirement_ref=requirement_ref,
            selected_sections=selected_sections,
            criteria=criteria,
            allowed_refs=allowed,
        )
        jobs.append(
            {
                "requirement": requirement,
                "requirement_ref": requirement_ref,
                "selected_sections": selected_sections,
                "evidence": evidence,
                "allowed": allowed,
                "criteria": criteria,
                "fragments": fragments,
            }
        )

    # A checkpoint can contain every criterion for a requirement but not yet the assembled
    # detail if the process died between those two host operations. Finish such work without
    # another model call before scheduling anything new.
    for job in jobs:
        if len(job["fragments"]) == len(job["criteria"]):
            working_state = _finish_requirement(
                job,
                working_state=working_state,
                requirement_order=requirement_order,
                completed_details=completed_details,
                checkpoint=checkpoint,
            )

    pending = deque(
        (job_index, criterion_index)
        for job_index, job in enumerate(jobs)
        if job["requirement_ref"] not in completed_details
        for criterion_index in range(len(job["criteria"]))
        if criterion_index not in job["fragments"]
    )
    remaining = len(pending)
    if remaining:
        workers = max(1, min(remaining, router_native_model_parallelism(router)))
        future_to_node: dict[Future[dict[str, Any]], tuple[int, int]] = {}

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="planning-acceptance-criterion",
        ) as pool:
            while pending or future_to_node:
                while pending and len(future_to_node) < workers:
                    job_index, criterion_index = pending.popleft()
                    job = jobs[job_index]
                    criterion = job["criteria"][criterion_index]
                    future = pool.submit(
                        copy_context().run,
                        _compile_criterion,
                        router,
                        requirement=job["requirement"],
                        requirement_ref=job["requirement_ref"],
                        criterion_index=criterion_index,
                        criterion=criterion,
                        selected_sections=job["selected_sections"],
                        evidence=job["evidence"],
                        allowed_refs=job["allowed"],
                    )
                    future_to_node[future] = (job_index, criterion_index)

                if not future_to_node:
                    reason = (
                        "DETAILED_PLAN_DAG_DEADLOCK: unfinished acceptance criteria remain "
                        "but no runnable criterion exists"
                    )
                    job_index, criterion_index = pending[0]
                    job = jobs[job_index]
                    working_state = _checkpoint_terminal_blocker(
                        working_state,
                        requirement_ref=job["requirement_ref"],
                        work_unit=f"acceptance_criterion:{criterion_index + 1}",
                        reason=reason,
                        checkpoint=checkpoint,
                    )
                    raise RuntimeError(reason)

                done, _ = wait(tuple(future_to_node), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: future_to_node[item]):
                    job_index, criterion_index = future_to_node.pop(future)
                    job = jobs[job_index]
                    criterion = job["criteria"][criterion_index]
                    try:
                        fragment = future.result()
                    except Exception as exc:
                        for in_flight in future_to_node:
                            in_flight.cancel()
                        reason = (
                            "DETAILED_PLAN_BLOCKED: atomic acceptance criterion failed without "
                            f"valid progress for {job['requirement_ref']}/criterion_{criterion_index + 1}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        working_state = _checkpoint_terminal_blocker(
                            working_state,
                            requirement_ref=job["requirement_ref"],
                            work_unit=f"acceptance_criterion:{criterion_index + 1}",
                            reason=reason,
                            checkpoint=checkpoint,
                        )
                        raise RuntimeError(reason) from exc

                    if criterion_index in job["fragments"]:
                        raise RuntimeError(
                            "DETAILED_PLAN_PROGRESS_INVARIANT: completed criterion was already checkpointed"
                        )
                    before = remaining
                    job["fragments"][criterion_index] = fragment
                    remaining -= 1
                    if remaining >= before:
                        raise RuntimeError(
                            "DETAILED_PLAN_NO_PROGRESS: unfinished acceptance criteria did not strictly decrease"
                        )

                    working_state = store_criterion_progress(
                        working_state,
                        requirement_ref=job["requirement_ref"],
                        selected_sections=job["selected_sections"],
                        criterion_index=criterion_index,
                        criterion=criterion,
                        fragment=fragment,
                    )
                    working_state = _checkpoint_state(working_state, checkpoint)
                    emit_root_cause(
                        "detailed_acceptance_criterion_checkpoint",
                        stage="planning_state",
                        operation="compile_progress_monotone_detailed_plans",
                        result="PASS",
                        details={
                            "requirement_ref": job["requirement_ref"],
                            "criterion_index": criterion_index + 1,
                            "criterion_count": len(job["criteria"]),
                            "remaining_acceptance_criteria": remaining,
                        },
                    )

                    if len(job["fragments"]) == len(job["criteria"]):
                        working_state = _finish_requirement(
                            job,
                            working_state=working_state,
                            requirement_order=requirement_order,
                            completed_details=completed_details,
                            checkpoint=checkpoint,
                        )

    if len(completed_details) != len(requirement_order):
        raise RuntimeError(
            "DETAILED_PLAN_NO_PROGRESS: scheduler exited with unfinished requirements"
        )

    final_state = _merge_completed_details(
        working_state,
        requirement_order=requirement_order,
        completed_details=completed_details,
    )
    validate_planning_state(final_state, prompt=prompt)
    return final_state


__all__ = ["compile_progress_monotone_detailed_plans"]