from __future__ import annotations

"""Progress-monotone detailed planning for small local models.

The host schedules a finite DAG of requirement/section contracts. Each section is first
attempted as one bounded structured output. Only a failed section is decomposed into the
existing atomic concern chunks. Completed requirement plans are merged into the canonical
planning state and checkpointed immediately, so resume never regenerates accepted work.
"""

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
import json
from typing import Any

from .model_concurrency import router_native_model_parallelism
from .planner_operation import planner_operation
from .planning_detail_template import (
    WORKSHEET_SECTIONS,
    normalize_required_sections,
    validate_worksheet_section,
    worksheet_section_schema,
)
from .planning_state_contract import validate_planning_state
from .planning_state_implementation import (
    _assemble_requirement_plan,
    _compile_worksheet_section,
    _requirement_decisions,
    _requirement_grounding,
    _section_dependencies,
    _section_messages,
)
from .root_cause_trace import emit_root_cause


Checkpoint = Callable[[dict[str, Any]], None]


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _generate_whole_section(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    selected_sections: tuple[str, ...],
    section: str,
    evidence: list[Mapping[str, Any]],
    allowed: set[str],
    completed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Try one complete structured section before any decomposition."""
    messages = _section_messages(
        requirement,
        selected_sections,
        section,
        evidence,
        completed,
    )
    schema = worksheet_section_schema(section)
    tool_name = f"submit_{section}_section"
    description = f"Submit the complete {section} engineering worksheet section."

    decoded: Mapping[str, Any] | None = None
    with planner_operation(f"detailed_section:{section}:whole"):
        if hasattr(router, "generate_tool_decision"):
            try:
                raw_decision = router.generate_tool_decision(
                    "planner",
                    messages,
                    tool_name=tool_name,
                    parameters=schema,
                    description=description,
                )
                if isinstance(raw_decision, Mapping):
                    decoded = raw_decision
            except Exception as exc:
                from .model_adapters import ModelConfigurationError

                if isinstance(exc, ModelConfigurationError):
                    raise
                emit_root_cause(
                    "detailed_section_whole_tool_fallback",
                    stage="planning_state",
                    operation=f"detailed_section:{section}:whole",
                    result="FALLBACK",
                    reason=f"{type(exc).__name__}: {exc}",
                    details={"section": section, "fallback": "structured_text"},
                )

        if decoded is None:
            raw = router.generate_text(
                "planner",
                messages,
                response_format="json",
                response_schema=schema,
                enable_tools=False,
            )
            parsed = json.loads(raw)
            if not isinstance(parsed, Mapping):
                raise ValueError("whole section output must be a JSON object")
            decoded = parsed

    return validate_worksheet_section(dict(decoded), allowed, section)


def _compile_section_adaptive(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    selected_sections: tuple[str, ...],
    section: str,
    evidence: list[Mapping[str, Any]],
    allowed: set[str],
    completed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Use failure-triggered decomposition instead of unconditional decomposition."""
    try:
        result = _generate_whole_section(
            router,
            requirement=requirement,
            selected_sections=selected_sections,
            section=section,
            evidence=evidence,
            allowed=allowed,
            completed=completed,
        )
    except Exception as exc:
        from .model_adapters import ModelConfigurationError

        if isinstance(exc, ModelConfigurationError):
            raise
        emit_root_cause(
            "detailed_section_adaptive_decomposition",
            stage="planning_state",
            operation=f"detailed_section:{section}",
            result="FALLBACK",
            reason=f"{type(exc).__name__}: {exc}",
            details={
                "requirement_ref": _text(requirement.get("requirement_id")),
                "section": section,
                "strategy": "whole_then_atomic_chunks",
            },
        )
        result = _compile_worksheet_section(
            router,
            requirement=requirement,
            selected_sections=selected_sections,
            section=section,
            evidence=evidence,
            allowed=allowed,
            completed=completed,
        )

    return result


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


def compile_progress_monotone_detailed_plans(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
    *,
    required_sections_by_requirement: Mapping[str, Iterable[str]],
    checkpoint: Checkpoint | None = None,
) -> dict[str, Any]:
    """Compile only unfinished contracts and persist each completed requirement.

    Termination is structural rather than call-count based: every successful future
    removes exactly one node from a finite pending DAG. A failure either decomposes that
    node once into the existing finite chunk protocol or propagates. If pending nodes
    exist but none are runnable, the function raises an explicit deadlock.
    """
    validate_planning_state(state, prompt=prompt)
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
    working_state = _merge_completed_details(
        state,
        requirement_order=requirement_order,
        completed_details=completed_details,
    )
    if checkpoint is not None and len(completed_details) != len(existing):
        checkpoint(deepcopy(working_state))
    if working_state.get("plan_ready") is True:
        return working_state

    section_rank = {section: index for index, section in enumerate(WORKSHEET_SECTIONS)}
    jobs: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_ref = _text(requirement.get("requirement_id"))
        if requirement_ref in completed_details:
            continue
        evidence, allowed = _requirement_grounding(working_state, requirement_ref)
        selected_sections = selections[requirement_ref]
        jobs.append(
            {
                "requirement": requirement,
                "requirement_ref": requirement_ref,
                "selected_sections": selected_sections,
                "evidence": evidence,
                "allowed": allowed,
                "completed": {},
                "pending": set(selected_sections),
                "submitted": set(),
            }
        )

    def ready_nodes() -> list[tuple[int, str]]:
        ready: list[tuple[int, str]] = []
        for job_index, job in enumerate(jobs):
            selected_sections = job["selected_sections"]
            completed = job["completed"]
            for section in selected_sections:
                if section not in job["pending"] or section in job["submitted"]:
                    continue
                dependencies = _section_dependencies(section, selected_sections)
                if all(dependency in completed for dependency in dependencies):
                    ready.append((job_index, section))
        return sorted(ready, key=lambda item: (section_rank[item[1]], item[0]))

    initial_pending_nodes = sum(len(job["pending"]) for job in jobs)
    remaining_nodes = initial_pending_nodes
    workers = max(1, min(remaining_nodes or 1, router_native_model_parallelism(router)))
    future_to_node: dict[Future[dict[str, Any]], tuple[int, str]] = {}

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="planning-adaptive-detail",
    ) as pool:
        while remaining_nodes or future_to_node:
            for job_index, section in ready_nodes():
                if len(future_to_node) >= workers:
                    break
                job = jobs[job_index]
                dependencies = _section_dependencies(section, job["selected_sections"])
                prerequisite_snapshot = {
                    dependency: deepcopy(job["completed"][dependency])
                    for dependency in dependencies
                }
                future = pool.submit(
                    _compile_section_adaptive,
                    router,
                    requirement=job["requirement"],
                    selected_sections=job["selected_sections"],
                    section=section,
                    evidence=job["evidence"],
                    allowed=job["allowed"],
                    completed=prerequisite_snapshot,
                )
                job["submitted"].add(section)
                future_to_node[future] = (job_index, section)

            if not future_to_node:
                pending = {
                    job["requirement_ref"]: sorted(
                        job["pending"], key=section_rank.__getitem__
                    )
                    for job in jobs
                    if job["pending"]
                }
                raise RuntimeError(
                    "DETAILED_PLAN_DAG_DEADLOCK: unsatisfied contracts remain but "
                    "no runnable section exists; " + repr(pending)
                )

            done, _ = wait(tuple(future_to_node), return_when=FIRST_COMPLETED)
            completed_futures = sorted(
                done,
                key=lambda future: (
                    section_rank[future_to_node[future][1]],
                    future_to_node[future][0],
                ),
            )
            for future in completed_futures:
                job_index, section = future_to_node.pop(future)
                job = jobs[job_index]
                result = future.result()
                if section not in job["pending"]:
                    raise RuntimeError(
                        "DETAILED_PLAN_PROGRESS_INVARIANT: completed node was not pending"
                    )
                before = remaining_nodes
                job["completed"][section] = result
                job["pending"].remove(section)
                job["submitted"].remove(section)
                remaining_nodes -= 1
                if remaining_nodes >= before:
                    raise RuntimeError(
                        "DETAILED_PLAN_NO_PROGRESS: pending work did not strictly decrease"
                    )

                if not job["pending"]:
                    worksheet = {
                        key: job["completed"][key]
                        for key in job["selected_sections"]
                    }
                    plan = _assemble_requirement_plan(
                        job["requirement"],
                        job["requirement_ref"],
                        job["selected_sections"],
                        worksheet,
                        job["allowed"],
                    )
                    completed_details[job["requirement_ref"]] = plan
                    working_state = _merge_completed_details(
                        working_state,
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
                            "completed_requirements": len(completed_details),
                            "total_requirements": len(requirement_order),
                            "remaining_section_nodes": remaining_nodes,
                            "initial_section_nodes": initial_pending_nodes,
                        },
                    )
                    if checkpoint is not None:
                        checkpoint(deepcopy(working_state))

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
