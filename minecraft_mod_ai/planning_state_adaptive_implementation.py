from __future__ import annotations

"""Requirement-first, progress-monotone detailed planning for small local models.

Each unfinished requirement gets one bounded whole-worksheet attempt first. Only a failed
requirement is decomposed, and individually valid dependency-closed sections from that
attempt are retained in memory so only missing/invalid sections are regenerated. Completed
requirements are merged into the canonical state and checkpointed immediately.
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
    validate_worksheet,
    validate_worksheet_section,
    worksheet_prompt,
    worksheet_schema,
    worksheet_section_schema,
)
from .planning_state_contract import validate_planning_state
from .planning_state_implementation import (
    _assemble_requirement_plan,
    _compile_worksheet_section,
    _evidence_context,
    _requirement_decisions,
    _requirement_grounding,
    _section_dependencies,
    _section_messages,
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


def _checkpoint_terminal_blocker(
    state: Mapping[str, Any],
    *,
    requirement_ref: str,
    section: str,
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
            "section": section,
            "statement": reason,
        }
    )
    value["plan_ready"] = False
    value = _rehash(value)
    validate_planning_state(value)
    emit_root_cause(
        "detailed_planning_terminal_blocker",
        stage="planning_state",
        operation="compile_progress_monotone_detailed_plans",
        result="FAIL",
        reason=reason,
        details={
            "requirement_ref": requirement_ref,
            "section": section,
            "terminal": True,
        },
    )
    if checkpoint is not None:
        checkpoint(deepcopy(value))
    return value


def _whole_requirement_messages(
    requirement: Mapping[str, Any],
    selected_sections: tuple[str, ...],
    evidence: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    acceptance = requirement.get("acceptance")
    acceptance_rows = (
        [_text(item) for item in acceptance if _text(item)]
        if isinstance(acceptance, list)
        else []
    )
    acceptance_text = "\n".join(f"- {row}" for row in acceptance_rows) or "- none supplied"
    return [
        {
            "role": "system",
            "content": (
                "Complete one host-selected engineering worksheet for exactly one requirement. "
                "Return only the JSON object required by the supplied response schema. Do not emit "
                "analysis, reasoning, commentary, markdown, code fences, or undeclared keys. Do not "
                "invent target API names, symbols, versions, repository paths, external facts, or "
                "evidence identifiers. Use only evidence_refs shown in the grounded context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requirement: {_text(requirement.get('statement'))}\n"
                "Acceptance observations supplied by the requirement:\n"
                f"{acceptance_text}\n"
                "Grounded implementation evidence:\n"
                f"{_evidence_context(evidence)}\n\n"
                f"{worksheet_prompt(selected_sections)}"
            ),
        },
    ]


def _generate_whole_requirement(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    requirement_ref: str,
    selected_sections: tuple[str, ...],
    evidence: list[Mapping[str, Any]],
) -> dict[str, Any]:
    messages = _whole_requirement_messages(requirement, selected_sections, evidence)
    schema = worksheet_schema(selected_sections)
    decoded: Mapping[str, Any] | None = None

    with planner_operation(f"detailed_requirement:{requirement_ref}:whole"):
        if hasattr(router, "generate_tool_decision"):
            try:
                raw_decision = router.generate_tool_decision(
                    "planner",
                    messages,
                    tool_name="submit_requirement_engineering_worksheet",
                    parameters=schema,
                    description="Submit the complete engineering worksheet for one requirement.",
                )
                if isinstance(raw_decision, Mapping):
                    decoded = raw_decision
            except Exception as exc:
                from .model_adapters import ModelConfigurationError

                if isinstance(exc, ModelConfigurationError):
                    raise
                emit_root_cause(
                    "detailed_requirement_whole_tool_fallback",
                    stage="planning_state",
                    operation=f"detailed_requirement:{requirement_ref}:whole",
                    result="FALLBACK",
                    reason=f"{type(exc).__name__}: {exc}",
                    details={"requirement_ref": requirement_ref, "fallback": "structured_text"},
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
                raise ValueError("whole requirement output must be a JSON object")
            decoded = parsed

    return dict(decoded)


def _salvage_dependency_closed_sections(
    decoded: Mapping[str, Any],
    *,
    selected_sections: tuple[str, ...],
    allowed: set[str],
) -> dict[str, dict[str, Any]]:
    """Keep only valid, unique sections whose direct prerequisites also survived."""
    valid: dict[str, dict[str, Any]] = {}
    seen_specifications: set[str] = set()
    for section in selected_sections:
        if section not in decoded:
            continue
        try:
            row = validate_worksheet_section(decoded[section], allowed, section)
        except (TypeError, ValueError):
            continue
        signature = json.dumps(
            row["specification"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).casefold()
        if signature in seen_specifications:
            continue
        seen_specifications.add(signature)
        valid[section] = row

    changed = True
    while changed:
        changed = False
        for section in tuple(valid):
            dependencies = _section_dependencies(section, selected_sections)
            if any(dependency not in valid for dependency in dependencies):
                del valid[section]
                changed = True
    return valid


def _attempt_whole_requirement(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    requirement_ref: str,
    selected_sections: tuple[str, ...],
    evidence: list[Mapping[str, Any]],
    allowed: set[str],
) -> tuple[dict[str, dict[str, Any]], str]:
    """Return validated whole output, or the dependency-closed valid subset plus failure."""
    try:
        decoded = _generate_whole_requirement(
            router,
            requirement=requirement,
            requirement_ref=requirement_ref,
            selected_sections=selected_sections,
            evidence=evidence,
        )
        try:
            validated = validate_worksheet(decoded, allowed, selected_sections)
            return validated, ""
        except (TypeError, ValueError) as exc:
            salvaged = _salvage_dependency_closed_sections(
                decoded,
                selected_sections=selected_sections,
                allowed=allowed,
            )
            return salvaged, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        from .model_adapters import ModelConfigurationError

        if isinstance(exc, ModelConfigurationError):
            raise
        return {}, f"{type(exc).__name__}: {exc}"


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
    messages = _section_messages(
        requirement,
        selected_sections,
        section,
        evidence,
        completed,
    )
    schema = worksheet_section_schema(section)
    decoded: Mapping[str, Any] | None = None

    with planner_operation(f"detailed_section:{section}:whole"):
        if hasattr(router, "generate_tool_decision"):
            try:
                raw_decision = router.generate_tool_decision(
                    "planner",
                    messages,
                    tool_name=f"submit_{section}_section",
                    parameters=schema,
                    description=f"Submit the complete {section} engineering worksheet section.",
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
    """Retry a failed requirement at section granularity, then atomic concerns only if needed."""
    try:
        return _generate_whole_section(
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
                "strategy": "whole_section_then_atomic_concerns",
            },
        )
        return _compile_worksheet_section(
            router,
            requirement=requirement,
            selected_sections=selected_sections,
            section=section,
            evidence=evidence,
            allowed=allowed,
            completed=completed,
        )


def _finish_requirement(
    job: dict[str, Any],
    *,
    working_state: Mapping[str, Any],
    requirement_order: tuple[str, ...],
    completed_details: dict[str, Mapping[str, Any]],
    checkpoint: Checkpoint | None,
) -> dict[str, Any]:
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
    result = _merge_completed_details(
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
    """Compile unfinished requirements without count-driven retries or unconditional fanout.

    Phase 1 has exactly one primary whole-worksheet attempt per unfinished requirement.
    A failed whole attempt transitions once into a finite section DAG; a failed section
    transitions once into the existing finite atomic-concern protocol. Nodes are never
    re-enqueued. Completed requirements and terminal failures are checkpointed, so resume
    cannot silently regenerate terminal work.
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
    working_state = _merge_completed_details(
        state,
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
        jobs.append(
            {
                "requirement": requirement,
                "requirement_ref": requirement_ref,
                "selected_sections": selected_sections,
                "evidence": evidence,
                "allowed": allowed,
                "completed": {},
                "pending": set(),
                "submitted": set(),
            }
        )

    if jobs:
        workers = max(1, min(len(jobs), router_native_model_parallelism(router)))
        future_to_job: dict[Future[tuple[dict[str, dict[str, Any]], str]], int] = {}
        remaining_whole = len(jobs)
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="planning-requirement-first",
        ) as pool:
            for job_index, job in enumerate(jobs):
                future = pool.submit(
                    _attempt_whole_requirement,
                    router,
                    requirement=job["requirement"],
                    requirement_ref=job["requirement_ref"],
                    selected_sections=job["selected_sections"],
                    evidence=job["evidence"],
                    allowed=job["allowed"],
                )
                future_to_job[future] = job_index

            while future_to_job:
                done, _ = wait(tuple(future_to_job), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: future_to_job[item]):
                    job_index = future_to_job.pop(future)
                    job = jobs[job_index]
                    before = remaining_whole
                    completed, failure = future.result()
                    remaining_whole -= 1
                    if remaining_whole >= before:
                        raise RuntimeError(
                            "DETAILED_PLAN_NO_PROGRESS: whole-requirement work did not strictly decrease"
                        )
                    job["completed"].update(completed)
                    if not failure and len(completed) == len(job["selected_sections"]):
                        working_state = _finish_requirement(
                            job,
                            working_state=working_state,
                            requirement_order=requirement_order,
                            completed_details=completed_details,
                            checkpoint=checkpoint,
                        )
                        continue

                    job["pending"] = set(job["selected_sections"]) - set(job["completed"])
                    emit_root_cause(
                        "detailed_requirement_adaptive_decomposition",
                        stage="planning_state",
                        operation="compile_progress_monotone_detailed_plans",
                        result="FALLBACK",
                        reason=failure or "whole worksheet incomplete",
                        details={
                            "requirement_ref": job["requirement_ref"],
                            "retained_sections": [
                                key for key in job["selected_sections"] if key in job["completed"]
                            ],
                            "fallback_sections": [
                                key for key in job["selected_sections"] if key in job["pending"]
                            ],
                        },
                    )

    section_rank = {section: index for index, section in enumerate(WORKSHEET_SECTIONS)}
    fallback_jobs = [job for job in jobs if job["pending"]]
    remaining_sections = sum(len(job["pending"]) for job in fallback_jobs)

    def ready_nodes() -> list[tuple[int, str]]:
        ready: list[tuple[int, str]] = []
        for job_index, job in enumerate(fallback_jobs):
            for section in job["selected_sections"]:
                if section not in job["pending"] or section in job["submitted"]:
                    continue
                dependencies = _section_dependencies(section, job["selected_sections"])
                if all(dependency in job["completed"] for dependency in dependencies):
                    ready.append((job_index, section))
        return sorted(ready, key=lambda item: (section_rank[item[1]], item[0]))

    if remaining_sections:
        workers = max(1, min(remaining_sections, router_native_model_parallelism(router)))
        future_to_node: dict[Future[dict[str, Any]], tuple[int, str]] = {}
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="planning-detail-fallback",
        ) as pool:
            while remaining_sections or future_to_node:
                for job_index, section in ready_nodes():
                    if len(future_to_node) >= workers:
                        break
                    job = fallback_jobs[job_index]
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
                        for job in fallback_jobs
                        if job["pending"]
                    }
                    reason = (
                        "DETAILED_PLAN_DAG_DEADLOCK: unsatisfied contracts remain but no "
                        f"runnable section exists; {pending!r}"
                    )
                    first_job = next(job for job in fallback_jobs if job["pending"])
                    working_state = _checkpoint_terminal_blocker(
                        working_state,
                        requirement_ref=first_job["requirement_ref"],
                        section="dependency_dag",
                        reason=reason,
                        checkpoint=checkpoint,
                    )
                    raise RuntimeError(reason)

                done, _ = wait(tuple(future_to_node), return_when=FIRST_COMPLETED)
                for future in sorted(
                    done,
                    key=lambda item: (
                        section_rank[future_to_node[item][1]],
                        future_to_node[item][0],
                    ),
                ):
                    job_index, section = future_to_node.pop(future)
                    job = fallback_jobs[job_index]
                    try:
                        result = future.result()
                    except Exception as exc:
                        for pending_future in future_to_node:
                            pending_future.cancel()
                        reason = (
                            "DETAILED_PLAN_BLOCKED: minimal fallback contract failed for "
                            f"{job['requirement_ref']}/{section}: {type(exc).__name__}: {exc}"
                        )
                        working_state = _checkpoint_terminal_blocker(
                            working_state,
                            requirement_ref=job["requirement_ref"],
                            section=section,
                            reason=reason,
                            checkpoint=checkpoint,
                        )
                        raise RuntimeError(reason) from exc

                    if section not in job["pending"]:
                        raise RuntimeError(
                            "DETAILED_PLAN_PROGRESS_INVARIANT: completed node was not pending"
                        )
                    before = remaining_sections
                    job["completed"][section] = result
                    job["pending"].remove(section)
                    job["submitted"].remove(section)
                    remaining_sections -= 1
                    if remaining_sections >= before:
                        raise RuntimeError(
                            "DETAILED_PLAN_NO_PROGRESS: pending section work did not strictly decrease"
                        )
                    if not job["pending"]:
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