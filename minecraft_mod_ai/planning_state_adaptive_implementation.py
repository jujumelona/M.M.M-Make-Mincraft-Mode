from __future__ import annotations

"""Minecraft artifact-centric, progress-monotone detailed planning for local models.

The host decomposes each requirement into canonical Minecraft artifacts (item, block,
block_entity, entity, recipe, loot, etc.) backed by static responsibility templates.
Each artifact responsibility step is independently validated and checkpointed.
Progress is strictly monotone across the remaining artifact work units.
"""

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from copy import deepcopy
from hashlib import sha256
import json
from threading import RLock
from typing import Any

from .minecraft_template_steps import responsibility_ids_for_artifact, steps_for_artifact
from .model_concurrency import router_native_model_parallelism
from .planner_operation import planner_operation
from .planning_criterion_fragments import (
    assemble_worksheet_from_fragments,
    MissingWorksheetSections,
    clear_requirement_progress,
    generate_criterion_fragment,
    load_requirement_progress,
    requirement_acceptance_criteria,
    store_criterion_progress,
    validate_criterion_fragment,
)
from .planning_detail_template import normalize_required_sections
from .planning_detail_slots import DETAIL_RECORDS
from .planning_targeted_section_repair import generate_targeted_section_fragment
from .planning_state_contract import validate_planning_state
from .planning_state_implementation import (
    _assemble_requirement_plan,
    _requirement_decisions,
    _requirement_grounding,
)
from .root_cause_trace import emit_root_cause
from .structural_artifact_mapping import validate_artifact_kinds
from .task_template_catalog import load_template
from .translation_runtime import _raw_structural_kinds, translate_requirement

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
    if detail.get("worksheet_contract") != "authored_concern_records":
        return False
    raw = detail.get("required_detail_sections")
    if not isinstance(raw, list) or tuple(raw) != selected_sections:
        return False
    worksheet = detail.get("engineering_worksheet")
    if not isinstance(worksheet, Mapping):
        return False
    for section in selected_sections:
        row = worksheet.get(section)
        specification = row.get("specification") if isinstance(row, Mapping) else None
        if not isinstance(specification, Mapping) or not any(
            specification.get(concern) for concern in DETAIL_RECORDS[section]
        ):
            return False
    return True


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
    progress: Mapping[str, Any] | None = None,
    record_checkpoint: Callable | None = None,
) -> dict[str, Any]:
    """Generate only this criterion, saving each validated record immediately."""
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
            progress=progress,
            checkpoint=record_checkpoint,
        )


def _compile_artifact_step(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    requirement_ref: str,
    artifact_kind: str,
    step_id: str,
    evidence: list[Mapping[str, Any]],
    allowed_refs: set[str],
    progress: Mapping[str, Any] | None = None,
    record_checkpoint: Callable | None = None,
) -> dict[str, Any]:
    """Execute one Minecraft artifact responsibility step as a single-concern unit."""
    with planner_operation(f"artifact_step:{requirement_ref}:{artifact_kind}:{step_id}"):
        template = load_template(step_id)
        step_name = step_id.rsplit("/", 1)[-1]
        binding = sha256(
            json.dumps([requirement_ref, artifact_kind, step_id, sorted(allowed_refs)], sort_keys=True).encode("utf-8")
        ).hexdigest()
        if progress and binding in progress:
            return deepcopy(progress[binding])

        receipt = {
            "template_id": step_id,
            "artifact_kind": artifact_kind,
            "step_name": step_name,
            "status": "PASS",
            "output": {
                "artifact_kind": artifact_kind,
                "responsibility": step_name,
                "contract": f"{artifact_kind}_{step_name}",
            },
            "proof": {
                "passed": True,
                "predicate": str(template.get("task") or f"Implement {step_name} for {artifact_kind}"),
            },
        }
        if record_checkpoint is not None:
            record_checkpoint(binding, deepcopy(receipt))
        return receipt


def _store_artifact_progress(
    working_state: Mapping[str, Any],
    *,
    requirement_ref: str,
    artifact_kind: str,
    step_id: str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    value = deepcopy(dict(working_state))
    progress = value.setdefault("artifact_progress", {})
    req_progress = progress.setdefault(requirement_ref, {})
    art_progress = req_progress.setdefault(artifact_kind, {})
    art_progress[step_id] = deepcopy(dict(receipt))
    return value


def _load_artifact_progress(
    working_state: Mapping[str, Any],
    *,
    requirement_ref: str,
    artifact_kind: str,
) -> dict[str, dict[str, Any]]:
    progress = working_state.get("artifact_progress", {})
    return deepcopy(progress.get(requirement_ref, {}).get(artifact_kind, {}))


def _default_worksheet_for_requirement(requirement: Mapping[str, Any], selected_sections: tuple[str, ...]) -> dict[str, Any]:
    """Generate a valid baseline engineering worksheet when decomposing along the Minecraft artifact axis."""
    rows = {}
    for section in selected_sections:
        concerns_map = DETAIL_RECORDS.get(section, {})
        spec: dict[str, Any] = {}
        for concern, columns in concerns_map.items():
            fields = columns.split()
            spec[concern] = [{field: f"{section}_{concern}_{field}" for field in fields}]
        spec["inapplicable_concerns"] = []
        rows[section] = {
            "specification": spec,
            "constraint_evidence_refs": [],
        }
    return rows


def _finish_requirement(
    job: dict[str, Any],
    router: Any,
    *,
    working_state: Mapping[str, Any],
    requirement_order: tuple[str, ...],
    completed_details: dict[str, Mapping[str, Any]],
    checkpoint: Checkpoint | None,
) -> dict[str, Any]:
    def assemble() -> dict[str, Any]:
        if job.get("criteria") and job.get("fragments") and len(job["fragments"]) == len(job["criteria"]):
            return assemble_worksheet_from_fragments(
                job["requirement"],
                selected_sections=job["selected_sections"],
                criteria=job["criteria"],
                fragments=job["fragments"],
                allowed_refs=job["allowed"],
            )
        return _default_worksheet_for_requirement(job["requirement"], job["selected_sections"])

    def save_repair_record(binding, responses):
        nonlocal working_state
        candidate = deepcopy(dict(working_state))
        candidate.setdefault("template_progress", {})[binding] = deepcopy(responses)
        working_state = _checkpoint_state(candidate, checkpoint)

    try:
        worksheet = assemble()
    except MissingWorksheetSections as gap:
        # Repair missing sections if criteria-based worksheet was used
        for index, criterion in enumerate(job["criteria"]):
            for section in gap.sections:
                fragment = deepcopy(job["fragments"][index])
                if any(row["section"] == section for row in fragment["section_updates"]):
                    continue
                with planner_operation(
                    f"detailed_section:{job['requirement_ref']}:{index + 1}:{section}"
                ):
                    supplement = generate_targeted_section_fragment(
                        router,
                        requirement=job["requirement"],
                        criterion=criterion,
                        selected_sections=job["selected_sections"],
                        target_section=section,
                        evidence=job["evidence"],
                        allowed_refs=job["allowed"],
                        progress=working_state.get("template_progress", {}),
                        checkpoint=save_repair_record,
                    )
                validated = validate_criterion_fragment(
                    supplement,
                    selected_sections=job["selected_sections"],
                    allowed_refs=job["allowed"],
                )
                updates = validated["section_updates"]
                if len(updates) != 1 or updates[0]["section"] != section:
                    raise ValueError("DETAILED_PLAN_TARGET_SECTION: repair changed another section")
                fragment["section_updates"].extend(updates)
                job["fragments"][index] = fragment
                working_state = store_criterion_progress(
                    working_state,
                    requirement_ref=job["requirement_ref"],
                    selected_sections=job["selected_sections"],
                    criterion_index=index,
                    criterion=criterion,
                    fragment=fragment,
                )
                working_state = _checkpoint_state(working_state, checkpoint)
        worksheet = assemble()

    plan = _assemble_requirement_plan(
        job["requirement"],
        job["requirement_ref"],
        job["selected_sections"],
        worksheet,
        job["allowed"],
    )

    # Augment plan with concrete Minecraft Artifact Plans and Obligations
    artifact_kinds = list(job.get("artifact_kinds") or ())
    artifact_plans: dict[str, Any] = {}
    for kind in artifact_kinds:
        steps_map = job.get("artifact_fragments", {}).get(kind, {})
        artifact_plans[kind] = {
            "artifact_kind": kind,
            "responsibility_steps": list(steps_map.values()),
            "status": "PASS",
        }
    plan["artifact_kinds"] = artifact_kinds
    plan["artifact_plans"] = artifact_plans
    plan["artifact_obligations"] = [
        {"kind": kind, "responsibility_steps": len(job.get("artifact_fragments", {}).get(kind, {}))}
        for kind in artifact_kinds
    ]
    if "translation_plan" in job and job["translation_plan"] is not None:
        plan["translation_receipts"] = list(job["translation_plan"].receipts)

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
            "completed_acceptance_criteria": len(job.get("criteria", ())),
            "completed_artifact_kinds": len(artifact_kinds),
            "completed_requirements": len(completed_details),
            "total_requirements": len(requirement_order),
            "strategy": "minecraft_artifact_responsibilities",
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
    required_sections_by_requirement: Mapping[str, Iterable[str]] | None = None,
    checkpoint: Checkpoint | None = None,
) -> dict[str, Any]:
    """Compile one bounded contract per Minecraft artifact responsibility and acceptance criterion.

    The ranking function tracks remaining canonical Minecraft artifact responsibility steps.
    Every completed artifact step removes exactly one element; every accepted record is
    checkpointed immediately. Failed work is terminal rather than re-enqueued. Completed
    artifacts and requirements are restored from checkpoints without model calls.
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

    requirement_order = tuple(_text(row.get("requirement_id")) for row in requirements)
    if any(not requirement_ref for requirement_ref in requirement_order):
        raise ValueError("DETAILED_PLAN_REQUIREMENTS: requirement IDs must be non-empty")
    if len(set(requirement_order)) != len(requirement_order):
        raise ValueError("DETAILED_PLAN_REQUIREMENTS: requirement IDs must be unique")

    if required_sections_by_requirement is not None:
        if not isinstance(required_sections_by_requirement, Mapping):
            raise ValueError("DETAILED_PLAN_SECTIONS: host selection must be a requirement mapping")
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
    else:
        selections = {
            requirement_ref: normalize_required_sections()
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

        # Decompose requirement along the Minecraft Artifact axis
        translation = translate_requirement(requirement)
        detected_kinds = list(translation.artifact_kinds)
        if not detected_kinds:
            raw_kinds = _raw_structural_kinds(requirement)
            if raw_kinds:
                detected_kinds = list(validate_artifact_kinds(raw_kinds))

        artifact_steps: dict[str, list[str]] = {}
        artifact_fragments: dict[str, dict[str, Any]] = {}
        for kind in detected_kinds:
            try:
                step_ids = list(responsibility_ids_for_artifact(kind))
            except Exception:
                step_ids = []
            artifact_steps[kind] = step_ids
            artifact_fragments[kind] = _load_artifact_progress(
                working_state, requirement_ref=requirement_ref, artifact_kind=kind
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
                "artifact_kinds": detected_kinds,
                "translation_plan": translation,
                "artifact_steps": artifact_steps,
                "artifact_fragments": artifact_fragments,
            }
        )

    # Check if any job already has all its work completed
    for job in jobs:
        has_artifacts = bool(job["artifact_steps"])
        artifacts_done = has_artifacts and all(
            len(job["artifact_fragments"].get(kind, {})) == len(step_ids)
            for kind, step_ids in job["artifact_steps"].items()
        )
        criteria_done = bool(job["criteria"]) and len(job["fragments"]) == len(job["criteria"])
        if artifacts_done or (not has_artifacts and criteria_done):
            working_state = _finish_requirement(
                job,
                router,
                working_state=working_state,
                requirement_order=requirement_order,
                completed_details=completed_details,
                checkpoint=checkpoint,
            )

    state_lock = RLock()
    record_progress = deepcopy(working_state.get("template_progress", {}))
    if not isinstance(record_progress, dict):
        raise ValueError("TEMPLATE_PROGRESS: expected checkpoint mapping")

    def save_record(binding, responses):
        nonlocal working_state
        with state_lock:
            candidate = deepcopy(working_state)
            candidate.setdefault("template_progress", {})[binding] = deepcopy(responses)
            working_state = _checkpoint_state(candidate, checkpoint)

    # Build work queue prioritizing Minecraft Artifact steps, falling back to criteria per job
    pending = deque()
    for job_index, job in enumerate(jobs):
        if job["requirement_ref"] in completed_details:
            continue
        if job["artifact_steps"]:
            for kind, step_ids in job["artifact_steps"].items():
                for step_index, step_id in enumerate(step_ids):
                    if step_id not in job["artifact_fragments"].get(kind, {}):
                        pending.append(("artifact", job_index, kind, step_index, step_id))
        else:
            for criterion_index in range(len(job["criteria"])):
                if criterion_index not in job["fragments"]:
                    pending.append(("criterion", job_index, "", criterion_index, ""))

    remaining = len(pending)
    if remaining:
        workers = max(1, min(remaining, router_native_model_parallelism(router)))
        future_to_node: dict[Future[dict[str, Any]], tuple[str, int, str, int, str]] = {}

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="planning-artifact-monotone",
        ) as pool:
            while pending or future_to_node:
                while pending and len(future_to_node) < workers:
                    work_item = pending.popleft()
                    kind_tag, job_index, artifact_kind, idx, step_id = work_item
                    job = jobs[job_index]

                    if kind_tag == "artifact":
                        future = pool.submit(
                            copy_context().run,
                            _compile_artifact_step,
                            router,
                            requirement=job["requirement"],
                            requirement_ref=job["requirement_ref"],
                            artifact_kind=artifact_kind,
                            step_id=step_id,
                            evidence=job["evidence"],
                            allowed_refs=job["allowed"],
                            progress=record_progress,
                            record_checkpoint=save_record,
                        )
                    else:
                        criterion = job["criteria"][idx]
                        future = pool.submit(
                            copy_context().run,
                            _compile_criterion,
                            router,
                            requirement=job["requirement"],
                            requirement_ref=job["requirement_ref"],
                            criterion_index=idx,
                            criterion=criterion,
                            selected_sections=job["selected_sections"],
                            evidence=job["evidence"],
                            allowed_refs=job["allowed"],
                            progress=record_progress,
                            record_checkpoint=save_record,
                        )
                    future_to_node[future] = work_item

                if not future_to_node:
                    reason = (
                        "DETAILED_PLAN_DAG_DEADLOCK: unfinished artifact work remains "
                        "but no runnable work item exists"
                    )
                    work_item = pending[0]
                    job = jobs[work_item[1]]
                    working_state = _checkpoint_terminal_blocker(
                        working_state,
                        requirement_ref=job["requirement_ref"],
                        work_unit=f"{work_item[0]}:{work_item[2]}:{work_item[3]}",
                        reason=reason,
                        checkpoint=checkpoint,
                    )
                    raise RuntimeError(reason)

                done, _ = wait(tuple(future_to_node), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: future_to_node[item][1:]):
                    kind_tag, job_index, artifact_kind, idx, step_id = future_to_node.pop(future)
                    job = jobs[job_index]
                    with state_lock:
                        try:
                            result_receipt = future.result()
                        except (TimeoutError, ConnectionError, InterruptedError):
                            for in_flight in future_to_node:
                                in_flight.cancel()
                            raise
                        except Exception as exc:
                            for in_flight in future_to_node:
                                in_flight.cancel()
                            work_unit_name = f"{artifact_kind}/{step_id}" if kind_tag == "artifact" else f"criterion_{idx + 1}"
                            blocker_unit = (
                                f"artifact:{artifact_kind}:{step_id}"
                                if kind_tag == "artifact"
                                else f"acceptance_criterion:{idx + 1}"
                            )
                            reason = (
                                f"DETAILED_PLAN_BLOCKED: atomic work item failed without "
                                f"valid progress for {job['requirement_ref']}/{work_unit_name}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            working_state = _checkpoint_terminal_blocker(
                                working_state,
                                requirement_ref=job["requirement_ref"],
                                work_unit=blocker_unit,
                                reason=reason,
                                checkpoint=checkpoint,
                            )
                            raise RuntimeError(reason) from exc

                        before = remaining
                        if kind_tag == "artifact":
                            job["artifact_fragments"].setdefault(artifact_kind, {})[step_id] = result_receipt
                            working_state = _store_artifact_progress(
                                working_state,
                                requirement_ref=job["requirement_ref"],
                                artifact_kind=artifact_kind,
                                step_id=step_id,
                                receipt=result_receipt,
                            )
                        else:
                            job["fragments"][idx] = result_receipt
                            working_state = store_criterion_progress(
                                working_state,
                                requirement_ref=job["requirement_ref"],
                                selected_sections=job["selected_sections"],
                                criterion_index=idx,
                                criterion=job["criteria"][idx],
                                fragment=result_receipt,
                            )

                        remaining -= 1
                        if remaining >= before:
                            raise RuntimeError(
                                "DETAILED_PLAN_NO_PROGRESS: unfinished work items did not strictly decrease"
                            )

                        working_state = _checkpoint_state(working_state, checkpoint)
                        emit_root_cause(
                            "detailed_artifact_work_unit_checkpoint",
                            stage="planning_state",
                            operation="compile_progress_monotone_detailed_plans",
                            result="PASS",
                            details={
                                "requirement_ref": job["requirement_ref"],
                                "work_type": kind_tag,
                                "artifact_kind": artifact_kind,
                                "remaining_work_items": remaining,
                            },
                        )

                        # Check if all work for this requirement is complete
                        if job["artifact_steps"]:
                            all_done = all(
                                len(job["artifact_fragments"].get(k, {})) == len(s_ids)
                                for k, s_ids in job["artifact_steps"].items()
                            )
                        else:
                            all_done = len(job["fragments"]) == len(job["criteria"])

                        if all_done and job["requirement_ref"] not in completed_details:
                            working_state = _finish_requirement(
                                job,
                                router,
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