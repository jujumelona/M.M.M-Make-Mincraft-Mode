from __future__ import annotations

"""Minecraft artifact-centric, progress-monotone detailed planning for local models.

The host decomposes each requirement into canonical Minecraft artifacts (item, block,
block_entity, entity, recipe, loot, etc.) backed by static responsibility templates.
Artifact responsibility work and public acceptance-detail work are independent
obligations: neither can substitute for the other. Progress is checkpointed after each
validated work unit and is strictly monotone across the remaining work set.
"""

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from hashlib import sha256
import json
from threading import RLock
from typing import Any

from .deadline_executor import (
    ParallelExecutionTimeout,
    ParallelTaskError,
    iter_completed_with_deadlines,
)
from .minecraft_template_steps import responsibility_ids_for_artifact
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
    # Plans produced before the acceptance/artifact gate was fixed may contain a
    # placeholder worksheet assembled without criterion work. Force those plans to be
    # regenerated instead of restoring them as completed checkpoints.
    if detail.get("acceptance_criteria_complete") is not True:
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


def _planning_only_artifact_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize static artifact planning so it can never impersonate runtime proof."""
    value = deepcopy(dict(receipt))
    legacy_proof = value.pop("proof", None)
    planned_validation = value.get("planned_validation")
    if not isinstance(planned_validation, Mapping):
        planned_validation = {}
    planned_validation = dict(planned_validation)
    if isinstance(legacy_proof, Mapping):
        scope = _text(legacy_proof.get("scope"))
        predicate = _text(legacy_proof.get("predicate"))
        if scope:
            planned_validation.setdefault("scope", scope)
        if predicate:
            planned_validation.setdefault("predicate", predicate)
    value["status"] = "PLANNED"
    value["verification_status"] = "PENDING_RUNTIME_VALIDATION"
    if planned_validation:
        value["planned_validation"] = planned_validation
    proof = value.get("proof")
    if value.get("status") == "PASS" or (
        isinstance(proof, Mapping) and proof.get("passed") is True
    ):
        raise RuntimeError(
            "ARTIFACT_PLAN_AUTHORITY: planning output cannot assert runtime PASS"
        )
    return value


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
    """Plan one static Minecraft artifact responsibility without claiming verification."""
    del router, requirement, evidence
    with planner_operation(f"artifact_step:{requirement_ref}:{artifact_kind}:{step_id}"):
        template = load_template(step_id)
        step_name = step_id.rsplit("/", 1)[-1]
        binding = sha256(
            json.dumps(
                [requirement_ref, artifact_kind, step_id, sorted(allowed_refs)],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if progress and binding in progress:
            cached = progress[binding]
            if not isinstance(cached, Mapping):
                raise ValueError("ARTIFACT_PLAN_PROGRESS: cached receipt must be a mapping")
            return _planning_only_artifact_receipt(cached)

        receipt = _planning_only_artifact_receipt(
            {
                "template_id": step_id,
                "artifact_kind": artifact_kind,
                "step_name": step_name,
                "output": {
                    "artifact_kind": artifact_kind,
                    "responsibility": step_name,
                    "contract": f"{artifact_kind}_{step_name}",
                },
                "planned_validation": {
                    "scope": "static_responsibility_template",
                    "predicate": str(
                        template.get("task")
                        or f"Implement {step_name} for {artifact_kind}"
                    ),
                },
            }
        )
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
    raw = progress.get(requirement_ref, {}).get(artifact_kind, {})
    if not isinstance(raw, Mapping):
        raise ValueError("ARTIFACT_PLAN_PROGRESS: artifact progress must be a mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for step_id, receipt in raw.items():
        if not isinstance(receipt, Mapping):
            raise ValueError("ARTIFACT_PLAN_PROGRESS: artifact receipt must be a mapping")
        normalized[str(step_id)] = _planning_only_artifact_receipt(receipt)
    return normalized


def _criteria_complete(job: Mapping[str, Any]) -> bool:
    criteria = tuple(job.get("criteria") or ())
    fragments = job.get("fragments") or {}
    return bool(criteria) and len(fragments) == len(criteria)


def _artifacts_complete(job: Mapping[str, Any]) -> bool:
    artifact_steps = job.get("artifact_steps") or {}
    artifact_fragments = job.get("artifact_fragments") or {}
    return all(
        len(artifact_fragments.get(kind, {})) == len(step_ids)
        for kind, step_ids in artifact_steps.items()
    )


def _requirement_work_complete(job: Mapping[str, Any]) -> bool:
    """Require public acceptance detail and artifact responsibilities independently."""
    return _criteria_complete(job) and _artifacts_complete(job)


def _pending_work_items(
    jobs: list[dict[str, Any]],
    completed_details: Mapping[str, Mapping[str, Any]],
) -> deque[tuple[str, int, str, int, str]]:
    """Return every missing work unit; artifacts never suppress criterion work."""
    pending: deque[tuple[str, int, str, int, str]] = deque()
    for job_index, job in enumerate(jobs):
        if job["requirement_ref"] in completed_details:
            continue
        for kind, step_ids in job["artifact_steps"].items():
            for step_index, step_id in enumerate(step_ids):
                if step_id not in job["artifact_fragments"].get(kind, {}):
                    pending.append(("artifact", job_index, kind, step_index, step_id))
        for criterion_index in range(len(job["criteria"])):
            if criterion_index not in job["fragments"]:
                pending.append(("criterion", job_index, "", criterion_index, ""))
    return pending


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
        if not _criteria_complete(job):
            raise RuntimeError(
                "DETAILED_PLAN_ACCEPTANCE_INCOMPLETE: acceptance criteria cannot be "
                "replaced by artifact responsibility planning"
            )
        return assemble_worksheet_from_fragments(
            job["requirement"],
            selected_sections=job["selected_sections"],
            criteria=job["criteria"],
            fragments=job["fragments"],
            allowed_refs=job["allowed"],
        )

    def save_repair_record(binding, responses):
        nonlocal working_state
        candidate = deepcopy(dict(working_state))
        candidate.setdefault("template_progress", {})[binding] = deepcopy(responses)
        working_state = _checkpoint_state(candidate, checkpoint)

    try:
        worksheet = assemble()
    except MissingWorksheetSections as gap:
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
                    raise ValueError(
                        "DETAILED_PLAN_TARGET_SECTION: repair changed another section"
                    )
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

    if not _artifacts_complete(job):
        raise RuntimeError(
            "DETAILED_PLAN_ARTIFACT_INCOMPLETE: artifact responsibility work is incomplete"
        )

    plan = _assemble_requirement_plan(
        job["requirement"],
        job["requirement_ref"],
        job["selected_sections"],
        worksheet,
        job["allowed"],
    )
    plan["acceptance_criteria_complete"] = True
    plan["acceptance_criteria_count"] = len(job["criteria"])

    artifact_kinds = list(job.get("artifact_kinds") or ())
    artifact_plans: dict[str, Any] = {}
    for kind in artifact_kinds:
        steps_map = job.get("artifact_fragments", {}).get(kind, {})
        responsibility_steps = [
            _planning_only_artifact_receipt(receipt)
            for receipt in steps_map.values()
        ]
        artifact_plans[kind] = {
            "artifact_kind": kind,
            "responsibility_steps": responsibility_steps,
            "status": "PLANNED",
            "verification_status": "PENDING_RUNTIME_VALIDATION",
        }
    plan["artifact_kinds"] = artifact_kinds
    plan["artifact_plans"] = artifact_plans
    plan["artifact_obligations"] = [
        {
            "kind": kind,
            "responsibility_steps": len(
                job.get("artifact_fragments", {}).get(kind, {})
            ),
        }
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
        result="OBSERVED",
        details={
            "requirement_ref": job["requirement_ref"],
            "verification_status": "pending_runtime_validation",
            "completed_acceptance_criteria": len(job.get("criteria", ())),
            "completed_artifact_kinds": len(artifact_kinds),
            "completed_requirements": len(completed_details),
            "total_requirements": len(requirement_order),
            "strategy": "minecraft_artifact_responsibilities_plus_acceptance",
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
    """Compile all artifact responsibilities and acceptance criteria for each requirement.

    Every completed work unit removes exactly one element from the pending set and is
    checkpointed immediately. Artifact responsibility work is structural planning only;
    it never substitutes for public acceptance-detail planning. Failed work is terminal
    rather than re-enqueued. Completed requirements are restored only when they carry
    the post-fix acceptance-completion marker.
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
            raise ValueError(
                "DETAILED_PLAN_SECTIONS: host selection must be a requirement mapping"
            )
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
        if not criteria:
            raise ValueError(
                f"DETAILED_PLAN_ACCEPTANCE: {requirement_ref} has no acceptance criterion"
            )
        fragments = load_requirement_progress(
            working_state,
            requirement_ref=requirement_ref,
            selected_sections=selected_sections,
            criteria=criteria,
            allowed_refs=allowed,
        )

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
                working_state,
                requirement_ref=requirement_ref,
                artifact_kind=kind,
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

    for job in jobs:
        if _requirement_work_complete(job):
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

    pending_items = list(_pending_work_items(jobs, completed_details))
    remaining = len(pending_items)
    if remaining:
        workers = max(1, min(remaining, router_native_model_parallelism(router)))

        def execute_work_item(
            work_item: tuple[str, int, str, int, str],
        ) -> dict[str, Any]:
            kind_tag, job_index, artifact_kind, idx, step_id = work_item
            job = jobs[job_index]
            if kind_tag == "artifact":
                return _compile_artifact_step(
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
            criterion = job["criteria"][idx]
            return _compile_criterion(
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

        try:
            completed_work = iter_completed_with_deadlines(
                pending_items,
                execute_work_item,
                max_workers=workers,
                stage="planning-artifact-monotone",
                sort_key=lambda item: item[1:],
            )
            for work_item, result_receipt in completed_work:
                kind_tag, job_index, artifact_kind, idx, step_id = work_item
                job = jobs[job_index]
                with state_lock:
                    before = remaining
                    if kind_tag == "artifact":
                        result_receipt = _planning_only_artifact_receipt(result_receipt)
                        job["artifact_fragments"].setdefault(artifact_kind, {})[
                            step_id
                        ] = result_receipt
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

                    if (
                        _requirement_work_complete(job)
                        and job["requirement_ref"] not in completed_details
                    ):
                        working_state = _finish_requirement(
                            job,
                            router,
                            working_state=working_state,
                            requirement_order=requirement_order,
                            completed_details=completed_details,
                            checkpoint=checkpoint,
                        )
        except ParallelExecutionTimeout as exc:
            kind_tag, job_index, artifact_kind, idx, step_id = exc.item
            job = jobs[job_index]
            work_unit_name = (
                f"{artifact_kind}/{step_id}"
                if kind_tag == "artifact"
                else f"criterion_{idx + 1}"
            )
            reason = (
                "DETAILED_PLAN_TIMEOUT: bounded atomic work deadline expired for "
                f"{job['requirement_ref']}/{work_unit_name}: {exc}"
            )
            emit_root_cause(
                "detailed_planning_timeout",
                stage="planning_state",
                operation="compile_progress_monotone_detailed_plans",
                result="FAIL",
                reason=reason,
                details={
                    "requirement_ref": job["requirement_ref"],
                    "work_type": kind_tag,
                    "artifact_kind": artifact_kind,
                    "work_index": idx,
                    "deadline_kind": exc.deadline_kind,
                    "elapsed_seconds": exc.elapsed_seconds,
                    "work_unit_timeout_seconds": exc.work_unit_timeout_seconds,
                    "policy": "retryable_abort_current_stage",
                },
            )
            raise TimeoutError(reason) from exc
        except ParallelTaskError as exc:
            kind_tag, job_index, artifact_kind, idx, step_id = exc.item
            job = jobs[job_index]
            cause = exc.cause
            work_unit_name = (
                f"{artifact_kind}/{step_id}"
                if kind_tag == "artifact"
                else f"criterion_{idx + 1}"
            )
            if isinstance(cause, (TimeoutError, ConnectionError, InterruptedError)):
                reason = (
                    "DETAILED_PLAN_TRANSPORT_INTERRUPTED: retryable atomic work interruption for "
                    f"{job['requirement_ref']}/{work_unit_name}: "
                    f"{type(cause).__name__}: {cause}"
                )
                emit_root_cause(
                    "detailed_planning_transport_interrupted",
                    stage="planning_state",
                    operation="compile_progress_monotone_detailed_plans",
                    result="FAIL",
                    reason=reason,
                    details={
                        "requirement_ref": job["requirement_ref"],
                        "work_type": kind_tag,
                        "artifact_kind": artifact_kind,
                        "work_index": idx,
                        "policy": "retryable_abort_current_stage",
                    },
                )
                raise cause

            blocker_unit = (
                f"artifact:{artifact_kind}:{step_id}"
                if kind_tag == "artifact"
                else f"acceptance_criterion:{idx + 1}"
            )
            reason = (
                "DETAILED_PLAN_BLOCKED: atomic work item failed without valid progress for "
                f"{job['requirement_ref']}/{work_unit_name}: "
                f"{type(cause).__name__}: {cause}"
            )
            working_state = _checkpoint_terminal_blocker(
                working_state,
                requirement_ref=job["requirement_ref"],
                work_unit=blocker_unit,
                reason=reason,
                checkpoint=checkpoint,
            )
            raise RuntimeError(reason) from cause
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