from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_between(text: str, start: str, end: str, new: str, *, label: str) -> str:
    start_at = text.find(start)
    if start_at < 0:
        raise RuntimeError(f"{label}: start marker not found")
    end_at = text.find(end, start_at)
    if end_at < 0:
        raise RuntimeError(f"{label}: end marker not found")
    return text[:start_at] + new + text[end_at:]


planner_path = ROOT / "minecraft_mod_ai/planning_state_adaptive_implementation.py"
planner = planner_path.read_text(encoding="utf-8")
planner = replace_once(
    planner,
    "from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait\nfrom contextvars import copy_context\n",
    "",
    label="planner concurrent imports",
)
planner = replace_once(
    planner,
    "from .minecraft_template_steps import responsibility_ids_for_artifact\n",
    "from .deadline_executor import (\n"
    "    ParallelExecutionTimeout,\n"
    "    ParallelTaskError,\n"
    "    iter_completed_with_deadlines,\n"
    ")\n"
    "from .minecraft_template_steps import responsibility_ids_for_artifact\n",
    label="planner deadline import",
)

artifact_block = '''def _planning_only_artifact_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
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


'''
planner = replace_between(
    planner,
    "def _compile_artifact_step(\n",
    "def _store_artifact_progress(\n",
    artifact_block,
    label="artifact planning authority",
)

planner = replace_once(
    planner,
    '''    progress = working_state.get("artifact_progress", {})
    return deepcopy(progress.get(requirement_ref, {}).get(artifact_kind, {}))
''',
    '''    progress = working_state.get("artifact_progress", {})
    raw = progress.get(requirement_ref, {}).get(artifact_kind, {})
    if not isinstance(raw, Mapping):
        raise ValueError("ARTIFACT_PLAN_PROGRESS: artifact progress must be a mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for step_id, receipt in raw.items():
        if not isinstance(receipt, Mapping):
            raise ValueError("ARTIFACT_PLAN_PROGRESS: artifact receipt must be a mapping")
        normalized[str(step_id)] = _planning_only_artifact_receipt(receipt)
    return normalized
''',
    label="artifact progress normalization",
)

planner = replace_once(
    planner,
    '''        artifact_plans[kind] = {
            "artifact_kind": kind,
            "responsibility_steps": list(steps_map.values()),
            "status": "PASS",
        }
''',
    '''        responsibility_steps = [
            _planning_only_artifact_receipt(receipt)
            for receipt in steps_map.values()
        ]
        artifact_plans[kind] = {
            "artifact_kind": kind,
            "responsibility_steps": responsibility_steps,
            "status": "PLANNED",
            "verification_status": "PENDING_RUNTIME_VALIDATION",
        }
''',
    label="artifact plan status",
)
planner = replace_once(
    planner,
    '''        result="PASS",
        details={
            "requirement_ref": job["requirement_ref"],
            "completed_acceptance_criteria": len(job.get("criteria", ())),
''',
    '''        result="OBSERVED",
        details={
            "requirement_ref": job["requirement_ref"],
            "verification_status": "pending_runtime_validation",
            "completed_acceptance_criteria": len(job.get("criteria", ())),
''',
    label="requirement planning event semantics",
)

scheduler_block = '''    pending_items = list(_pending_work_items(jobs, completed_details))
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
'''
planner = replace_between(
    planner,
    "    pending = _pending_work_items(jobs, completed_details)\n",
    "    if len(completed_details) != len(requirement_order):\n",
    scheduler_block,
    label="planner bounded scheduler",
)
planner_path.write_text(planner, encoding="utf-8")


research_path = ROOT / "minecraft_mod_ai/planning_state_research.py"
research = research_path.read_text(encoding="utf-8")
research = replace_once(
    research,
    "from concurrent.futures import ThreadPoolExecutor\nfrom contextvars import copy_context\n",
    "",
    label="research concurrent imports",
)
research = replace_once(
    research,
    "from .catalog_first_grounded_rag import forced_rag_bundle\n",
    "from .catalog_first_grounded_rag import forced_rag_bundle\n"
    "from .deadline_executor import (\n"
    "    ParallelExecutionTimeout,\n"
    "    ParallelTaskError,\n"
    "    iter_completed_with_deadlines,\n"
    ")\n",
    label="research deadline import",
)

research_scheduler = '''    domain_results: dict[str, dict[str, Any]] = {}
    if runnable_domains:
        workers = max(
            1,
            min(len(runnable_domains), router_native_model_parallelism(router)),
        )
        try:
            completed_domains = iter_completed_with_deadlines(
                runnable_domains,
                research_domain,
                max_workers=workers,
                stage="planning-research-domain",
                sort_key=lambda domain: str(domain.get("domain_id") or ""),
            )
            for _domain, result in completed_domains:
                domain_results[result["domain_id"]] = result
        except ParallelExecutionTimeout as exc:
            domain = exc.item if isinstance(exc.item, Mapping) else {}
            domain_id = str(domain.get("domain_id") or "unknown")
            reason = (
                "PLANNING_RESEARCH_TIMEOUT: bounded research-domain deadline expired for "
                f"{domain_id}: {exc}"
            )
            emit_root_cause(
                "planning_research_timeout",
                stage="planning_state",
                operation="collect_planning_state_research",
                result="FAIL",
                reason=reason,
                details={
                    "domain_id": domain_id,
                    "deadline_kind": exc.deadline_kind,
                    "elapsed_seconds": exc.elapsed_seconds,
                    "work_unit_timeout_seconds": exc.work_unit_timeout_seconds,
                    "policy": "retryable_abort_current_stage",
                },
            )
            raise TimeoutError(reason) from exc
        except ParallelTaskError as exc:
            domain = exc.item if isinstance(exc.item, Mapping) else {}
            domain_id = str(domain.get("domain_id") or "unknown")
            cause = exc.cause
            if isinstance(cause, (TimeoutError, ConnectionError, InterruptedError)):
                reason = (
                    "PLANNING_RESEARCH_TRANSPORT_INTERRUPTED: retryable research-domain "
                    f"interruption for {domain_id}: {type(cause).__name__}: {cause}"
                )
                emit_root_cause(
                    "planning_research_transport_interrupted",
                    stage="planning_state",
                    operation="collect_planning_state_research",
                    result="FAIL",
                    reason=reason,
                    details={
                        "domain_id": domain_id,
                        "policy": "retryable_abort_current_stage",
                    },
                )
                raise TimeoutError(reason) from cause
            raise cause
'''
research = replace_between(
    research,
    "    domain_results: dict[str, dict[str, Any]] = {}\n",
    "    notes: list[dict[str, Any]] = []\n",
    research_scheduler,
    label="research bounded scheduler",
)
research_path.write_text(research, encoding="utf-8")


workflow_path = ROOT / ".github/workflows/planner-liveness.yml"
workflow = workflow_path.read_text(encoding="utf-8")
workflow = replace_once(
    workflow,
    "      - 'minecraft_mod_ai/planning_state_adaptive_implementation.py'\n",
    "      - 'minecraft_mod_ai/planning_state_adaptive_implementation.py'\n"
    "      - 'minecraft_mod_ai/deadline_executor.py'\n"
    "      - 'minecraft_mod_ai/model_concurrency.py'\n",
    label="workflow pull request source paths",
)
# The same source path occurs again in the push block after the first replacement.
workflow = replace_once(
    workflow,
    "      - 'minecraft_mod_ai/planning_state_adaptive_implementation.py'\n",
    "      - 'minecraft_mod_ai/planning_state_adaptive_implementation.py'\n"
    "      - 'minecraft_mod_ai/deadline_executor.py'\n"
    "      - 'minecraft_mod_ai/model_concurrency.py'\n",
    label="workflow push source paths",
)
workflow = replace_once(
    workflow,
    "      - 'tests/test_planning_progress_monotone.py'\n",
    "      - 'tests/test_planning_progress_monotone.py'\n"
    "      - 'tests/test_planning_deadline_hardening.py'\n",
    label="workflow pull request test path",
)
workflow = replace_once(
    workflow,
    "      - 'tests/test_planning_progress_monotone.py'\n",
    "      - 'tests/test_planning_progress_monotone.py'\n"
    "      - 'tests/test_planning_deadline_hardening.py'\n",
    label="workflow push test path",
)
workflow = replace_once(
    workflow,
    "          minecraft_mod_ai/planning_state_adaptive_implementation.py\n",
    "          minecraft_mod_ai/planning_state_adaptive_implementation.py\n"
    "          minecraft_mod_ai/deadline_executor.py\n"
    "          minecraft_mod_ai/model_concurrency.py\n",
    label="workflow compile sources",
)
workflow = replace_once(
    workflow,
    "          tests/test_planning_progress_monotone.py\n          tests/test_planning_convergence_contract.py\n",
    "          tests/test_planning_progress_monotone.py\n"
    "          tests/test_planning_deadline_hardening.py\n"
    "          tests/test_planning_convergence_contract.py\n",
    label="workflow compile tests",
)
workflow = replace_once(
    workflow,
    "          tests/test_planning_progress_monotone.py\n          tests/test_planning_convergence_contract.py\n",
    "          tests/test_planning_progress_monotone.py\n"
    "          tests/test_planning_deadline_hardening.py\n"
    "          tests/test_planning_convergence_contract.py\n",
    label="workflow pytest tests",
)
workflow_path.write_text(workflow, encoding="utf-8")


test_path = ROOT / "tests/test_planning_deadline_hardening.py"
test_path.write_text(
    r'''from __future__ import annotations

from hashlib import sha256
import json
import time
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import planning_state_adaptive_implementation as adaptive
from minecraft_mod_ai import planning_state_research as research
from minecraft_mod_ai.deadline_executor import (
    ParallelExecutionTimeout,
    ParallelTaskError,
    iter_completed_with_deadlines,
)
from minecraft_mod_ai.model_concurrency import (
    ModelExecutionDeadlineExceeded,
    ReentrantCapacityGate,
)


class _Router:
    pass


def _minimal_planning_state() -> dict[str, object]:
    return {
        "decisions": [],
        "coverage": [],
        "unresolved": [],
        "blockers": [],
        "detail_progress": [],
        "state_sha256": "test",
    }


def test_artifact_planning_never_claims_runtime_pass(monkeypatch) -> None:
    monkeypatch.setattr(adaptive, "load_template", lambda _step_id: {"task": "Implement item model"})
    receipt = adaptive._compile_artifact_step(
        _Router(),
        requirement={},
        requirement_ref="req_1",
        artifact_kind="item",
        step_id="item/model",
        evidence=[],
        allowed_refs=set(),
    )
    assert receipt["status"] == "PLANNED"
    assert receipt["verification_status"] == "PENDING_RUNTIME_VALIDATION"
    assert "proof" not in receipt
    assert receipt["planned_validation"]["predicate"] == "Implement item model"


def test_legacy_artifact_pass_checkpoint_is_normalized_to_unverified(monkeypatch) -> None:
    monkeypatch.setattr(adaptive, "load_template", lambda _step_id: {"task": "Implement item model"})
    binding = sha256(
        json.dumps(["req_1", "item", "item/model", []], sort_keys=True).encode("utf-8")
    ).hexdigest()
    receipt = adaptive._compile_artifact_step(
        _Router(),
        requirement={},
        requirement_ref="req_1",
        artifact_kind="item",
        step_id="item/model",
        evidence=[],
        allowed_refs=set(),
        progress={
            binding: {
                "template_id": "item/model",
                "artifact_kind": "item",
                "step_name": "model",
                "status": "PASS",
                "proof": {
                    "passed": True,
                    "scope": "legacy_static_template",
                    "predicate": "legacy predicate",
                },
            }
        },
    )
    assert receipt["status"] == "PLANNED"
    assert receipt["verification_status"] == "PENDING_RUNTIME_VALIDATION"
    assert "proof" not in receipt
    assert receipt["planned_validation"]["predicate"] == "legacy predicate"


def test_deadline_executor_returns_fast_results() -> None:
    results = list(
        iter_completed_with_deadlines(
            [1, 2, 3],
            lambda value: value * 2,
            max_workers=2,
            stage="test-fast",
            sort_key=lambda value: value,
        )
    )
    assert dict(results) == {1: 2, 2: 4, 3: 6}


def test_deadline_executor_does_not_wait_for_blocked_worker(monkeypatch) -> None:
    monkeypatch.setenv("MMM_PLANNING_WORK_UNIT_TIMEOUT_SECONDS", "0.03")

    def blocked(_value: int) -> int:
        time.sleep(0.25)
        return 1

    started = time.monotonic()
    with pytest.raises(ParallelExecutionTimeout):
        list(
            iter_completed_with_deadlines(
                [1],
                blocked,
                max_workers=1,
                stage="test-blocked",
            )
        )
    assert time.monotonic() - started < 0.20


def test_model_capacity_wait_inherits_work_unit_deadline(monkeypatch) -> None:
    monkeypatch.setenv("MMM_PLANNING_WORK_UNIT_TIMEOUT_SECONDS", "0.03")
    gate = ReentrantCapacityGate(lambda: 1)
    gate.acquire()
    try:
        with pytest.raises(ParallelTaskError) as caught:
            list(
                iter_completed_with_deadlines(
                    [1],
                    lambda _value: gate.acquire(),
                    max_workers=1,
                    stage="test-capacity-gate",
                )
            )
        assert isinstance(caught.value.cause, ModelExecutionDeadlineExceeded)
    finally:
        gate.release()


def test_detailed_planning_hung_work_unit_exits_with_bounded_timeout(monkeypatch) -> None:
    monkeypatch.setenv("MMM_PLANNING_WORK_UNIT_TIMEOUT_SECONDS", "0.03")
    requirement = {
        "requirement_id": "req_1",
        "statement": "requirement 1",
        "acceptance": ["acceptance 1"],
    }
    monkeypatch.setattr(adaptive, "validate_planning_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(adaptive, "_requirement_decisions", lambda _state: [requirement])
    monkeypatch.setattr(adaptive, "_requirement_grounding", lambda *_args: ([], set()))
    monkeypatch.setattr(adaptive, "load_requirement_progress", lambda *args, **kwargs: {})
    monkeypatch.setattr(adaptive, "router_native_model_parallelism", lambda _router: 1)
    monkeypatch.setattr(
        adaptive,
        "translate_requirement",
        lambda _requirement: SimpleNamespace(artifact_kinds=(), receipts=()),
    )

    def blocked_criterion(*_args, **_kwargs):
        time.sleep(0.25)
        return {"section_updates": []}

    monkeypatch.setattr(adaptive, "_compile_criterion", blocked_criterion)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="DETAILED_PLAN_TIMEOUT"):
        adaptive.compile_progress_monotone_detailed_plans(
            _Router(),
            "prompt",
            _minimal_planning_state(),
        )
    assert time.monotonic() - started < 0.20


def test_research_hung_domain_exits_with_bounded_timeout(monkeypatch) -> None:
    from minecraft_mod_ai import pre_design_grounded_rag as project_rag
    from minecraft_mod_ai import pre_design_research_pipeline as research_pipeline

    monkeypatch.setenv("MMM_PLANNING_WORK_UNIT_TIMEOUT_SECONDS", "0.03")
    monkeypatch.setattr(research, "validate_planning_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        research,
        "merge_repository_candidates",
        lambda existing, _new: list(existing),
    )
    monkeypatch.setattr(research, "_compile_pending_queries", lambda *_args: None)
    domain = {
        "domain_id": "r_1",
        "objective": "research",
        "requirements": ["research"],
        "evidence_kinds": ["gameplay_reference"],
        "queries": ["reference research"],
        "providers": ["wikipedia"],
        "required_anchor_terms": ["Reference"],
        "depends_on": [],
    }
    monkeypatch.setattr(
        research,
        "_research_brief",
        lambda *_args: ({"domains": [domain]}, {"r_1"}, {}),
    )
    monkeypatch.setattr(research, "router_native_model_parallelism", lambda _router: 1)

    def blocked_grounding(_domain):
        time.sleep(0.25)
        return {"queries": []}

    monkeypatch.setattr(research, "_grounded_reference_domain", blocked_grounding)
    monkeypatch.setattr(
        project_rag,
        "_materialize_domain_evidence_document",
        lambda *_args, **_kwargs: "document",
    )
    monkeypatch.setattr(
        research,
        "research_document_domain",
        lambda *_args, **_kwargs: {
            "domain_id": "r_1",
            "claims": [],
            "sufficient": False,
        },
    )
    monkeypatch.setattr(
        research_pipeline,
        "_validate_document_grounding",
        lambda *_args, **_kwargs: None,
    )
    state = {
        "repository_candidates": [],
        "unresolved": [],
        "research_queue": [
            {
                "research_id": "r_1",
                "resolves": [],
                "objective": "research",
                "information_needed": "research",
                "source_kinds": ["reference_sources"],
                "queries": ["reference research"],
                "status": "pending",
            }
        ],
        "evidence": [],
        "resolved": [],
        "blockers": [],
        "state_sha256": "test",
    }
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="PLANNING_RESEARCH_TIMEOUT"):
        research.collect_planning_state_research(_Router(), "prompt", state)
    assert time.monotonic() - started < 0.20
''',
    encoding="utf-8",
)

print("planning deadline hardening patch applied", flush=True)
