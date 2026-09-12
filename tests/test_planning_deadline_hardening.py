from __future__ import annotations

from hashlib import sha256
import json
import time
from threading import Thread
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
    bind_model_execution_deadline,
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


def test_model_capacity_wait_inherits_work_unit_deadline() -> None:
    gate = ReentrantCapacityGate(lambda: 1)
    gate.acquire()
    errors: list[BaseException] = []

    def contend() -> None:
        try:
            with bind_model_execution_deadline(time.monotonic() + 0.03):
                gate.acquire()
        except BaseException as exc:
            errors.append(exc)

    contender = Thread(target=contend)
    started = time.monotonic()
    try:
        contender.start()
        contender.join(timeout=0.20)
        assert not contender.is_alive()
        assert time.monotonic() - started < 0.20
        assert len(errors) == 1
        assert isinstance(errors[0], ModelExecutionDeadlineExceeded)
    finally:
        gate.release()
        contender.join(timeout=0.20)


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
