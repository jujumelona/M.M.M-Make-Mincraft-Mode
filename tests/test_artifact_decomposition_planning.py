from __future__ import annotations

from copy import deepcopy
import pytest

from minecraft_mod_ai import planning_state_adaptive_implementation as adaptive
from minecraft_mod_ai.minecraft_template_steps import responsibility_ids_for_artifact


class _Router:
    pass


def _base_state(*, blockers=None, decisions=None) -> dict[str, object]:
    return {
        "decisions": list(decisions or []),
        "coverage": [],
        "unresolved": [],
        "blockers": list(blockers or []),
        "state_sha256": "test",
    }


def _fake_requirement_plan(requirement, requirement_ref, selected_sections, worksheet, allowed):
    del requirement, worksheet, allowed
    return {
        "requirement_ref": requirement_ref,
        "required_detail_sections": list(selected_sections),
        "engineering_worksheet": {},
        "worksheet_contract": "authored_concern_records",
        "implementation_capabilities": [],
        "implementation_obligations": [],
        "artifact_obligations": [],
        "grounded_bindings": [],
        "reuse_candidates": [],
        "verification_obligations": [],
    }


def _default_compile_criterion(
    _router,
    *,
    requirement_ref,
    criterion_index,
    **_kwargs,
):
    return {
        "requirement_ref": requirement_ref,
        "criterion_index": criterion_index,
        "section_updates": [],
    }


def _patch_compile_boundaries(monkeypatch, requirements):
    monkeypatch.setattr(adaptive, "validate_planning_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(adaptive, "_requirement_decisions", lambda _state: requirements)
    monkeypatch.setattr(
        adaptive,
        "_requirement_grounding",
        lambda _state, _requirement_ref: ([], set()),
    )
    monkeypatch.setattr(adaptive, "router_native_model_parallelism", lambda _router: 8)
    # These tests exercise artifact decomposition/checkpointing rather than the record
    # template transport. Acceptance work remains mandatory, so make that independent
    # boundary deterministic instead of relying on an empty fixture router.
    monkeypatch.setattr(adaptive, "_compile_criterion", _default_compile_criterion)
    monkeypatch.setattr(adaptive, "assemble_worksheet_from_fragments", lambda *args, **kwargs: {})
    monkeypatch.setattr(adaptive, "_assemble_requirement_plan", _fake_requirement_plan)


def test_requirement_with_minecraft_artifacts_decomposes_to_canonical_responsibility_steps(monkeypatch):
    requirements = [
        {
            "requirement_id": "req_ruby_sword",
            "statement": "Add a ruby sword item with custom durability and attack damage.",
            "structural_artifacts": ["item"],
            "acceptance": ["Item is registered and functional in game."],
        }
    ]
    _patch_compile_boundaries(monkeypatch, requirements)

    executed_artifact_steps: list[tuple[str, str, str]] = []

    def mock_compile_artifact_step(
        _router,
        *,
        requirement_ref,
        artifact_kind,
        step_id,
        **_kwargs,
    ):
        executed_artifact_steps.append((requirement_ref, artifact_kind, step_id))
        return {
            "template_id": step_id,
            "artifact_kind": artifact_kind,
            "step_name": step_id.rsplit("/", 1)[-1],
            "status": "PASS",
            "output": {"artifact_kind": artifact_kind, "responsibility": step_id.rsplit("/", 1)[-1]},
            "proof": {"passed": True, "predicate": f"Verified {step_id}"},
        }

    monkeypatch.setattr(adaptive, "_compile_artifact_step", mock_compile_artifact_step)

    checkpoints: list[dict[str, object]] = []
    final_state = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        _base_state(),
        checkpoint=lambda s: checkpoints.append(deepcopy(s)),
    )

    expected_steps = responsibility_ids_for_artifact("item")
    assert len(executed_artifact_steps) == len(expected_steps)
    assert set(s[2] for s in executed_artifact_steps) == set(expected_steps)

    assert final_state["plan_ready"] is True
    details = [
        d for d in final_state["decisions"]
        if d.get("decision_type") == "detailed_implementation_plan"
    ]
    assert len(details) == 1
    plan = details[0]
    assert plan["requirement_ref"] == "req_ruby_sword"
    assert plan["artifact_kinds"] == ["item"]
    assert "item" in plan["artifact_plans"]
    assert len(plan["artifact_plans"]["item"]["responsibility_steps"]) == len(expected_steps)
    assert plan["artifact_obligations"] == [{"kind": "item", "responsibility_steps": len(expected_steps)}]
    assert plan["acceptance_criteria_complete"] is True


def test_artifact_step_checkpointing_and_resumption_skips_completed_steps(monkeypatch):
    requirements = [
        {
            "requirement_id": "req_ruby_sword",
            "statement": "Add a ruby sword item with custom durability.",
            "structural_artifacts": ["item"],
            "acceptance": ["Item works."],
        }
    ]
    _patch_compile_boundaries(monkeypatch, requirements)

    expected_steps = list(responsibility_ids_for_artifact("item"))
    fail_on_index = 3
    calls: list[str] = []

    def failing_compile_artifact_step(
        _router,
        *,
        requirement_ref,
        artifact_kind,
        step_id,
        **_kwargs,
    ):
        calls.append(step_id)
        if len(calls) == fail_on_index + 1:
            raise ValueError(f"Simulated failure on {step_id}")
        return {
            "template_id": step_id,
            "artifact_kind": artifact_kind,
            "step_name": step_id.rsplit("/", 1)[-1],
            "status": "PASS",
            "output": {"artifact_kind": artifact_kind, "responsibility": step_id.rsplit("/", 1)[-1]},
            "proof": {"passed": True, "predicate": f"Verified {step_id}"},
        }

    monkeypatch.setattr(adaptive, "_compile_artifact_step", failing_compile_artifact_step)
    monkeypatch.setattr(adaptive, "router_native_model_parallelism", lambda _router: 1)

    checkpoints: list[dict[str, object]] = []
    with pytest.raises(RuntimeError, match="DETAILED_PLAN_BLOCKED"):
        adaptive.compile_progress_monotone_detailed_plans(
            _Router(),
            "prompt",
            _base_state(),
            checkpoint=lambda s: checkpoints.append(deepcopy(s)),
        )

    assert len(calls) == fail_on_index + 1
    last_checkpoint = checkpoints[-1]
    blockers = [
        b for b in last_checkpoint["blockers"]
        if b.get("stage") == "detailed_planning" and b.get("terminal") is True
    ]
    assert len(blockers) == 1
    failed_step = expected_steps[fail_on_index]
    assert blockers[0]["section"] == f"artifact:item:{failed_step}"

    last_checkpoint["blockers"] = []
    resume_calls: list[str] = []

    def successful_compile_artifact_step(
        _router,
        *,
        requirement_ref,
        artifact_kind,
        step_id,
        **_kwargs,
    ):
        resume_calls.append(step_id)
        return {
            "template_id": step_id,
            "artifact_kind": artifact_kind,
            "step_name": step_id.rsplit("/", 1)[-1],
            "status": "PASS",
            "output": {"artifact_kind": artifact_kind, "responsibility": step_id.rsplit("/", 1)[-1]},
            "proof": {"passed": True, "predicate": f"Verified {step_id}"},
        }

    monkeypatch.setattr(adaptive, "_compile_artifact_step", successful_compile_artifact_step)

    resumed_state = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        last_checkpoint,
    )

    assert resumed_state["plan_ready"] is True
    assert resume_calls == expected_steps[fail_on_index:]


def test_heterogeneous_batch_artifacts_and_fallback_criteria(monkeypatch):
    requirements = [
        {
            "requirement_id": "req_block",
            "statement": "Add a copper ore block.",
            "structural_artifacts": ["block"],
            "acceptance": ["Block places and drops."],
        },
        {
            "requirement_id": "req_logic",
            "statement": "Compute custom difficulty scaling formula.",
            "acceptance": ["Difficulty increases over days.", "Difficulty caps at 100."],
        },
    ]
    _patch_compile_boundaries(monkeypatch, requirements)

    artifact_calls: list[tuple[str, str, str]] = []
    criteria_calls: list[tuple[str, int]] = []

    def mock_compile_artifact_step(
        _router,
        *,
        requirement_ref,
        artifact_kind,
        step_id,
        **_kwargs,
    ):
        artifact_calls.append((requirement_ref, artifact_kind, step_id))
        return {
            "template_id": step_id,
            "artifact_kind": artifact_kind,
            "step_name": step_id.rsplit("/", 1)[-1],
            "status": "PASS",
            "output": {"artifact_kind": artifact_kind, "responsibility": step_id.rsplit("/", 1)[-1]},
            "proof": {"passed": True, "predicate": f"Verified {step_id}"},
        }

    def mock_compile_criterion(
        _router,
        *,
        requirement_ref,
        criterion_index,
        **_kwargs,
    ):
        criteria_calls.append((requirement_ref, criterion_index))
        return {
            "requirement_ref": requirement_ref,
            "criterion_index": criterion_index,
            "section_updates": [],
        }

    monkeypatch.setattr(adaptive, "_compile_artifact_step", mock_compile_artifact_step)
    monkeypatch.setattr(adaptive, "_compile_criterion", mock_compile_criterion)

    state = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        _base_state(),
    )

    assert state["plan_ready"] is True
    block_steps = responsibility_ids_for_artifact("block")
    assert len(artifact_calls) == len(block_steps)
    assert all(c[0] == "req_block" for c in artifact_calls)

    # Artifact planning and acceptance planning are independent obligations. The block
    # requirement therefore contributes one criterion in addition to the two logic ones.
    assert set(criteria_calls) == {
        ("req_block", 0),
        ("req_logic", 0),
        ("req_logic", 1),
    }
