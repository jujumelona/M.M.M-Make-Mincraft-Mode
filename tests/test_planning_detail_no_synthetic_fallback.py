from __future__ import annotations

from copy import deepcopy
import inspect

import pytest

import minecraft_mod_ai.planning_state_pipeline as pipeline


def _state() -> dict:
    return {
        "plan_ready": False,
        "state_sha256": "sha256:test",
        "decisions": [],
        "detail_progress": [],
        "artifact_progress": {},
        "template_progress": {},
    }


def _silence_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "emit_root_cause", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "emit_planning_goal_satisfied", lambda *args, **kwargs: None)


def test_detailed_planning_resumes_only_after_durable_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    _silence_trace(monkeypatch)
    calls: list[dict] = []

    def compile_stub(router, prompt, state, *, required_sections_by_requirement, checkpoint):
        del router, prompt, required_sections_by_requirement
        calls.append(deepcopy(state))
        if len(calls) == 1:
            progressed = deepcopy(state)
            progressed["template_progress"] = {"binding-1": {"field": "validated"}}
            checkpoint(progressed)
            raise ValueError("structured transport interrupted after a validated checkpoint")
        completed = deepcopy(state)
        completed["plan_ready"] = True
        return completed

    monkeypatch.setattr(pipeline, "compile_progress_monotone_detailed_plans", compile_stub)

    checkpoints: list[dict] = []
    result = pipeline._compile_detailed_plans_resumable(
        object(),
        "prompt",
        _state(),
        {},
        lambda value: checkpoints.append(deepcopy(value)),
    )

    assert result["plan_ready"] is True
    assert len(calls) == 2
    assert "binding-1" in calls[1]["template_progress"]
    assert any("binding-1" in row.get("template_progress", {}) for row in checkpoints)


def test_detailed_planning_never_retries_without_new_obligation_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trace(monkeypatch)
    calls = 0

    def compile_stub(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("same invalid generation")

    monkeypatch.setattr(pipeline, "compile_progress_monotone_detailed_plans", compile_stub)

    with pytest.raises(RuntimeError, match="DETAILED_PLAN_RUNTIME_STALLED"):
        pipeline._compile_detailed_plans_resumable(object(), "prompt", _state(), {}, None)

    assert calls == 1


def test_completed_requirement_is_monotone_even_when_criterion_checkpoint_is_cleared() -> None:
    before = _state()
    before["detail_progress"] = [
        {"requirement_ref": "req_001", "criterion_index": 0}
    ]
    after = deepcopy(before)
    after["detail_progress"] = []
    after["decisions"] = [
        {
            "decision_type": "detailed_implementation_plan",
            "requirement_ref": "req_001",
        }
    ]

    assert pipeline._detail_progress_strictly_advanced(
        pipeline._detail_progress_position(before),
        pipeline._detail_progress_position(after),
    )


def test_template_content_rewrite_without_new_binding_is_not_progress() -> None:
    before = _state()
    before["template_progress"] = {"binding-1": {"field": "first"}}
    after = deepcopy(before)
    after["template_progress"] = {"binding-1": {"field": "different"}}

    assert not pipeline._detail_progress_strictly_advanced(
        pipeline._detail_progress_position(before),
        pipeline._detail_progress_position(after),
    )


def test_non_ready_result_cannot_be_promoted_by_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _silence_trace(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "compile_progress_monotone_detailed_plans",
        lambda *args, **kwargs: _state(),
    )

    with pytest.raises(RuntimeError, match="DETAILED_PLAN_NOT_READY"):
        pipeline._compile_detailed_plans_resumable(object(), "prompt", _state(), {}, None)


def test_synthetic_detailed_plan_fallback_surface_is_absent() -> None:
    source = inspect.getsource(pipeline)
    assert "_host_complete_detailed_plans" not in source
    assert "_host_record" not in source
    assert "requirement_text} | {section} | {concern}" not in source
