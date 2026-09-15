from __future__ import annotations

import minecraft_mod_ai.planning_state_pipeline as pipeline


def test_observability_failure_cannot_change_planning_control_flow(monkeypatch) -> None:
    def broken_observer(*args, **kwargs):
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr(pipeline, "emit_root_cause", broken_observer)

    pipeline._observe("test_event", result="CONTINUE")
    result = pipeline._transition(
        "test_transition",
        lambda: {"plan_ready": False},
        input_state={"plan_ready": False},
    )

    assert result == {"plan_ready": False}


def test_goal_observability_failure_is_fail_open(monkeypatch) -> None:
    def broken_goal_observer(*args, **kwargs):
        raise RuntimeError("goal telemetry unavailable")

    monkeypatch.setattr(pipeline, "emit_planning_goal_satisfied", broken_goal_observer)

    pipeline._observe_goal_satisfied({"plan_ready": True})


def test_detailed_planning_exception_without_progress_stays_pending(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    checkpoints: list[dict] = []
    initial = {"plan_ready": False, "decisions": []}

    monkeypatch.setattr(
        pipeline,
        "emit_root_cause",
        lambda event, **fields: events.append((event, fields)),
    )

    def fail_without_progress(*args, **kwargs):
        raise RuntimeError("recoverable model failure")

    monkeypatch.setattr(
        pipeline,
        "compile_progress_monotone_detailed_plans",
        fail_without_progress,
    )

    result = pipeline._compile_detailed_plans_resumable(
        object(),
        "prompt",
        initial,
        {},
        checkpoints.append,
    )

    assert result["decisions"] == initial["decisions"]
    assert result["generation_interruption"]["reason"] == "recoverable model failure"
    assert result["plan_ready"] is False
    assert checkpoints[-1]["plan_ready"] is False
    pending = [fields for event, fields in events if event == "detailed_planning_pending"]
    assert pending
    assert pending[-1]["result"] == "RESUMABLE"
    assert not any(fields.get("result") in {"FAIL", "ERROR", "BLOCKED"} for _, fields in events)


def test_detailed_planning_not_ready_result_stays_pending(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    checkpoints: list[dict] = []
    pending_result = {
        "plan_ready": False,
        "decisions": [],
        "detail_progress": [],
    }

    monkeypatch.setattr(
        pipeline,
        "emit_root_cause",
        lambda event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(
        pipeline,
        "compile_progress_monotone_detailed_plans",
        lambda *args, **kwargs: pending_result,
    )

    result = pipeline._compile_detailed_plans_resumable(
        object(),
        "prompt",
        {"plan_ready": False, "decisions": []},
        {},
        checkpoints.append,
    )

    assert result is pending_result
    assert result["plan_ready"] is False
    assert checkpoints[-1]["plan_ready"] is False
    pending = [fields for event, fields in events if event == "detailed_planning_pending"]
    assert pending
    assert pending[-1]["result"] == "RESUMABLE"
    assert not any(fields.get("result") in {"FAIL", "ERROR", "BLOCKED"} for _, fields in events)


def test_detailed_planning_progress_is_requeued_until_ready(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    checkpoints: list[dict] = []
    calls = 0
    progressed = {
        "plan_ready": False,
        "decisions": [],
        "detail_progress": [{"requirement_ref": "req_001", "criterion_index": 0}],
    }
    ready = {
        "plan_ready": True,
        "decisions": [],
        "detail_progress": [{"requirement_ref": "req_001", "criterion_index": 0}],
    }

    monkeypatch.setattr(
        pipeline,
        "emit_root_cause",
        lambda event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(pipeline, "emit_planning_goal_satisfied", lambda *args, **kwargs: None)

    def compile_until_ready(*args, **kwargs):
        nonlocal calls
        calls += 1
        return progressed if calls == 1 else ready

    monkeypatch.setattr(
        pipeline,
        "compile_progress_monotone_detailed_plans",
        compile_until_ready,
    )

    result = pipeline._compile_detailed_plans_resumable(
        object(),
        "prompt",
        {"plan_ready": False, "decisions": []},
        {},
        checkpoints.append,
    )

    assert result is ready
    assert calls == 2
    resumed = [
        fields
        for event, fields in events
        if event == "detailed_planning_resume_after_progress"
    ]
    assert resumed
    assert resumed[-1]["result"] == "CONTINUE"
    assert not any(fields.get("result") in {"FAIL", "ERROR", "BLOCKED"} for _, fields in events)


def test_detailed_planning_ready_result_remains_truthful(monkeypatch) -> None:
    ready_result = {"plan_ready": True, "decisions": []}
    checkpoints: list[dict] = []

    monkeypatch.setattr(
        pipeline,
        "compile_progress_monotone_detailed_plans",
        lambda *args, **kwargs: ready_result,
    )
    monkeypatch.setattr(pipeline, "emit_root_cause", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "emit_planning_goal_satisfied", lambda *args, **kwargs: None)

    result = pipeline._compile_detailed_plans_resumable(
        object(),
        "prompt",
        {"plan_ready": False, "decisions": []},
        {},
        checkpoints.append,
    )

    assert result is ready_result
    assert result["plan_ready"] is True
    assert checkpoints[-1]["plan_ready"] is True
