from __future__ import annotations

import json

import pytest

import minecraft_mod_ai.evidence_first_execution as execution


def test_local_observation_is_bounded_and_excludes_global_request() -> None:
    observation = execution.build_local_error_observation(
        task_id="task-1",
        action_id="action-1",
        failed_action={"tool": "apply_source_edit", "arguments": {"file": "A.java"}},
        json_pointer="/arguments/path",
        code="missing_required_key",
        message="path is required",
        allowed_keys=["path", "old_text", "new_text", "path"],
        local_context={"arguments": {"file": "A.java"}},
    )

    assert observation["task_id"] == "task-1"
    assert observation["action_id"] == "action-1"
    assert observation["error"]["json_pointer"] == "/arguments/path"
    assert observation["error"]["allowed_keys"] == ["new_text", "old_text", "path"]
    assert len(observation["failed_action_digest"]) == 64
    assert len(observation["observation_sha256"]) == 64
    assert "request" not in json.dumps(observation)

    with pytest.raises(execution.EvidenceExecutionError, match="global field"):
        execution.build_local_error_observation(
            task_id="task-1",
            action_id="action-1",
            failed_action={},
            json_pointer="/",
            code="bad",
            message="bad",
            local_context={"request_catalog": {"requirements": ["do not replay"]}},
        )


def test_local_repair_receives_only_observation_and_preserves_identity() -> None:
    seen = []

    def validate(action):
        if "path" not in action["arguments"]:
            raise ValueError("missing path")

    def describe(exc, action):
        return {
            "json_pointer": "/arguments/path",
            "code": "missing_required_key",
            "message": str(exc),
            "allowed_keys": ["path", "old_text", "new_text"],
            "local_context": {"arguments": dict(action["arguments"])},
        }

    def repair(observation):
        seen.append(observation)
        assert set(observation) == {
            "schema",
            "task_id",
            "action_id",
            "error",
            "local_context",
            "failed_action_digest",
            "observation_sha256",
        }
        return {
            "task_id": "task-1",
            "action_id": "action-1",
            "tool": "apply_source_edit",
            "arguments": {"path": "src/A.java", "old_text": "x", "new_text": "y"},
        }

    result = execution.run_local_action_repair_loop(
        task_id="task-1",
        action_id="action-1",
        initial_action={
            "task_id": "task-1",
            "action_id": "action-1",
            "tool": "apply_source_edit",
            "arguments": {"file": "src/A.java", "old_text": "x", "new_text": "y"},
        },
        validate_action=validate,
        execute_action=lambda action: {"edited": action["arguments"]["path"]},
        describe_error=describe,
        repair_action=repair,
        max_attempts=2,
    )

    assert result["attempts"] == 2
    assert result["result"] == {"edited": "src/A.java"}
    assert len(seen) == 1


def test_local_repair_cannot_change_task_or_action_identity() -> None:
    with pytest.raises(execution.EvidenceExecutionError, match="changed task_id"):
        execution.run_local_action_repair_loop(
            task_id="task-1",
            action_id="action-1",
            initial_action={"task_id": "task-1", "action_id": "action-1"},
            validate_action=lambda _action: (_ for _ in ()).throw(ValueError("bad")),
            execute_action=lambda _action: None,
            describe_error=lambda exc, _action: {
                "json_pointer": "/",
                "code": "bad",
                "message": str(exc),
            },
            repair_action=lambda _observation: {
                "task_id": "other-task",
                "action_id": "action-1",
            },
            max_attempts=2,
        )


def test_checkpoint_is_durable_and_resumable(tmp_path) -> None:
    path = tmp_path / "run-state.json"
    state = execution.EvidenceExecutionState.load_or_create(path, plan_sha256="plan-sha")
    state.checkpoint_success("task-a", {"result": "ok"})

    resumed = execution.EvidenceExecutionState.load_or_create(path, plan_sha256="plan-sha")
    assert resumed.completed_task_ids == ["task-a"]
    assert resumed.task_receipts == {"task-a": {"result": "ok"}}
    assert resumed.as_dict()["state_sha256"]

    with pytest.raises(execution.EvidenceExecutionError, match="different plan"):
        execution.EvidenceExecutionState.load_or_create(path, plan_sha256="other-plan")


def _tasks():
    return [
        {
            "task_id": "task-edit",
            "depends_on": [],
            "ownership": [{"kind": "source", "locator": "src/A.java"}],
        },
        {
            "task_id": "task-dependent",
            "depends_on": ["task-edit"],
            "ownership": [{"kind": "resource", "locator": "assets/a.json"}],
        },
        {
            "task_id": "task-transitive",
            "depends_on": ["task-dependent"],
            "ownership": [{"kind": "source", "locator": "src/C.java"}],
        },
        {
            "task_id": "task-unrelated",
            "depends_on": [],
            "ownership": [{"kind": "source", "locator": "src/Z.java"}],
        },
    ]


def test_changed_path_replans_downstream_only() -> None:
    impacted = execution.impacted_task_ids_for_paths(
        _tasks(),
        ["src/A.java"],
        completed_task_ids=["task-edit"],
    )
    assert impacted == ["task-dependent", "task-transitive"]


def test_success_checkpoint_refreshes_index_and_replans_only_impact(tmp_path) -> None:
    state = execution.EvidenceExecutionState.load_or_create(
        tmp_path / "run-state.json", plan_sha256="plan-sha"
    )
    received = []
    previous_index = {
        "files": {
            "src/A.java": "before",
            "src/Z.java": "unchanged",
        }
    }

    result = execution.checkpoint_refresh_and_replan(
        state=state,
        completed_task_id="task-edit",
        success_receipt={"edited": "src/A.java"},
        tasks=_tasks(),
        previous_index=previous_index,
        index_builder=lambda: {
            "files": {
                "src/A.java": "after",
                "src/Z.java": "unchanged",
            }
        },
        replanner=lambda tasks: received.extend(task["task_id"] for task in tasks) or "replanned",
    )

    assert result["impacted_task_ids"] == ["task-dependent", "task-transitive"]
    assert result["replan_result"] == "replanned"
    assert received == ["task-dependent", "task-transitive"]
    assert state.completed_task_ids == ["task-edit"]

    resumed = execution.EvidenceExecutionState.load_or_create(
        tmp_path / "run-state.json", plan_sha256="plan-sha"
    )
    assert resumed.completed_task_ids == ["task-edit"]
    assert resumed.impact_events[-1]["impacted_task_ids"] == [
        "task-dependent",
        "task-transitive",
    ]
    assert "task-unrelated" not in resumed.impact_events[-1]["impacted_task_ids"]
