from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import root_cause_trace
from minecraft_mod_ai import runner_parallel_validation_contract as gradle_contract
from minecraft_mod_ai import task_template_runner


class _BatchRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.registry = SimpleNamespace(role=lambda *_args, **_kwargs: SimpleNamespace(adapter="llama_cpp"))
        self.profile = "test"

    def generate_tool_decision(
        self,
        role,
        messages,
        *,
        tool_name,
        parameters,
        description="",
    ):
        del role, messages, description
        self.calls.append((tool_name, dict(parameters)))
        if tool_name == "submit_content_property_batch":
            return {
                name: f"value_{name}"
                for name in parameters["properties"]
            }
        if tool_name == "submit_content_relation_batch":
            return {"target_ids": [], "relation_types": [], "overflow": False}
        raise AssertionError(f"unexpected tool call: {tool_name}")


def test_content_properties_use_three_field_atomic_batches():
    router = _BatchRouter()
    properties = [
        "display_name",
        "category",
        "health",
        "speed",
        "tracking_range",
        "width",
        "height",
    ]

    result = task_template_runner.run_record_template(
        router,
        "design/content_property",
        context={
            "allowed_properties": properties,
            "required_properties": properties,
        },
        allowed_refs=set(),
    )

    assert [row["property"] for row in result["records"]] == properties
    assert len(router.calls) == 3
    assert all(name == "submit_content_property_batch" for name, _ in router.calls)
    assert [len(schema["properties"]) for _, schema in router.calls] == [3, 3, 1]


def test_relations_are_selected_per_source_not_every_ordered_pair():
    router = _BatchRouter()
    entity_ids = ["alpha", "beta", "gamma", "delta", "epsilon"]

    result = task_template_runner.run_record_template(
        router,
        "design/content_relation",
        context={"entity_ids": entity_ids, "requirement": "No explicit relations."},
        allowed_refs=set(),
    )

    assert result["records"] == []
    assert len(router.calls) == len(entity_ids)
    assert len(router.calls) < len(entity_ids) * (len(entity_ids) - 1)
    assert all(name == "submit_content_relation_batch" for name, _ in router.calls)


def test_gradle_hot_path_is_incremental_parallel_and_daemon_reused(monkeypatch):
    monkeypatch.delenv("MMM_GRADLE_WORKERS", raising=False)
    monkeypatch.setattr(gradle_contract.os, "cpu_count", lambda: 8)

    args = gradle_contract._gradle_execution_arguments("build")

    assert "clean" not in args
    assert "--no-daemon" not in args
    assert "--daemon" in args
    assert "--parallel" in args
    assert "--max-workers=7" in args
    assert "--build-cache" in args


def test_trace_artifacts_fsync_only_on_failure(monkeypatch, tmp_path):
    artifact_sync: list[bool] = []
    journal_sync: list[bool] = []

    def fake_save(value, directory, *, sync=True):
        del value, directory
        artifact_sync.append(bool(sync))
        return {"path": "trace.json", "sha256": "x", "bytes": 1}

    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr(
        "minecraft_mod_ai.planner_trace_artifacts.save_trace_artifact",
        fake_save,
    )
    monkeypatch.setattr(
        root_cause_trace,
        "_append_durable_line",
        lambda _line, *, sync: journal_sync.append(bool(sync)),
    )
    monkeypatch.setattr(root_cause_trace, "_stderr_line", lambda *_args, **_kwargs: None)

    root_cause_trace.emit_root_cause(
        "success",
        result="PASS",
        details={"payload": "ok"},
    )
    root_cause_trace.emit_root_cause(
        "failure",
        result="FAIL",
        details={"payload": "bad"},
    )

    assert artifact_sync == [False, True]
    assert journal_sync == [False, True]
