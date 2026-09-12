from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import root_cause_trace
from minecraft_mod_ai import runner_parallel_validation_contract as gradle_contract
from minecraft_mod_ai import task_template_runner


class _AtomicRouter:
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
        **kwargs,
    ):
        del role, parameters, description, kwargs
        context = json.loads(messages[-1]["content"])
        self.calls.append((tool_name, context))
        if tool_name == "submit_one_design_content_property":
            requested = str(context["requested_property"])
            assert context["allowed_properties"] == [requested]
            return {"property": requested, "value": f"value_{requested}"}
        if tool_name == "submit_one_design_content_relation_count":
            return {"count": 0}
        raise AssertionError(f"unexpected tool call: {tool_name}")


def test_content_properties_are_one_property_atomic_calls():
    router = _AtomicRouter()
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
    assert len(router.calls) == len(properties)
    assert all(name == "submit_one_design_content_property" for name, _ in router.calls)
    assert [call[1]["requested_property"] for call in router.calls] == properties


def test_relations_use_host_owned_pair_cardinality_without_model_continuation():
    router = _AtomicRouter()
    entity_ids = ["alpha", "beta", "gamma", "delta", "epsilon"]

    result = task_template_runner.run_record_template(
        router,
        "design/content_relation",
        context={"entity_ids": entity_ids, "requirement": "No explicit relations."},
        allowed_refs=set(),
    )

    assert result["records"] == []
    expected_pairs = len(entity_ids) * (len(entity_ids) - 1)
    assert len(router.calls) == expected_pairs
    assert all(name == "submit_one_design_content_relation_count" for name, _ in router.calls)
    assert all(call[1]["source_id"] != call[1]["target_id"] for call in router.calls)


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
