from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import root_cause_trace
from minecraft_mod_ai import runner_parallel_validation_contract as gradle_contract
from minecraft_mod_ai import task_template_runner


class _AtomicRouter:
    """Replay current planner JSON transport rather than deprecated tool calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.registry = SimpleNamespace(
            role=lambda *_args, **_kwargs: SimpleNamespace(adapter="llama_cpp")
        )
        self.profile = "test"

    def generate_text(self, role, messages, *, response_schema, **kwargs):
        assert role == "planner"
        assert kwargs.get("enable_tools") is False
        assert kwargs.get("force_non_thinking") is True
        user_message = next(
            (message["content"] for message in messages if message.get("role") == "user"),
            None,
        )
        assert user_message is not None
        context = json.loads(str(user_message))
        self.calls.append(("planner-json", context))
        fields = response_schema["properties"]
        if "property" in fields or "value" in fields:
            requested = str(context["requested_property"])
            assert context["allowed_properties"] == [requested]
            # Current single_record_template projects one logical record into
            # two independent finite schema pages, never a combined JSON call.
            assert len(fields) == 1
            if "property" in fields:
                return json.dumps({"property": requested})
            values = {
                "display_name": "Test Entity",
                "category": "monster",
                "health": "20",
                "speed": "0.25",
                "tracking_range": "16",
                "width": "0.6",
                "height": "1.8",
            }
            chosen = values[requested]
            enum = fields["value"].get("enum")
            if enum is not None:
                chosen = enum[0]
            return json.dumps({"value": chosen})
        if "count" in fields:
            assert "source_entity" in context
            assert "target_entity" in context
            assert 0 in fields["count"]["enum"]
            return '{"count":0}'
        raise AssertionError(f"unexpected schema: {response_schema}")


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
    assert len(router.calls) == 2 * len(properties)
    assert all(name == "planner-json" for name, _ in router.calls)
    observed = [context["requested_property"] for _, context in router.calls]
    assert observed == [name for name in properties for _ in range(2)]


def test_relations_use_host_owned_pair_cardinality_without_model_continuation():
    router = _AtomicRouter()
    entity_ids = ["alpha", "beta", "gamma", "delta", "epsilon"]
    entities = [{"entity_id": name, "kind": "item"} for name in entity_ids]

    result = task_template_runner.run_record_template(
        router,
        "design/content_relation",
        context={
            "entity_ids": entity_ids,
            "entities": entities,
            "requirement": "No explicit relations.",
        },
        allowed_refs=set(),
    )

    assert result["records"] == []
    expected_pairs = len(entity_ids) * (len(entity_ids) - 1)
    assert len(router.calls) == expected_pairs
    assert all(name == "planner-json" for name, _ in router.calls)
    observed_pairs = {
        (call["source_entity"]["entity_id"], call["target_entity"]["entity_id"])
        for _, call in router.calls
    }
    expected = {(source, target) for source in entity_ids for target in entity_ids if source != target}
    assert observed_pairs == expected


def test_gradle_hot_path_is_incremental_parallel_and_process_isolated(monkeypatch):
    monkeypatch.delenv("MMM_GRADLE_WORKERS", raising=False)
    monkeypatch.setattr(gradle_contract.os, "cpu_count", lambda: 8)

    args = gradle_contract._gradle_execution_arguments("build")

    assert "clean" not in args
    assert "--no-daemon" in args
    assert "--daemon" not in args
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
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_ARTIFACTS", "all")
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
