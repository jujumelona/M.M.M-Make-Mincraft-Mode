from __future__ import annotations

import hashlib

import pytest

from minecraft_mod_ai import agent_tool_runtime
from minecraft_mod_ai import source_patch as source_patch_module
from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime, AgentToolRuntimeError
from minecraft_mod_ai.source_edit_scalar_protocol_contract import (
    SOURCE_EDIT_SCHEMA,
    materialize_model_source_edit,
)
from minecraft_mod_ai.source_mutation_contract import mutation_payload_applied
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher


def _project(workspace, name: str = "demo"):
    project = workspace / name
    (project / "src/main/java/example").mkdir(parents=True)
    (project / "settings.gradle").write_text('rootProject.name = "demo"\n', encoding="utf-8")
    return project


def _source_edit_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "semantic edit",
            "parameters": SOURCE_EDIT_SCHEMA,
        },
    }


def test_generation_tool_schema_hides_raw_patch_and_exposes_one_semantic_edit(monkeypatch, tmp_path) -> None:
    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)
    raw_patch = {
        "name": "apply_source_patch",
        "description": "host transaction primitive",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_root": {"type": "string"},
                "operations": {"type": "array"},
            },
            "required": ["project_root", "operations"],
        },
    }
    monkeypatch.setattr(runtime, "_run_async", lambda *_args: [raw_patch])
    monkeypatch.setattr(runtime._external_bridge, "tool_schemas", lambda _stage: ())

    exposed = runtime.tool_schemas("generation")
    names = [item["function"]["name"] for item in exposed]

    assert "apply_source_patch" not in names
    assert names.count("apply_source_edit") == 1
    source_edit = next(item for item in exposed if item["function"]["name"] == "apply_source_edit")
    assert source_edit["function"]["parameters"] == SOURCE_EDIT_SCHEMA


def test_host_stage_call_preserves_raw_strict_patch_contract(monkeypatch, tmp_path) -> None:
    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)
    captured: dict[str, object] = {}

    def fake_run_async(_function, *args):
        captured["args"] = args
        return {"structured_content": {"ok": True}}

    monkeypatch.setattr(runtime, "_run_async", fake_run_async)
    raw = {
        "project_root": "demo",
        "operations": [
            {
                "operation": "create",
                "path": "src/main/java/example/Example.java",
                "content": "package example;\nfinal class Example {}\n",
            }
        ],
    }

    runtime.call("generation", "apply_source_patch", raw)

    assert captured["args"] == ("generation", "apply_source_patch", raw)


def test_model_scoped_call_rejects_host_only_patch_protocol(tmp_path) -> None:
    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)

    with pytest.raises(AgentToolRuntimeError, match="not model-callable"):
        runtime.call_scoped(
            "generation",
            "apply_source_patch",
            {"project_root": "demo", "operations": []},
            external_server_ids=(),
        )


def test_model_scoped_source_edit_materializes_internal_patch(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    source.write_text("final class Example { int oldValue; }\n", encoding="utf-8")
    before = source.read_bytes()
    runtime = AgentToolRuntime(profile="test", workspace_root=workspace)
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime, "tool_schemas", lambda _stage: (_source_edit_tool(),))
    runtime._allowed_tool_cache["generation"] = frozenset({"apply_source_edit"})

    def fake_run_async(_function, *args):
        captured["args"] = args
        return {"structured_content": {"ok": True}}

    monkeypatch.setattr(runtime, "_run_async", fake_run_async)

    runtime.call_scoped(
        "generation",
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": "src/main/java/example/Example.java",
            "old": "oldValue",
            "new": "newValue",
        },
        external_server_ids=(),
    )

    stage, name, payload = captured["args"]
    assert stage == "generation"
    assert name == "apply_source_patch"
    assert payload["project_root"] == "demo"
    assert payload["operations"] == [
        {
            "operation": "edit",
            "path": "src/main/java/example/Example.java",
            "expected_sha256": "sha256:" + hashlib.sha256(before).hexdigest(),
            "replacements": [{"old": "oldValue", "new": "newValue", "count": 1}],
        }
    ]


def test_host_materializes_unique_span_with_exact_sha(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    old = "package example;\nfinal class Example {}\n"
    source.write_bytes(old.encode("utf-8"))

    payload = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "replace_exact",
            "path": "src/main/java/example/Example.java",
            "old": "final class Example {}",
            "new": "final class Example { int value = 1; }",
        },
    )

    operation = payload["operations"][0]
    assert operation["operation"] == "edit"
    assert operation["expected_sha256"] == "sha256:" + hashlib.sha256(old.encode("utf-8")).hexdigest()
    receipt = TransactionalSourcePatcher(project).apply(payload["operations"])
    assert receipt["status"] == "APPLIED"
    assert receipt["changed_paths"] == ["src/main/java/example/Example.java"]
    assert mutation_payload_applied("apply_source_patch", {"ok": True, "result": receipt}) is True
    assert "int value = 1" in source.read_text(encoding="utf-8")


def test_noop_patch_skips_commit_and_does_not_prove_source_diff(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    content = "final class Example { int value = 1; }\n"
    source.write_text(content, encoding="utf-8")
    expected = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
    committed: list[object] = []

    monkeypatch.setattr(
        source_patch_module,
        "_commit_staged_path",
        lambda *args: committed.append(args),
    )

    receipt = TransactionalSourcePatcher(project).apply(
        [
            {
                "operation": "replace",
                "path": "src/main/java/example/Example.java",
                "expected_sha256": expected,
                "content": content,
            }
        ]
    )

    assert receipt["status"] == "UNCHANGED"
    assert receipt["changed_paths"] == []
    assert committed == []
    assert mutation_payload_applied(
        "apply_source_patch",
        {
            "ok": True,
            "result": receipt,
            "_mmm_source_mutation": {
                "tool": "apply_source_patch",
                "status": "APPLIED_BY_HOST_RUNTIME",
            },
        },
    ) is False


def test_source_edit_cannot_target_gradle_or_build_infrastructure(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace)

    with pytest.raises(AgentToolRuntimeError, match="limited to src/main/java"):
        materialize_model_source_edit(
            agent_tool_runtime,
            workspace,
            {
                "operation": "create_file",
                "path": "gradle/wrapper/gradle-wrapper.properties",
                "content": "distributionUrl=should-never-be-model-owned\n",
            },
        )


def test_host_refuses_ambiguous_project_root_instead_of_asking_model(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace, "one")
    _project(workspace, "two")

    with pytest.raises(AgentToolRuntimeError, match="exactly one source project"):
        materialize_model_source_edit(
            agent_tool_runtime,
            workspace,
            {
                "operation": "create_file",
                "path": "src/main/java/example/Example.java",
                "content": "final class Example {}\n",
            },
        )
