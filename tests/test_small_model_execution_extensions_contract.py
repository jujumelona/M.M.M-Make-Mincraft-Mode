from __future__ import annotations

import hashlib

import pytest

from minecraft_mod_ai import agent_tool_runtime
from minecraft_mod_ai.small_model_execution_extensions_contract import _compose_skills
from minecraft_mod_ai.source_edit_scalar_protocol_contract import (
    SOURCE_EDIT_SCHEMA,
    materialize_model_source_edit,
)
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher


def _project(workspace):
    project = workspace / "demo"
    (project / "src/main/java/example").mkdir(parents=True)
    (project / "settings.gradle").write_text('rootProject.name = "demo"\n', encoding="utf-8")
    return project


def _skill(identity: str, *, requires=(), provides=(), confidence=1.0):
    return {
        "skill_id": identity,
        "name": identity,
        "activate_when": [identity],
        "steps": [f"run {identity}"],
        "constraints": [],
        "requires": list(requires),
        "provides": list(provides),
        "confidence": confidence,
    }


def test_source_edit_schema_is_scalar_semantic_action() -> None:
    assert SOURCE_EDIT_SCHEMA["required"] == ["operation", "path"]
    properties = SOURCE_EDIT_SCHEMA["properties"]
    assert "edits" not in properties
    assert all(value.get("type") != "array" for value in properties.values())
    operations = set(properties["operation"]["enum"])
    assert {
        "replace_exact",
        "insert_before",
        "insert_after",
        "create_file",
        "create_java_type",
        "add_java_import",
        "insert_java_member",
        "delete_file",
    } <= operations
    assert "append_file" not in operations
    assert "replace_file" not in operations
    assert "maxLength" not in properties["new"]
    assert "maxLength" not in properties["content"]


def test_partial_source_edit_materializes_one_exact_edit_and_host_hash(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    old = "final class Example {\n    int oldValue;\n}\n"
    source.write_text(old, encoding="utf-8")
    original_bytes = source.read_bytes()

    payload = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "replace_exact",
            "path": "src/main/java/example/Example.java",
            "old": "oldValue",
            "new": "newValue",
        },
    )

    assert payload["project_root"] == "demo"
    operation = payload["operations"][0]
    assert operation["operation"] == "edit"
    assert operation["expected_sha256"] == "sha256:" + hashlib.sha256(original_bytes).hexdigest()
    assert operation["replacements"] == [
        {"old": "oldValue", "new": "newValue", "count": 1}
    ]

    receipt = TransactionalSourcePatcher(project).apply(payload["operations"])
    assert receipt["status"] == "APPLIED"
    assert "newValue" in source.read_text(encoding="utf-8")


def test_scalar_source_write_materializes_new_file_and_host_applies(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    target = project / "src/main/java/example/Created.java"
    content = "package example;\n\nfinal class Created {\n}\n"

    payload = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "create_java_type",
            "path": "src/main/java/example/Created.java",
            "package_name": "example",
            "declaration": "final class Created",
        },
    )

    assert payload["operations"] == [
        {
            "operation": "create",
            "path": "src/main/java/example/Created.java",
            "content": content,
        }
    ]
    receipt = TransactionalSourcePatcher(project).apply(payload["operations"])
    assert receipt["status"] == "APPLIED"
    assert target.read_text(encoding="utf-8") == content


def test_scalar_source_write_accepts_lossless_create_alias(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace)

    payload = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "create",
            "path": "src/main/resources/demo.txt",
            "content": "created\n",
        },
    )

    assert payload["operations"][0]["operation"] == "create"


def test_scalar_source_create_rejects_existing_target_without_mutation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    before = "final class Example {}\n"
    source.write_text(before, encoding="utf-8")

    with pytest.raises(agent_tool_runtime.AgentToolRuntimeError, match="already exists"):
        materialize_model_source_edit(
            agent_tool_runtime,
            workspace,
            {
                "operation": "create_java_type",
                "path": "src/main/java/example/Example.java",
                "package_name": "example",
                "declaration": "final class Example",
            },
        )
    assert source.read_text(encoding="utf-8") == before


def test_partial_source_edit_sequences_changes_across_turns(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    source.write_text("final class Example {\n    int oldValue;\n}\n", encoding="utf-8")

    first = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "replace_exact",
            "path": "src/main/java/example/Example.java",
            "old": "oldValue",
            "new": "newValue",
        },
    )
    TransactionalSourcePatcher(project).apply(first["operations"])

    second = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "insert_before",
            "path": "src/main/java/example/Example.java",
            "anchor": "}\n",
            "content": "    void run() {}\n",
        },
    )
    TransactionalSourcePatcher(project).apply(second["operations"])

    assert source.read_text(encoding="utf-8") == (
        "final class Example {\n    int newValue;\n    void run() {}\n}\n"
    )


def test_partial_source_edit_rejects_ambiguous_anchor_without_mutation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    before = "token token\n"
    source.write_text(before, encoding="utf-8")

    with pytest.raises(agent_tool_runtime.AgentToolRuntimeError, match="expected 1 matches, found 2"):
        materialize_model_source_edit(
            agent_tool_runtime,
            workspace,
            {
                "operation": "insert_after",
                "path": "src/main/java/example/Example.java",
                "anchor": "token",
                "content": "!",
            },
        )
    assert source.read_text(encoding="utf-8") == before


def test_scoped_materialization_failure_reports_unchanged_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    before = "token token\n"
    source.write_text(before, encoding="utf-8")
    runtime = agent_tool_runtime.AgentToolRuntime(profile="test", workspace_root=workspace)
    runtime.tool_schemas = lambda _stage: (
        {
            "type": "function",
            "function": {
                "name": "apply_source_edit",
                "parameters": SOURCE_EDIT_SCHEMA,
            },
        },
    )
    runtime._allowed_tool_cache["generation"] = frozenset({"apply_source_edit"})

    with pytest.raises(
        agent_tool_runtime.AgentToolRuntimeError,
        match=r"expected 1 matches, found 2.*\[workspace_impact=unchanged\]",
    ):
        runtime.call_scoped(
            "generation",
            "apply_source_edit",
            {
                "operation": "replace_exact",
                "path": "src/main/java/example/Example.java",
                "old": "token",
                "new": "updated",
            },
            external_server_ids=(),
        )

    assert source.read_text(encoding="utf-8") == before


def test_ordered_skill_composition_adds_provider_before_consumer() -> None:
    provider = _skill("provider", provides=["compiled registry"])
    consumer = _skill("consumer", requires=["compiled registry"])

    result = _compose_skills("consumer", [consumer, provider], [consumer])

    assert [item["skill_id"] for item in result["ordered_skills"]] == ["provider", "consumer"]
    assert result["unresolved_requirements"] == []
    assert result["cycles"] == []
    assert result["composition_policy"] == "explicit_requires_provides_only"


def test_ordered_skill_composition_blocks_unresolved_requirement() -> None:
    consumer = _skill("consumer", requires=["missing capability"])

    result = _compose_skills("consumer", [consumer], [consumer])

    assert result["ordered_skills"] == []
    assert result["unresolved_requirements"] == [
        {"skill_id": "consumer", "requirement": "missing capability"}
    ]
    assert result["blocked_skill_ids"] == ["consumer"]


def test_ordered_skill_composition_detects_cycle_and_does_not_execute_it() -> None:
    first = _skill("first", requires=["b"], provides=["a"])
    second = _skill("second", requires=["a"], provides=["b"])

    result = _compose_skills("first", [first, second], [first])

    assert result["ordered_skills"] == []
    assert result["cycles"] == [["first", "second"]]
    assert result["blocked_skill_ids"] == ["first", "second"]


def test_skill_composition_does_not_infer_dependencies_from_similar_words() -> None:
    first = _skill("registry producer", provides=["registry output"])
    second = _skill("registry consumer")

    result = _compose_skills("registry consumer", [first, second], [second])

    assert [item["skill_id"] for item in result["ordered_skills"]] == ["registry consumer"]
    assert result["dependency_edges"] == []
