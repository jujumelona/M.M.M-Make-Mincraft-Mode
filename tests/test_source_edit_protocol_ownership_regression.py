from __future__ import annotations

import hashlib

from minecraft_mod_ai import agent_tool_runtime
from minecraft_mod_ai import small_model_execution_extensions_contract as execution_extensions
from minecraft_mod_ai import source_edit_scalar_protocol_contract as scalar_protocol
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher


def _project(workspace):
    project = workspace / "demo"
    (project / "src/main/java/example").mkdir(parents=True)
    (project / "settings.gradle").write_text(
        'rootProject.name = "demo"\n', encoding="utf-8"
    )
    return project


def test_execution_extension_uses_scalar_protocol_as_single_schema_owner() -> None:
    assert execution_extensions._SOURCE_EDIT_SCHEMA is scalar_protocol.SOURCE_EDIT_SCHEMA

    properties = execution_extensions._SOURCE_EDIT_SCHEMA["properties"]
    assert "edits" not in properties
    assert execution_extensions._SOURCE_EDIT_SCHEMA["required"] == ["operation", "path"]
    assert set(properties["operation"]["enum"]) >= {
        "replace_exact",
        "insert_before",
        "insert_after",
        "create_file",
        "replace_file",
        "delete_file",
        "replace",
        "create",
        "delete",
    }


def test_scalar_protocol_install_validates_without_late_monkeypatch() -> None:
    schema_before = execution_extensions._SOURCE_EDIT_SCHEMA
    materializer_before = execution_extensions._materialize_model_source_edit

    scalar_protocol.install(execution_extensions, agent_tool_runtime)

    assert execution_extensions._SOURCE_EDIT_SCHEMA is schema_before
    assert execution_extensions._materialize_model_source_edit is materializer_before
    assert execution_extensions._SOURCE_EDIT_SCHEMA is scalar_protocol.SOURCE_EDIT_SCHEMA


def test_replace_file_materializes_transactional_replace_with_host_hash(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    before = "final class Example {}\n"
    after = "final class Example { int value; }\n"
    source.write_text(before, encoding="utf-8")

    payload = execution_extensions._materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "replace_file",
            "path": "src/main/java/example/Example.java",
            "content": after,
        },
    )

    assert payload["project_root"] == "demo"
    assert payload["operations"] == [
        {
            "operation": "replace",
            "path": "src/main/java/example/Example.java",
            "expected_sha256": "sha256:" + hashlib.sha256(before.encode()).hexdigest(),
            "content": after,
        }
    ]
    receipt = TransactionalSourcePatcher(project).apply(payload["operations"])
    assert receipt["status"] == "APPLIED"
    assert source.read_text(encoding="utf-8") == after


def test_delete_file_materializes_transactional_delete_with_host_hash(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Obsolete.java"
    before = "final class Obsolete {}\n"
    source.write_text(before, encoding="utf-8")

    payload = execution_extensions._materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "delete_file",
            "path": "src/main/java/example/Obsolete.java",
        },
    )

    assert payload["operations"] == [
        {
            "operation": "delete",
            "path": "src/main/java/example/Obsolete.java",
            "expected_sha256": "sha256:" + hashlib.sha256(before.encode()).hexdigest(),
        }
    ]
    receipt = TransactionalSourcePatcher(project).apply(payload["operations"])
    assert receipt["status"] == "APPLIED"
    assert not source.exists()


def test_whole_file_aliases_keep_single_scalar_protocol(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    before = "final class Example {}\n"
    source.write_text(before, encoding="utf-8")

    replace_payload = execution_extensions._materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "delete",
            "path": "src/main/java/example/Example.java",
        },
    )

    assert replace_payload["operations"][0]["operation"] == "delete"
