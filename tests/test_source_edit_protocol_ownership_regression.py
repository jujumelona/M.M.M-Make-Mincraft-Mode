from __future__ import annotations

import hashlib

from minecraft_mod_ai import agent_tool_runtime
from minecraft_mod_ai import source_edit_scalar_protocol_contract as scalar_protocol
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher


def _project(workspace):
    project = workspace / "demo"
    (project / "src/main/java/example").mkdir(parents=True)
    (project / "settings.gradle").write_text(
        'rootProject.name = "demo"\n', encoding="utf-8"
    )
    return project


def test_scalar_protocol_is_the_single_source_edit_schema_owner() -> None:
    schema = scalar_protocol.SOURCE_EDIT_SCHEMA
    properties = schema["properties"]

    assert agent_tool_runtime.SOURCE_EDIT_SCHEMA is schema
    assert "edits" not in properties
    assert schema["required"] == ["operation", "path"]
    assert set(properties["operation"]["enum"]) == {
        "replace_exact",
        "insert_before",
        "insert_after",
        "create_file",
        "delete_file",
        "replace",
        "create",
        "delete",
    }
    assert "replace_file" not in properties["operation"]["enum"]
    assert "append_file" not in properties["operation"]["enum"]


def test_agent_runtime_uses_canonical_materializer_directly() -> None:
    assert agent_tool_runtime.materialize_model_source_edit is scalar_protocol.materialize_model_source_edit


def test_replace_exact_materializes_transactional_edit_with_host_hash(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    before = "final class Example {}\n"
    source.write_text(before, encoding="utf-8")

    payload = scalar_protocol.materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "replace_exact",
            "path": "src/main/java/example/Example.java",
            "old": "Example {}",
            "new": "Example { int value; }",
        },
    )

    assert payload["project_root"] == "demo"
    assert payload["operations"] == [
        {
            "operation": "edit",
            "path": "src/main/java/example/Example.java",
            "expected_sha256": "sha256:"
            + hashlib.sha256(source.read_bytes()).hexdigest(),
            "replacements": [
                {
                    "old": "Example {}",
                    "new": "Example { int value; }",
                    "count": 1,
                }
            ],
        }
    ]
    receipt = TransactionalSourcePatcher(project).apply(payload["operations"])
    assert receipt["status"] == "APPLIED"
    assert source.read_text(encoding="utf-8") == "final class Example { int value; }\n"


def test_delete_file_materializes_transactional_delete_with_host_hash(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Obsolete.java"
    before = "final class Obsolete {}\n"
    source.write_text(before, encoding="utf-8")

    payload = scalar_protocol.materialize_model_source_edit(
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
            "expected_sha256": "sha256:"
            + hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    receipt = TransactionalSourcePatcher(project).apply(payload["operations"])
    assert receipt["status"] == "APPLIED"
    assert not source.exists()


def test_lossless_aliases_stay_inside_single_scalar_protocol(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    source.write_text("final class Example {}\n", encoding="utf-8")

    delete_payload = scalar_protocol.materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "delete",
            "path": "src/main/java/example/Example.java",
        },
    )

    assert delete_payload["operations"][0]["operation"] == "delete"
