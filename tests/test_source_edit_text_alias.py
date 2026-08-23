from __future__ import annotations

import hashlib

import pytest

from minecraft_mod_ai import agent_tool_runtime
from minecraft_mod_ai.source_edit_scalar_protocol_contract import (
    SOURCE_EDIT_SCHEMA,
    materialize_model_source_edit,
)


def _project(workspace):
    project = workspace / "demo"
    (project / "src/main/java/example").mkdir(parents=True)
    (project / "settings.gradle").write_text('rootProject.name = "demo"\n', encoding="utf-8")
    return project


def test_source_edit_schema_accepts_only_explicit_text_alias() -> None:
    assert SOURCE_EDIT_SCHEMA["additionalProperties"] is False
    properties = SOURCE_EDIT_SCHEMA["properties"]
    assert properties["text"]["type"] == "string"
    operations = set(properties["operation"]["enum"])
    assert "replace_exact" in operations
    assert "create_file" in operations
    assert "delete_file" in operations
    assert "append_file" not in operations
    assert "replace_file" not in operations


def test_replace_exact_materializes_qwen_text_as_new(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    source.write_text("final class Example { int oldValue; }\n", encoding="utf-8")

    payload = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "replace_exact",
            "path": "src/main/java/example/Example.java",
            "old": "oldValue",
            "text": "newValue",
        },
    )

    assert payload["operations"][0]["replacements"] == [
        {"old": "oldValue", "new": "newValue", "count": 1}
    ]


def test_create_file_materializes_qwen_text_as_content(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace)

    payload = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "create_file",
            "path": "src/main/java/example/Created.java",
            "text": "package example;\nfinal class Created {}\n",
        },
    )

    assert payload["operations"] == [
        {
            "operation": "create",
            "path": "src/main/java/example/Created.java",
            "content": "package example;\nfinal class Created {}\n",
        }
    ]


def test_canonical_replacement_field_wins_over_text_alias(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    source.write_text("final class Example { int oldValue; }\n", encoding="utf-8")

    payload = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "replace_exact",
            "path": "src/main/java/example/Example.java",
            "old": "oldValue",
            "new": "canonicalValue",
            "text": "aliasValue",
        },
    )

    assert payload["operations"][0]["replacements"][0]["new"] == "canonicalValue"


def test_source_edit_still_rejects_unlisted_parameters(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    source.write_text("final class Example { int oldValue; }\n", encoding="utf-8")

    with pytest.raises(agent_tool_runtime.AgentToolRuntimeError, match="Unknown model-facing"):
        materialize_model_source_edit(
            agent_tool_runtime,
            workspace,
            {
                "operation": "replace_exact",
                "path": "src/main/java/example/Example.java",
                "old": "oldValue",
                "new": "newValue",
                "unexpected": "still rejected",
            },
        )


def test_delete_file_materializes_hash_guarded_delete(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    source.write_text("final class Example {}\n", encoding="utf-8")
    before = source.read_bytes()

    payload = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "delete_file",
            "path": "src/main/java/example/Example.java",
        },
    )

    assert payload["operations"] == [
        {
            "operation": "delete",
            "path": "src/main/java/example/Example.java",
            "expected_sha256": "sha256:" + hashlib.sha256(before).hexdigest(),
        }
    ]
