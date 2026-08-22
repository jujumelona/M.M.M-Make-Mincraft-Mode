from __future__ import annotations

import pytest

from minecraft_mod_ai import agent_tool_runtime
from minecraft_mod_ai.small_model_execution_extensions_contract import (
    _SOURCE_EDIT_SCHEMA,
    _materialize_model_source_edit,
)


def _project(workspace):
    project = workspace / "demo"
    (project / "src/main/java/example").mkdir(parents=True)
    (project / "settings.gradle").write_text('rootProject.name = "demo"\n', encoding="utf-8")
    return project


def test_source_edit_schema_accepts_only_explicit_text_alias() -> None:
    assert _SOURCE_EDIT_SCHEMA["additionalProperties"] is False
    properties = _SOURCE_EDIT_SCHEMA["properties"]
    assert properties["text"]["type"] == "string"
    assert properties["text"]["maxLength"] == properties["new"]["maxLength"]


def test_replace_exact_materializes_qwen_text_as_new(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    source.write_text("final class Example { int oldValue; }\n", encoding="utf-8")

    payload = _materialize_model_source_edit(
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

    payload = _materialize_model_source_edit(
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

    payload = _materialize_model_source_edit(
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
        _materialize_model_source_edit(
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
