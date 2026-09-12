from __future__ import annotations

from minecraft_mod_ai import agent_tool_runtime
from minecraft_mod_ai.source_edit_scalar_protocol_contract import materialize_model_source_edit


def _project(workspace):
    project = workspace / "demo"
    (project / "src/main/java/dev/mmm/debug").mkdir(parents=True)
    (project / "src/main/resources").mkdir(parents=True)
    (project / "settings.gradle").write_text('rootProject.name = "demo"\n', encoding="utf-8")
    return project


def test_model_create_file_allows_fresh_java_source(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace)

    payload = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "create_file",
            "path": "src/main/java/dev/mmm/debug/DebugToken.java",
            "content": "package dev.mmm.debug;\n\npublic final class DebugToken {}\n",
        },
    )

    assert payload["project_root"] == "demo"
    assert payload["operations"] == [
        {
            "operation": "create",
            "path": "src/main/java/dev/mmm/debug/DebugToken.java",
            "content": "package dev.mmm.debug;\n\npublic final class DebugToken {}\n",
        }
    ]


def test_model_create_file_still_allows_non_java_resource(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace)

    payload = materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "operation": "create_file",
            "path": "src/main/resources/debug-token.json",
            "content": '{"debug":true}\n',
        },
    )

    assert payload["project_root"] == "demo"
    assert payload["operations"] == [
        {
            "operation": "create",
            "path": "src/main/resources/debug-token.json",
            "content": '{"debug":true}\n',
        }
    ]
