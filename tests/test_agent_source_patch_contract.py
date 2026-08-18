from __future__ import annotations

import hashlib

import pytest

from minecraft_mod_ai.agent_tool_runtime import (
    AgentToolRuntimeError,
    _MODEL_SOURCE_PATCH_SCHEMA,
    _materialize_model_source_patch,
)
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher


def _project(workspace, name: str = "demo"):
    project = workspace / name
    (project / "src").mkdir(parents=True)
    (project / "settings.gradle").write_text('rootProject.name = "demo"\n', encoding="utf-8")
    return project


def test_model_source_patch_schema_exposes_only_files_and_content() -> None:
    assert _MODEL_SOURCE_PATCH_SCHEMA["required"] == ["files"]
    assert set(_MODEL_SOURCE_PATCH_SCHEMA["properties"]) == {"files"}
    file_schema = _MODEL_SOURCE_PATCH_SCHEMA["properties"]["files"]["items"]
    assert file_schema["required"] == ["path", "content"]
    assert set(file_schema["properties"]) == {"path", "content"}
    assert "operation" not in file_schema["properties"]
    assert "expected_sha256" not in file_schema["properties"]
    assert "project_root" not in _MODEL_SOURCE_PATCH_SCHEMA["properties"]


def test_host_resolves_project_and_derives_replace_and_exact_sha(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    source.parent.mkdir(parents=True)
    old = "package example;\nfinal class Example {}\n"
    updated = "package example;\nfinal class Example { int value = 1; }\n"
    source.write_text(old, encoding="utf-8")

    payload = _materialize_model_source_patch(
        workspace,
        {
            "files": [
                {
                    "path": "src/main/java/example/Example.java",
                    "content": updated,
                }
            ],
        },
    )

    assert payload["project_root"] == "demo"
    assert payload["operations"] == [
        {
            "operation": "replace",
            "path": "src/main/java/example/Example.java",
            "expected_sha256": "sha256:" + hashlib.sha256(old.encode("utf-8")).hexdigest(),
            "content": updated,
        }
    ]

    receipt = TransactionalSourcePatcher(project).apply(payload["operations"])
    assert receipt["status"] == "APPLIED"
    assert source.read_text(encoding="utf-8") == updated


def test_host_derives_create_without_model_patch_metadata(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace)

    payload = _materialize_model_source_patch(
        workspace,
        {
            "files": [
                {
                    "path": "src/main/resources/assets/demo/lang/en_us.json",
                    "content": '{"item.demo.example":"Example"}\n',
                }
            ],
        },
    )

    assert payload["project_root"] == "demo"
    assert payload["operations"] == [
        {
            "operation": "create",
            "path": "src/main/resources/assets/demo/lang/en_us.json",
            "content": '{"item.demo.example":"Example"}\n',
        }
    ]


def test_model_source_patch_cannot_target_gradle_or_build_infrastructure(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace)

    with pytest.raises(AgentToolRuntimeError, match="limited to src/main/java"):
        _materialize_model_source_patch(
            workspace,
            {
                "files": [
                    {
                        "path": "gradle/wrapper/gradle-wrapper.properties",
                        "content": "distributionUrl=should-never-be-model-owned\n",
                    }
                ],
            },
        )


def test_model_cannot_choose_project_root_or_smuggle_patch_fields(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace)

    with pytest.raises(AgentToolRuntimeError, match="host-owned project/patch fields are forbidden"):
        _materialize_model_source_patch(
            workspace,
            {
                "project_root": "demo",
                "files": [
                    {
                        "path": "src/main/java/example/Example.java",
                        "content": "final class Example {}\n",
                    }
                ],
                "operations": [],
            },
        )


def test_host_refuses_ambiguous_project_root_instead_of_asking_model(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace, "one")
    _project(workspace, "two")

    with pytest.raises(AgentToolRuntimeError, match="exactly one source project"):
        _materialize_model_source_patch(
            workspace,
            {
                "files": [
                    {
                        "path": "src/main/java/example/Example.java",
                        "content": "final class Example {}\n",
                    }
                ]
            },
        )
