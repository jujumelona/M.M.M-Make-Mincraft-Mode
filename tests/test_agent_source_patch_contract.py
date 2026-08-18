from __future__ import annotations

import hashlib

import pytest

from minecraft_mod_ai.agent_tool_runtime import (
    AgentToolRuntimeError,
    _MODEL_SOURCE_PATCH_SCHEMA,
    _materialize_model_source_patch,
)


def test_model_source_patch_schema_exposes_only_files_and_content() -> None:
    assert _MODEL_SOURCE_PATCH_SCHEMA["required"] == ["project_root", "files"]
    file_schema = _MODEL_SOURCE_PATCH_SCHEMA["properties"]["files"]["items"]
    assert file_schema["required"] == ["path", "content"]
    assert set(file_schema["properties"]) == {"path", "content"}
    assert "operation" not in file_schema["properties"]
    assert "expected_sha256" not in file_schema["properties"]


def test_host_derives_replace_and_exact_sha(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "demo"
    source = project / "src/main/java/example/Example.java"
    source.parent.mkdir(parents=True)
    old = "package example;\nfinal class Example {}\n"
    source.write_text(old, encoding="utf-8")

    payload = _materialize_model_source_patch(
        workspace,
        {
            "project_root": "demo",
            "files": [
                {
                    "path": "src/main/java/example/Example.java",
                    "content": "package example;\nfinal class Example { int value = 1; }\n",
                }
            ],
        },
    )

    assert payload["project_root"] == "demo"
    assert payload["operations"] == [
        {
            "operation": "replace",
            "path": "src/main/java/example/Example.java",
            "expected_sha256": hashlib.sha256(old.encode("utf-8")).hexdigest(),
            "content": "package example;\nfinal class Example { int value = 1; }\n",
        }
    ]


def test_host_derives_create_without_model_patch_metadata(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "demo"
    project.mkdir(parents=True)

    payload = _materialize_model_source_patch(
        workspace,
        {
            "project_root": "demo",
            "files": [
                {
                    "path": "src/main/resources/assets/demo/lang/en_us.json",
                    "content": '{"item.demo.example":"Example"}\n',
                }
            ],
        },
    )

    assert payload["operations"] == [
        {
            "operation": "create",
            "path": "src/main/resources/assets/demo/lang/en_us.json",
            "content": '{"item.demo.example":"Example"}\n',
        }
    ]


def test_model_source_patch_cannot_target_gradle_or_build_infrastructure(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "demo"
    project.mkdir(parents=True)

    with pytest.raises(AgentToolRuntimeError, match="limited to src/main/java"):
        _materialize_model_source_patch(
            workspace,
            {
                "project_root": "demo",
                "files": [
                    {
                        "path": "gradle/wrapper/gradle-wrapper.properties",
                        "content": "distributionUrl=should-never-be-model-owned\n",
                    }
                ],
            },
        )


def test_model_cannot_smuggle_host_owned_patch_fields(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "demo"
    project.mkdir(parents=True)

    with pytest.raises(AgentToolRuntimeError, match="host-owned patch fields are forbidden"):
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
