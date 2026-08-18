from __future__ import annotations

import hashlib

import pytest

from minecraft_mod_ai.agent_tool_runtime import (
    AgentToolRuntime,
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


def _capture_first_party_payload(runtime, monkeypatch):
    captured = {}
    runtime._schema_cache["generation"] = ()
    runtime._allowed_tool_cache["generation"] = frozenset({"apply_source_patch"})

    def fake_run_async(_function, *args):
        captured["args"] = args
        return {
            "structured_content": {"ok": True},
            "text": [],
            "parsed_text": None,
            "resources": [],
        }

    monkeypatch.setattr(runtime, "_run_async", fake_run_async)
    return captured


def test_model_source_patch_schema_exposes_only_files_and_content() -> None:
    assert _MODEL_SOURCE_PATCH_SCHEMA["required"] == ["files"]
    assert set(_MODEL_SOURCE_PATCH_SCHEMA["properties"]) == {"files"}
    file_schema = _MODEL_SOURCE_PATCH_SCHEMA["properties"]["files"]["items"]
    assert file_schema["required"] == ["path", "content"]
    assert set(file_schema["properties"]) == {"path", "content"}
    assert "operation" not in file_schema["properties"]
    assert "expected_sha256" not in file_schema["properties"]
    assert "project_root" not in _MODEL_SOURCE_PATCH_SCHEMA["properties"]


def test_generation_tool_schema_hides_raw_patch_protocol(monkeypatch, tmp_path) -> None:
    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)
    raw_schema = {
        "name": "apply_source_patch",
        "description": "raw strict patch",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_root": {"type": "string"},
                "operations": {"type": "array"},
            },
            "required": ["project_root", "operations"],
        },
    }
    monkeypatch.setattr(runtime, "_run_async", lambda *_args: [raw_schema])
    monkeypatch.setattr(runtime._external_bridge, "tool_schemas", lambda _stage: ())

    exposed = runtime.tool_schemas("generation")

    patch_tool = next(
        item for item in exposed if item["function"]["name"] == "apply_source_patch"
    )
    assert patch_tool["function"]["parameters"] == _MODEL_SOURCE_PATCH_SCHEMA
    assert set(patch_tool["function"]["parameters"]["properties"]) == {"files"}


def test_host_stage_call_preserves_raw_strict_patch_contract(monkeypatch, tmp_path) -> None:
    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)
    captured = _capture_first_party_payload(runtime, monkeypatch)
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


def test_model_scoped_call_materializes_host_patch_metadata(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace)
    runtime = AgentToolRuntime(profile="test", workspace_root=workspace)
    captured = _capture_first_party_payload(runtime, monkeypatch)

    runtime.call_scoped(
        "generation",
        "apply_source_patch",
        {
            "files": [
                {
                    "path": "src/main/java/example/Example.java",
                    "content": "package example;\nfinal class Example {}\n",
                }
            ]
        },
        external_server_ids=(),
    )

    stage, name, payload = captured["args"]
    assert stage == "generation"
    assert name == "apply_source_patch"
    assert payload["project_root"] == "demo"
    assert payload["operations"] == [
        {
            "operation": "create",
            "path": "src/main/java/example/Example.java",
            "content": "package example;\nfinal class Example {}\n",
        }
    ]


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
