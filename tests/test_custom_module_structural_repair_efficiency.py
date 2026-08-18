from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.scale_policy import ScalePolicy


class _StructuralRepairRouter:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.messages: list[list[dict[str, str]]] = []

    def bind_agent_workspace(self, *_args, **_kwargs):
        return self

    def generate_text(self, *_args, **_kwargs):
        raise AssertionError("custom-module production must not fall back to free-form JSON repair")

    def generate_tool_decision(
        self,
        role,
        messages,
        *,
        tool_name,
        parameters,
        description="",
    ):
        assert role == "coder"
        self.calls.append(
            {
                "tool_name": tool_name,
                "parameters": parameters,
                "description": description,
            }
        )
        self.messages.append([dict(message) for message in messages])
        request = json.loads(messages[-1]["content"])
        if tool_name == "return_custom_module_file_plan":
            assert request["phase"] == "plan_files"
            return {
                "files": [
                    {
                        "path": "src/main/java/example/Generated.java",
                        "purpose": "Implement the approved custom module.",
                    }
                ],
                "runtime_tests": [],
            }
        if tool_name == "return_custom_module_file_content":
            assert request["phase"] == "write_file"
            assert request["path"] == "src/main/java/example/Generated.java"
            return {
                "content": "package example; final class Generated {}\n",
                "runtime_tests": [],
            }
        raise AssertionError(f"unexpected structured return channel: {tool_name}")


def test_custom_module_native_return_channel_never_builds_rag_or_enters_json_repair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    def forbidden_index(*_args, **_kwargs):
        raise AssertionError("custom-module structured return must not build project RAG")

    from minecraft_mod_ai.production_tools import ProductionToolService

    monkeypatch.setattr(ProductionToolService, "index_project_rag", forbidden_index)
    router = _StructuralRepairRouter()
    target = adapter_for_target("1.20.1", "fabric")
    result = CustomModuleGenerator(
        router,
        policy=ScalePolicy(model_context_bytes=4096),
    ).generate(
        root,
        module=ProductionModule("shape_repair", "custom_java", {"feature": "shape"}),
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )

    assert result["status"] == "SOURCE_GENERATED"
    assert [call["tool_name"] for call in router.calls] == [
        "return_custom_module_file_plan",
        "return_custom_module_file_content",
    ]
    assert all(call["parameters"]["additionalProperties"] is False for call in router.calls)
    assert (root / "src/main/java/example/Generated.java").is_file()

    plan_request = json.loads(router.messages[0][-1]["content"])
    file_request = json.loads(router.messages[1][-1]["content"])
    assert plan_request["host_owned"] == [
        "create/replace/edit decision",
        "expected_sha256",
        "patch transaction",
        "pagination/cursor/progress/completion",
        "path canonicalization and protection",
    ]
    assert file_request["existing_file"] is False
    assert file_request["path"] == "src/main/java/example/Generated.java"
