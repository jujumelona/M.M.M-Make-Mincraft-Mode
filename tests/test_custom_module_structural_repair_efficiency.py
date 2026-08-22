from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.model_adapters.base import ModelConfigurationError
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.scale_policy import ScalePolicy


def _implement_request(messages) -> dict:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("phase") == "implement_module":
            return payload
    raise AssertionError("No implement_module request was found in the coder message history.")


class _AgenticRouter:
    def __init__(self) -> None:
        self.workspace: Path | None = None
        self.calls: list[dict] = []
        self.messages: list[list[dict[str, str]]] = []

    def bind_agent_workspace(self, workspace_root, **_kwargs):
        self.workspace = Path(workspace_root)
        return self

    def generate_tool_decision(self, *_args, **_kwargs):
        raise AssertionError("custom-module production must not use a file-plan structured return channel")

    def generate_text(self, role, messages, **kwargs):
        assert role == "coder"
        assert kwargs["response_format"] == "text"
        assert kwargs["tool_stage"] == "generation"
        assert kwargs["enable_tools"] is True
        assert self.workspace is not None
        self.calls.append(dict(kwargs))
        self.messages.append([dict(message) for message in messages])
        request = _implement_request(messages)
        assert request["phase"] == "implement_module"
        project = self.workspace / request["workspace_project_root"]
        target = project / "src/main/java/example/Generated.java"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("package example; final class Generated {}\n", encoding="utf-8")
        return "Implemented the approved module."


def test_custom_module_uses_coding_agent_tool_loop_not_file_plan(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    router = _AgenticRouter()
    target = adapter_for_target("1.20.1", "fabric")

    result = CustomModuleGenerator(
        router,
        policy=ScalePolicy(model_context_bytes=4096),
    ).generate(
        root,
        module=ProductionModule("agentic_custom", "custom_java", {"feature": "shape"}),
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )

    assert result["status"] == "SOURCE_GENERATED"
    assert len(router.calls) == 1
    assert (root / "src/main/java/example/Generated.java").is_file()
    assert result["touched_paths"] == ["src/main/java/example/Generated.java"]
    request = _implement_request(router.messages[0])
    assert request["task"].startswith("Implement the approved Minecraft/Fabric mod feature")
    assert any("workspace/RAG/MCP tools" in rule for rule in request["rules"])
    assert all("return_custom_module_file_plan" not in str(message.get("content", "")) for message in router.messages[0])


def test_out_of_scope_agent_edit_is_discarded_without_touching_real_project(tmp_path: Path) -> None:
    root = tmp_path / "project"
    wrapper = root / "gradle/wrapper/gradle-wrapper.properties"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("distributionUrl=original\n", encoding="utf-8")

    class _MixedRouter(_AgenticRouter):
        def generate_text(self, role, messages, **kwargs):
            summary = super().generate_text(role, messages, **kwargs)
            request = _implement_request(messages)
            assert self.workspace is not None
            project = self.workspace / request["workspace_project_root"]
            staged_wrapper = project / "gradle/wrapper/gradle-wrapper.properties"
            staged_wrapper.write_text("distributionUrl=wrong\n", encoding="utf-8")
            return summary

    router = _MixedRouter()
    target = adapter_for_target("1.20.1", "fabric")
    result = CustomModuleGenerator(router, policy=ScalePolicy(model_context_bytes=4096)).generate(
        root,
        module=ProductionModule("safe_scope", "custom_java", {"feature": "shape"}),
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )

    assert wrapper.read_text(encoding="utf-8") == "distributionUrl=original\n"
    assert "gradle/wrapper/gradle-wrapper.properties" in result["discarded_out_of_scope_paths"]
    assert (root / "src/main/java/example/Generated.java").is_file()


def test_exhausted_causal_resync_discards_staged_edit_before_error_escapes(
    tmp_path: Path,
) -> None:
    """Lock the exact Colab failure's workspace-impact boundary.

    The model may already have used the staged edit tool before it repeats a stale
    action.  The resulting exception currently carries no durable tool transcript or
    cleanup receipt, so the real project must remain untouched and the orchestrator
    must not infer that replay is safe from the exception text alone.
    """

    root = tmp_path / "project"
    root.mkdir()

    class _CausalResyncFailureRouter(_AgenticRouter):
        def generate_text(self, role, messages, **kwargs):
            super().generate_text(role, messages, **kwargs)
            raise ModelConfigurationError(
                "Model failed the single causal-frontier re-synchronization attempt; "
                "forced='search_project_rag' rejected=apply_source_edit "
                "visible=search_project_rag,java_workspace_symbols,search_code_rag"
            )

    router = _CausalResyncFailureRouter()
    target = adapter_for_target("1.20.1", "fabric")
    with pytest.raises(ModelConfigurationError, match="causal-frontier"):
        CustomModuleGenerator(
            router,
            policy=ScalePolicy(model_context_bytes=4096),
        ).generate(
            root,
            module=ProductionModule(
                "causal_retry_guard",
                "custom_java",
                {"feature": "shape"},
            ),
            minecraft_version=target.minecraft_version,
            loader=target.loader,
            mappings=target.yarn_mappings,
        )

    assert not (root / "src/main/java/example/Generated.java").exists()
    assert router.workspace is not None
    assert not router.workspace.exists()
