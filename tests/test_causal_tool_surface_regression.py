from __future__ import annotations

import pytest

from minecraft_mod_ai.causal_frontier_adapter import (
    CausalFrontierAdapter,
    FrontierExecutionGate,
    remember_authorized_tools,
)
from minecraft_mod_ai.causal_tool_frontier_contract import _FrontierRuntimeProxy
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
    ToolCall,
)
from minecraft_mod_ai.model_adapters import llama_cpp_adapter
from minecraft_mod_ai.small_model_hybrid_search_contract import _modes
from minecraft_mod_ai.tool_validation_surface_contract import install as install_tool_validation_surface


def _schema(name: str, properties: dict | None = None) -> dict:
    properties = properties or {}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def test_hidden_authorized_qwen_tool_is_parseable_but_not_executable() -> None:
    install_tool_validation_surface()
    visible = _schema("inspect_github_repository", {"repository": {"type": "string"}})
    edit = _schema(
        "apply_source_edit",
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair it"},),
        tools=(visible,),
        tool_validation_schemas=(visible, edit),
        tool_choice="auto",
    )
    message = {
        "content": (
            "<tool_call><function=apply_source_edit>"
            "<parameter=path>src/Main.java</parameter>"
            "<parameter=content>class Main {}</parameter>"
            "</function></tool_call>"
        )
    }

    response = llama_cpp_adapter._qwen_tool_generation_response(message, request)
    assert [call.name for call in response.tool_calls] == ["apply_source_edit"]

    class Runtime:
        def call(self, stage, name, arguments):
            return {"stage": stage, "name": name, "arguments": dict(arguments)}

    gate = FrontierExecutionGate()
    gate.set_visible(("inspect_github_repository",))
    proxy = _FrontierRuntimeProxy(Runtime(), gate)
    with pytest.raises(RuntimeError, match="not exposed on the current causal frontier"):
        proxy.call("generation", "apply_source_edit", {"path": "src/Main.java"})


def test_truly_unauthorized_qwen_tool_still_fails_validation() -> None:
    install_tool_validation_surface()
    visible = _schema("inspect_github_repository", {"repository": {"type": "string"}})
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair it"},),
        tools=(visible,),
        tool_validation_schemas=(visible,),
        tool_choice="auto",
    )
    message = {
        "content": (
            "<tool_call><function=apply_source_edit>"
            "<parameter=path>src/Main.java</parameter>"
            "</function></tool_call>"
        )
    }
    with pytest.raises(RuntimeError, match="unexposed tool 'apply_source_edit'"):
        llama_cpp_adapter._qwen_tool_generation_response(message, request)


def test_live_causal_adapter_uses_frozen_authorized_surface() -> None:
    edit = _schema(
        "apply_source_edit",
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
    )
    nested = _schema("inspect_huggingface_model", {"model": {"type": "string"}})

    class Inner:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            return GenerationResponse(content="ok")

    inner = Inner()
    adapter = CausalFrontierAdapter(
        inner,
        stage="generation",
        role="coder",
        require_fresh_evidence=False,
        authorized_surface=(edit,),
        preference={"apply_source_edit": 0},
    )

    # Simulate a nested retrieval/model call overwriting the compatibility ContextVar.
    remember_authorized_tools((nested,), {"inspect_huggingface_model": 0})
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair source"},),
        tools=(nested,),
        tool_choice={
            "type": "function",
            "function": {"name": "apply_source_edit"},
        },
    )
    assert adapter.generate_turn(request).content == "ok"

    sent = inner.requests[-1]
    assert [item["function"]["name"] for item in sent.tools] == ["apply_source_edit"]
    assert [
        item["function"]["name"] for item in sent.tool_validation_schemas
    ] == ["apply_source_edit"]


def test_live_causal_adapter_stops_repeated_stale_authorized_call() -> None:
    visible = _schema("inspect_github_repository", {"repository": {"type": "string"}})
    edit = _schema(
        "apply_source_edit",
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
    )
    stale_call = ToolCall(
        id="stale-edit",
        name="apply_source_edit",
        arguments={"path": "src/Main.java", "content": "class Main {}"},
        raw_arguments='{"path":"src/Main.java","content":"class Main {}"}',
    )

    class Inner:
        def generate_turn(self, request):
            return GenerationResponse(tool_calls=(stale_call,))

    adapter = CausalFrontierAdapter(
        Inner(),
        stage="generation",
        role="coder",
        require_fresh_evidence=False,
        authorized_surface=(visible, edit),
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair source"},),
        tools=(visible, edit),
        tool_choice={
            "type": "function",
            "function": {"name": "inspect_github_repository"},
        },
    )

    assert adapter.generate_turn(request).tool_calls == (stale_call,)
    with pytest.raises(
        ModelConfigurationError,
        match="repeated stale authorized tool calls without causal frontier progress",
    ):
        adapter.generate_turn(request)


def test_generic_code_rag_dense_escalation_is_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    assert _modes("semantic", False, False) == ((False, False, "lexical"),)

    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    modes = _modes("semantic", False, False)
    assert modes[0] == (False, False, "lexical")
    assert modes[1] == (False, True, "lexical+rerank")
    assert modes[2] == (True, True, "semantic+rerank")
