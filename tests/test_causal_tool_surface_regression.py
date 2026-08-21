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
from minecraft_mod_ai.tool_validation_surface_contract import (
    _validation_surface,
    install as install_tool_validation_surface,
)


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


def test_visible_schema_wins_over_stale_authorized_schema_for_same_tool() -> None:
    install_tool_validation_surface()
    visible_edit = _schema(
        "apply_source_edit",
        {
            "operation": {
                "type": "string",
                "enum": ["replace_exact", "insert_before", "insert_after"],
            },
            "path": {"type": "string"},
        },
    )
    stale_host_edit = _schema(
        "apply_source_edit",
        {
            "operation": {
                "type": "string",
                "enum": ["create", "replace", "edit"],
            },
            "path": {"type": "string"},
        },
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair it"},),
        tools=(visible_edit,),
        tool_validation_schemas=(stale_host_edit,),
        tool_choice="auto",
    )
    message = {
        "content": (
            "<tool_call><function=apply_source_edit>"
            "<parameter=operation>replace_exact</parameter>"
            "<parameter=path>src/main/java/example/Main.java</parameter>"
            "</function></tool_call>"
        )
    }

    response = llama_cpp_adapter._qwen_tool_generation_response(message, request)
    assert response.tool_calls[0].arguments["operation"] == "replace_exact"


def test_duplicate_name_inside_one_validation_surface_fails_closed() -> None:
    first = _schema("apply_source_edit", {"path": {"type": "string"}})
    second = _schema(
        "apply_source_edit",
        {"operation": {"type": "string", "enum": ["replace_exact"]}},
    )
    with pytest.raises(
        RuntimeError,
        match="duplicate tool schema name 'apply_source_edit'.*authorized-validation",
    ):
        _validation_surface((), (first, second))


def test_duplicate_name_inside_causal_authorization_fails_before_selection() -> None:
    first = _schema("inspect_github_repository", {"repository": {"type": "string"}})
    second = _schema("inspect_github_repository", {"owner": {"type": "string"}})
    with pytest.raises(
        RuntimeError,
        match="duplicate tool schema name 'inspect_github_repository'.*causal-authorized",
    ):
        remember_authorized_tools((first, second))


def test_reasoning_continuation_preserves_full_request_contract() -> None:
    install_tool_validation_surface()
    visible = _schema("inspect_github_repository", {"repository": {"type": "string"}})
    hidden = _schema("apply_source_edit", {"path": {"type": "string"}})
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair it"},),
        response_format="json",
        response_schema={"type": "object"},
        tools=(visible,),
        tool_validation_schemas=(visible, hidden),
        tool_choice="auto",
        parallel_tool_calls=False,
        task="repair-task",
        prompt="repair-prompt",
        metadata={"trace": "schema-regression"},
    )

    continued = llama_cpp_adapter._reasoning_continuation_request(request, "thinking")

    assert continued.tool_validation_schemas == request.tool_validation_schemas
    assert continued.task == request.task
    assert continued.prompt == request.prompt
    assert continued.metadata == request.metadata
    assert continued.response_format == request.response_format
    assert continued.response_schema == request.response_schema
    assert continued.tools == request.tools
    assert continued.tool_choice == request.tool_choice
    assert continued.parallel_tool_calls is False
    assert continued.media_paths == ()
    assert continued.messages[-1]["role"] == "user"


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


def test_live_causal_adapter_restores_context_dropped_by_core_turn_copy() -> None:
    edit = _schema("apply_source_edit", {"path": {"type": "string"}})
    forced = {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }
    template = GenerationRequest(
        messages=({"role": "user", "content": "repair source"},),
        response_format="json",
        response_schema={"type": "object"},
        tools=(edit,),
        tool_validation_schemas=(edit,),
        tool_choice=forced,
        parallel_tool_calls=False,
        task="repair-task",
        prompt="repair-prompt",
        metadata={"trace": "live-loop"},
    )
    # Mirrors an older ModelRouter copy constructor that carried only generation fields.
    derived = GenerationRequest(
        messages=({"role": "user", "content": "repair source"},),
        response_format="json",
        response_schema={"type": "object"},
        tools=(edit,),
        tool_choice=forced,
        parallel_tool_calls=False,
    )

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
        request_template=template,
    )

    assert adapter.generate_turn(derived).content == "ok"
    sent = inner.requests[-1]
    assert sent.task == "repair-task"
    assert sent.prompt == "repair-prompt"
    assert sent.metadata == {"trace": "live-loop"}
    assert sent.tool_validation_schemas == (edit,)
    assert sent.response_schema == {"type": "object"}


def test_live_causal_adapter_final_synthesis_clears_tool_validation_only() -> None:
    edit = _schema("apply_source_edit", {"path": {"type": "string"}})
    template = GenerationRequest(
        messages=({"role": "user", "content": "repair source"},),
        tools=(edit,),
        tool_validation_schemas=(edit,),
        tool_choice="auto",
        task="repair-task",
        prompt="repair-prompt",
        metadata={"trace": "final-synthesis"},
    )
    final_copy = GenerationRequest(
        messages=({"role": "user", "content": "final answer"},),
        tools=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )

    class Inner:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            return GenerationResponse(content="done")

    inner = Inner()
    adapter = CausalFrontierAdapter(
        inner,
        stage="generation",
        role="coder",
        require_fresh_evidence=False,
        authorized_surface=(edit,),
        request_template=template,
    )

    assert adapter.generate_turn(final_copy).content == "done"
    sent = inner.requests[-1]
    assert sent.tools == ()
    assert sent.tool_validation_schemas == ()
    assert sent.tool_choice is None
    assert sent.parallel_tool_calls is False
    assert sent.task == "repair-task"
    assert sent.prompt == "repair-prompt"
    assert sent.metadata == {"trace": "final-synthesis"}


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
