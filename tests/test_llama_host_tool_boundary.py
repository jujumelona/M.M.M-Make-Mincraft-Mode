from __future__ import annotations

from minecraft_mod_ai.forced_tool_execution_contract import _single_tool_request
from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    LlamaCppAdapter,
    _qwen_tool_generation_response,
)


_QWEN_PRECISE_PROFILE = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}


def _qwen_extra() -> dict[str, object]:
    return {
        "runtime_contract": "qwen",
        "qwen_family": "qwen3.5",
        "qwen_tool_markup": "qwen3_coder_xml",
        "qwen_action_thinking_control": "enable_thinking_false",
        "qwen_preserve_thinking": False,
        "qwen_reasoning_effort": False,
        "qwen_assistant_prefill": True,
        "request_policy": "task_aware_sampling",
        "sampling_profiles": {
            "precise_coding": dict(_QWEN_PRECISE_PROFILE),
            "non_thinking": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repeat_penalty": 1.0,
            },
        },
    }


def _tool_schema(name: str = "search_code_rag") -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Search code",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def test_native_tool_payload_keeps_jinja_tools_visible_for_host_parser() -> None:
    adapter = LlamaCppAdapter(
        AdapterConfig(
            role="coder_safe",
            adapter="llama_cpp",
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            extra=_qwen_extra(),
        )
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "find evidence"},),
        tools=(_tool_schema(),),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    payload = _server_payload(adapter, request)

    assert request.tool_choice == "auto"
    assert payload["tools"] == [_tool_schema()]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 1.5
    assert payload["repeat_penalty"] == 1.0
    for forbidden in ("response_format", "json_schema", "grammar"):
        assert forbidden not in payload
    rendered = "\n".join(
        str(message.get("content", "")) for message in payload["messages"]
    )
    assert "host-tool-envelope" not in rendered
    assert "REVIEWED_TOOL_CATALOG" not in rendered


def test_local_host_forced_qwen35_turn_is_required_and_deterministic_on_wire() -> None:
    adapter = LlamaCppAdapter(
        AdapterConfig(
            role="coder_safe",
            adapter="llama_cpp",
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            extra=_qwen_extra(),
        )
    )
    named = GenerationRequest(
        messages=({"role": "user", "content": "search now"},),
        tools=(_tool_schema(),),
        tool_choice={
            "type": "function",
            "function": {"name": "search_code_rag"},
        },
    )
    request = _single_tool_request(named, "search_code_rag")

    payload = _server_payload(adapter, request)

    assert request.tool_choice == "required"
    assert request.parallel_tool_calls is False
    assert payload["tool_choice"] == "required"
    assert payload["temperature"] == 0.0
    assert "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    for key in (
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repeat_penalty",
        "repetition_penalty",
    ):
        assert key not in payload


def test_multi_tool_named_choice_narrows_wire_and_host_enforces_exact_name() -> None:
    adapter = LlamaCppAdapter(
        AdapterConfig(
            role="coder_safe",
            adapter="llama_cpp",
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            extra=_qwen_extra(),
        )
    )
    selected = _tool_schema("search_code_rag")
    other = _tool_schema("java_workspace_symbols")
    request = GenerationRequest(
        messages=({"role": "user", "content": "search now"},),
        tools=(other, selected),
        tool_choice={
            "type": "function",
            "function": {"name": "search_code_rag"},
        },
    )

    payload = _server_payload(adapter, request)

    assert payload["tool_choice"] == "required"
    assert payload["tools"] == [selected]
    raw_selected = {
        "content": (
            "<tool_call><function=search_code_rag>"
            "<parameter=query>registry</parameter>"
            "</function></tool_call>"
        )
    }
    turn = _qwen_tool_generation_response(raw_selected, request)
    assert [call.name for call in turn.tool_calls] == ["search_code_rag"]

    raw_other = {
        "content": (
            "<tool_call><function=java_workspace_symbols>"
            "<parameter=query>registry</parameter>"
            "</function></tool_call>"
        )
    }
    try:
        _qwen_tool_generation_response(raw_other, request)
    except RuntimeError as exc:
        assert "violated named tool_choice" in str(exc)
    else:
        raise AssertionError("host must reject a non-selected named tool")


def test_semantic_none_remains_none_without_narrowing_tool_schemas() -> None:
    adapter = LlamaCppAdapter(
        AdapterConfig(
            role="coder_safe",
            adapter="llama_cpp",
            model_id="generic/model",
        )
    )
    first = _tool_schema("search_code_rag")
    second = _tool_schema("java_workspace_symbols")
    request = GenerationRequest(
        messages=({"role": "user", "content": "answer without tools"},),
        tools=(first, second),
        tool_choice="none",
    )

    payload = _server_payload(adapter, request)

    assert payload["tool_choice"] == "none"
    assert payload["tools"] == [first, second]


def test_qwen35_tool_profile_is_registry_scoped() -> None:
    request = GenerationRequest(
        messages=({"role": "user", "content": "find evidence"},),
        tools=(_tool_schema(),),
    )
    qwen = LlamaCppAdapter(
        AdapterConfig(
            role="coder_safe",
            adapter="llama_cpp",
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            extra=_qwen_extra(),
        )
    )
    generic = LlamaCppAdapter(
        AdapterConfig(
            role="coder_safe",
            adapter="llama_cpp",
            model_id="generic/model",
        )
    )

    qwen_payload = _server_payload(qwen, request)
    generic_payload = _server_payload(generic, request)

    assert qwen_payload["temperature"] == 0.7
    assert "reasoning_effort" not in qwen_payload
    assert qwen_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert qwen_payload["top_p"] == 0.8
    assert qwen_payload["top_k"] == 20
    assert qwen_payload["min_p"] == 0.0
    assert qwen_payload["presence_penalty"] == 1.5
    assert qwen_payload["repeat_penalty"] == 1.0
    assert generic_payload["temperature"] == 0.0
    assert "top_p" not in generic_payload
    assert "top_k" not in generic_payload
    assert "presence_penalty" not in generic_payload
