from __future__ import annotations

from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    LlamaCppAdapter,
    _tool_server_payload,
)


def _adapter() -> LlamaCppAdapter:
    return LlamaCppAdapter(
        AdapterConfig(
            role="coder_safe",
            adapter="llama_cpp",
            model_id="test/native-llama",
            max_new_tokens=512,
        )
    )


def _tool_schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def test_auto_tool_payload_preserves_visible_frontier() -> None:
    first = _tool_schema("search_code_rag")
    second = _tool_schema("java_workspace_symbols")
    request = GenerationRequest(
        messages=({"role": "user", "content": "find evidence"},),
        tools=(first, second),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    payload = _tool_server_payload(_adapter(), request)

    assert payload["tools"] == [first, second]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "grammar" not in payload


def test_named_tool_choice_narrows_wire_to_exact_selected_schema() -> None:
    selected = _tool_schema("search_code_rag")
    other = _tool_schema("java_workspace_symbols")
    request = GenerationRequest(
        messages=({"role": "user", "content": "search now"},),
        tools=(other, selected),
        tool_choice={
            "type": "function",
            "function": {"name": "search_code_rag"},
        },
        parallel_tool_calls=True,
    )

    payload = _tool_server_payload(_adapter(), request)

    assert payload["tools"] == [selected]
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is True
    assert payload["temperature"] == 0.0
    for key in (
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repeat_penalty",
        "repetition_penalty",
    ):
        assert key not in payload


def test_semantic_none_keeps_schemas_but_disables_tool_selection() -> None:
    first = _tool_schema("search_code_rag")
    second = _tool_schema("java_workspace_symbols")
    request = GenerationRequest(
        messages=({"role": "user", "content": "answer without tools"},),
        tools=(first, second),
        tool_choice="none",
    )

    payload = _tool_server_payload(_adapter(), request)

    assert payload["tool_choice"] == "none"
    assert payload["tools"] == [first, second]


def test_base_server_payload_and_transport_payload_agree_on_tool_choice() -> None:
    tool = _tool_schema("search_code_rag")
    request = GenerationRequest(
        messages=({"role": "user", "content": "search"},),
        tools=(tool,),
        tool_choice="required",
        parallel_tool_calls=False,
    )

    base = _server_payload(_adapter(), request)
    transport = _tool_server_payload(_adapter(), request)

    assert base["tool_choice"] == transport["tool_choice"] == "required"
    assert base["tools"] == transport["tools"] == [tool]
    assert base["parallel_tool_calls"] is False
    assert transport["parallel_tool_calls"] is False
