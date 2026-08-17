from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    LlamaCppAdapter,
    _native_server_payload,
)


def _tool_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "search_code_rag",
            "description": "Search code",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def test_native_tool_payload_uses_llama_openai_fields_without_grammar() -> None:
    adapter = LlamaCppAdapter(
        AdapterConfig(
            role="coder_safe",
            adapter="llama_cpp",
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        )
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "find evidence"},),
        tools=(_tool_schema(),),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    payload = _native_server_payload(adapter, request)

    assert payload["tools"] == [_tool_schema()]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    for forbidden in ("response_format", "json_schema", "grammar"):
        assert forbidden not in payload
    rendered = "\n".join(
        str(message.get("content", "")) for message in payload["messages"]
    )
    assert "host-tool-envelope" not in rendered
    assert "REVIEWED_TOOL_CATALOG" not in rendered


def test_qwen35_tool_profile_is_model_scoped() -> None:
    request = GenerationRequest(
        messages=({"role": "user", "content": "find evidence"},),
        tools=(_tool_schema(),),
    )
    qwen = LlamaCppAdapter(
        AdapterConfig(
            role="coder_safe",
            adapter="llama_cpp",
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        )
    )
    generic = LlamaCppAdapter(
        AdapterConfig(
            role="coder_safe",
            adapter="llama_cpp",
            model_id="generic/model",
        )
    )

    qwen_payload = _native_server_payload(qwen, request)
    generic_payload = _native_server_payload(generic, request)

    assert qwen_payload["temperature"] == 0.7
    assert qwen_payload["top_p"] == 0.8
    assert qwen_payload["top_k"] == 20
    assert qwen_payload["presence_penalty"] == 1.5
    assert generic_payload["temperature"] == 0.0
    assert "top_p" not in generic_payload
    assert "top_k" not in generic_payload
    assert "presence_penalty" not in generic_payload
