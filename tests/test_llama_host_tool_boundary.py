from __future__ import annotations

from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


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
        "request_policy": "task_aware_sampling",
        "sampling_profiles": {
            "precise_coding": dict(_QWEN_PRECISE_PROFILE),
        },
    }


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


def test_native_tool_payload_keeps_server_peg_disabled() -> None:
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
    assert payload["tool_choice"] == "none"
    assert payload["parallel_tool_calls"] is True
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 0.0
    assert payload["repeat_penalty"] == 1.0
    for forbidden in ("response_format", "json_schema", "grammar"):
        assert forbidden not in payload
    rendered = "\n".join(
        str(message.get("content", "")) for message in payload["messages"]
    )
    assert "host-tool-envelope" not in rendered
    assert "REVIEWED_TOOL_CATALOG" not in rendered


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

    assert qwen_payload["temperature"] == 0.6
    assert qwen_payload["top_p"] == 0.95
    assert qwen_payload["top_k"] == 20
    assert qwen_payload["min_p"] == 0.0
    assert qwen_payload["presence_penalty"] == 0.0
    assert qwen_payload["repeat_penalty"] == 1.0
    assert generic_payload["temperature"] == 0.0
    assert "top_p" not in generic_payload
    assert "top_k" not in generic_payload
    assert "presence_penalty" not in generic_payload
