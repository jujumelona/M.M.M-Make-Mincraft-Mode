from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters import llama_cpp_adapter
from minecraft_mod_ai.model_adapters.base import (
    AdapterConfig,
    GenerationRequest,
    ModelBackendError,
)
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.model_adapters.qwen_tool_parser import ToolCallValidationError


def _tool(name: str = "write_file") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "write a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


def _adapter(monkeypatch: pytest.MonkeyPatch) -> LlamaCppAdapter:
    adapter = LlamaCppAdapter(
        AdapterConfig(
            role="coder",
            adapter="llama_cpp",
            model_id="test.gguf",
            max_new_tokens=128,
        )
    )
    monkeypatch.setattr(adapter, "_server_url", lambda request: "http://llama.test/v1")

    from minecraft_mod_ai import llama_exact_context, llama_stream_efficiency_contract

    monkeypatch.setattr(
        llama_exact_context,
        "capacity_safe_payload",
        lambda server_url, payload, structured_output=False: dict(payload),
    )
    monkeypatch.setattr(
        llama_stream_efficiency_contract,
        "_report_server_connection",
        lambda server_url: None,
    )
    return adapter


def _request(*, choice="required", parallel: bool = False) -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "write x"},),
        tools=(_tool(),),
        tool_choice=choice,
        parallel_tool_calls=parallel,
    )


def _raw_call(name: str, arguments: str, *, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_native_tool_turn_uses_one_completion_and_structured_tool_calls(monkeypatch):
    adapter = _adapter(monkeypatch)
    seen: list[dict] = []

    def completion(server_url, payload):
        seen.append(dict(payload))
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_raw_call("write_file", '{"path":"src/Main.java"}')],
        }

    monkeypatch.setattr(llama_cpp_adapter, "_completion_message", completion)
    response = adapter.generate_turn(_request())

    assert len(seen) == 1
    assert seen[0]["tools"] == [_tool()]
    assert seen[0]["tool_choice"] == "required"
    assert response.tool_calls[0].name == "write_file"
    assert response.tool_calls[0].arguments == {"path": "src/Main.java"}


def test_required_tool_without_native_tool_call_fails_without_retry(monkeypatch):
    adapter = _adapter(monkeypatch)
    calls = 0

    def completion(server_url, payload):
        nonlocal calls
        calls += 1
        return {"role": "assistant", "content": "I did not call a tool."}

    monkeypatch.setattr(llama_cpp_adapter, "_completion_message", completion)
    with pytest.raises(ModelBackendError) as exc_info:
        adapter.generate_turn(_request())

    assert calls == 1
    assert isinstance(exc_info.value.cause, ToolCallValidationError)
    assert "required" in str(exc_info.value.cause)


def test_invalid_native_argument_json_is_fail_closed_recoverable_observation(monkeypatch):
    adapter = _adapter(monkeypatch)
    calls = 0

    def completion(server_url, payload):
        nonlocal calls
        calls += 1
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_raw_call("write_file", '{"path":')],
        }

    monkeypatch.setattr(llama_cpp_adapter, "_completion_message", completion)
    response = adapter.generate_turn(_request())

    assert calls == 1
    assert len(response.tool_calls) == 1
    rejection = response.tool_calls[0]
    assert rejection.name == "__mmm_rejected_tool_call__"
    assert rejection.arguments["failure_code"] == "TOOL_ARGUMENT_JSON_INVALID"
    assert rejection.arguments["original_tool"] == "write_file"


def test_non_visible_native_tool_is_fail_closed_recoverable_observation(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        llama_cpp_adapter,
        "_completion_message",
        lambda server_url, payload: {
            "role": "assistant",
            "content": None,
            "tool_calls": [_raw_call("hidden_tool", "{}")],
        },
    )

    response = adapter.generate_turn(_request(choice="auto"))

    assert len(response.tool_calls) == 1
    rejection = response.tool_calls[0]
    assert rejection.name == "__mmm_rejected_tool_call__"
    assert rejection.arguments["failure_code"] == "TOOL_NOT_VISIBLE"
    assert rejection.arguments["original_tool"] == "hidden_tool"


def test_schema_invalid_native_arguments_are_fail_closed_recoverable_observation(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        llama_cpp_adapter,
        "_completion_message",
        lambda server_url, payload: {
            "role": "assistant",
            "content": None,
            "tool_calls": [_raw_call("write_file", "{}")],
        },
    )

    response = adapter.generate_turn(_request())

    rejection = response.tool_calls[0]
    assert rejection.name == "__mmm_rejected_tool_call__"
    assert rejection.arguments["failure_code"] == "TOOL_SCHEMA_INVALID"


def test_mixed_valid_and_non_visible_calls_preserve_valid_call_and_reject_only_invalid() -> None:
    request = GenerationRequest(
        tools=(_tool(),),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    message = {
        "content": "",
        "tool_calls": [
            _raw_call("write_file", '{"path":"src/Main.java"}', call_id="valid"),
            _raw_call("hidden_tool", "{}", call_id="stale"),
        ],
    }

    response = llama_cpp_adapter._native_tool_generation_response(message, request)

    assert [call.name for call in response.tool_calls] == [
        "write_file",
        "__mmm_rejected_tool_call__",
    ]
    assert response.tool_calls[0].arguments == {"path": "src/Main.java"}
    assert response.tool_calls[1].arguments["failure_code"] == "TOOL_NOT_VISIBLE"
