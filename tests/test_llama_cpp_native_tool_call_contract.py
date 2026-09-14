from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest, ModelBackendError
from minecraft_mod_ai.model_adapters import llama_cpp_adapter
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


def _request(*, choice="required") -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "write x"},),
        tools=(_tool(),),
        tool_choice=choice,
        parallel_tool_calls=False,
    )


def test_native_tool_turn_uses_one_completion_and_structured_tool_calls(monkeypatch):
    adapter = _adapter(monkeypatch)
    seen: list[dict] = []

    def completion(server_url, payload):
        seen.append(dict(payload))
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":"src/Main.java"}',
                    },
                }
            ],
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


def test_invalid_native_argument_json_fails_without_recovery(monkeypatch):
    adapter = _adapter(monkeypatch)
    calls = 0

    def completion(server_url, payload):
        nonlocal calls
        calls += 1
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":',
                    },
                }
            ],
        }

    monkeypatch.setattr(llama_cpp_adapter, "_completion_message", completion)
    with pytest.raises(ModelBackendError) as exc_info:
        adapter.generate_turn(_request())

    assert calls == 1
    assert isinstance(exc_info.value.cause, ToolCallValidationError)
    assert "invalid JSON" in str(exc_info.value.cause)


def test_non_visible_native_tool_is_rejected(monkeypatch):
    adapter = _adapter(monkeypatch)

    monkeypatch.setattr(
        llama_cpp_adapter,
        "_completion_message",
        lambda server_url, payload: {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "hidden_tool", "arguments": "{}"},
                }
            ],
        },
    )
    with pytest.raises(ModelBackendError) as exc_info:
        adapter.generate_turn(_request(choice="auto"))

    assert isinstance(exc_info.value.cause, ToolCallValidationError)
    assert "non-visible tool" in str(exc_info.value.cause)


def test_schema_invalid_native_arguments_are_rejected(monkeypatch):
    adapter = _adapter(monkeypatch)

    monkeypatch.setattr(
        llama_cpp_adapter,
        "_completion_message",
        lambda server_url, payload: {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                }
            ],
        },
    )
    with pytest.raises(ModelBackendError) as exc_info:
        adapter.generate_turn(_request())

    assert isinstance(exc_info.value.cause, ToolCallValidationError)
    assert "schema-invalid" in str(exc_info.value.cause)
