from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters import llama_cpp_adapter
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest, ModelBackendError
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


def _request(*, parallel: bool = True) -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "write two files"},),
        tools=(_tool(),),
        tool_choice="auto",
        parallel_tool_calls=parallel,
    )


def _valid_call(call_id: str = "call_valid") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "write_file",
            "arguments": '{"path":"src/Main.java"}',
        },
    }


def test_malformed_native_sibling_does_not_discard_valid_call(monkeypatch):
    adapter = _adapter(monkeypatch)
    completions = 0

    def completion(server_url, payload):
        nonlocal completions
        completions += 1
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _valid_call(),
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":',
                    },
                },
            ],
        }

    monkeypatch.setattr(llama_cpp_adapter, "_completion_message", completion)
    response = adapter.generate_turn(_request())

    assert completions == 1
    assert [call.name for call in response.tool_calls] == [
        "write_file",
        "__mmm_rejected_tool_call__",
    ]
    assert response.tool_calls[0].arguments == {"path": "src/Main.java"}
    rejected = response.tool_calls[1]
    assert rejected.arguments["original_tool"] == "write_file"
    assert rejected.arguments["failure_code"] == "TOOL_ARGUMENT_JSON_INVALID"
    assert "invalid JSON" in rejected.arguments["error"]
    assert rejected.arguments["raw_arguments"] == '{"path":'


def test_schema_invalid_native_sibling_does_not_discard_valid_call(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        llama_cpp_adapter,
        "_completion_message",
        lambda server_url, payload: {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _valid_call(),
                {
                    "id": "call_bad_schema",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                },
            ],
        },
    )

    response = adapter.generate_turn(_request())

    assert response.tool_calls[0].name == "write_file"
    rejected = response.tool_calls[1]
    assert rejected.name == "__mmm_rejected_tool_call__"
    assert rejected.arguments["original_tool"] == "write_file"
    assert rejected.arguments["failure_code"] == "TOOL_SCHEMA_INVALID"
    assert "schema-invalid" in rejected.arguments["error"]


def test_all_invalid_native_calls_return_non_executable_rejection(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        llama_cpp_adapter,
        "_completion_message",
        lambda server_url, payload: {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":',
                    },
                }
            ],
        },
    )

    response = adapter.generate_turn(_request())

    assert len(response.tool_calls) == 1
    rejected = response.tool_calls[0]
    assert rejected.name == "__mmm_rejected_tool_call__"
    assert rejected.arguments["original_tool"] == "write_file"
    assert rejected.arguments["failure_code"] == "TOOL_ARGUMENT_JSON_INVALID"
    assert "invalid JSON" in rejected.arguments["error"]


def test_parallel_disabled_does_not_use_isolation_to_bypass_protocol(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        llama_cpp_adapter,
        "_completion_message",
        lambda server_url, payload: {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _valid_call(),
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":',
                    },
                },
            ],
        },
    )

    with pytest.raises(ModelBackendError) as exc_info:
        adapter.generate_turn(_request(parallel=False))

    assert isinstance(exc_info.value.cause, ToolCallValidationError)
    assert "parallel tool calls" in str(exc_info.value.cause)
