from __future__ import annotations

import pytest

import minecraft_mod_ai.llama_stream_efficiency_contract as stream_contract
import minecraft_mod_ai.model_adapters.llama_cpp_adapter as llama
from minecraft_mod_ai.model_adapters.base import GenerationRequest, ToolDefinition
from minecraft_mod_ai.model_adapters.llama_turn_retry import _ensure_adapter_guards


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def _native_call(name: str, query: str = "x") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_test",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": '{"query":"' + query + '"}',
                },
            }
        ],
    }


def test_valid_first_tool_completion_is_not_regenerated(monkeypatch):
    _ensure_adapter_guards(llama.LlamaCppAdapter)
    calls: list[int] = []

    def fake_completion(*args, **kwargs):
        calls.append(1)
        return _native_call("apply_source_edit")

    monkeypatch.setattr(llama, "_completion_message_with_prefill", fake_completion)
    monkeypatch.setattr(stream_contract, "_report_server_connection", lambda *_: None)
    monkeypatch.setattr(llama, "_tool_server_payload", lambda *args, **kwargs: {})
    request = GenerationRequest(
        tools=(_tool("apply_source_edit"),),
        tool_choice="required",
        parallel_tool_calls=False,
    )

    result = llama._tool_semantic_completion(object(), "http://local", request)

    assert [call.name for call in result.tool_calls] == ["apply_source_edit"]
    assert len(calls) == 1


def test_reasoning_only_completion_has_one_explicit_semantic_continuation(monkeypatch):
    _ensure_adapter_guards(llama.LlamaCppAdapter)
    messages = [
        {"role": "assistant", "content": "", "reasoning_content": "reasoning"},
        _native_call("apply_source_edit"),
    ]
    calls: list[int] = []

    def fake_completion(*args, **kwargs):
        calls.append(1)
        return messages.pop(0)

    monkeypatch.setattr(llama, "_completion_message_with_prefill", fake_completion)
    monkeypatch.setattr(stream_contract, "_report_server_connection", lambda *_: None)
    monkeypatch.setattr(llama, "_tool_server_payload", lambda *args, **kwargs: {})
    request = GenerationRequest(
        tools=(_tool("apply_source_edit"),),
        tool_choice="required",
        parallel_tool_calls=False,
    )

    result = llama._tool_semantic_completion(object(), "http://local", request)

    assert [call.name for call in result.tool_calls] == ["apply_source_edit"]
    assert len(calls) == 2


def test_stale_tool_name_is_rejected_at_current_adapter_frontier(capsys):
    _ensure_adapter_guards(llama.LlamaCppAdapter)
    request = GenerationRequest(
        tools=(_tool("search_code_rag"),),
        tool_choice="required",
        parallel_tool_calls=False,
    )

    with pytest.raises(Exception, match="not exposed in the current tool frontier"):
        llama._qwen_tool_generation_response(_native_call("apply_source_edit"), request)

    captured = capsys.readouterr().out
    assert '"parse_status": "tool_name_not_allowed"' in captured
    assert '"outcome": "discarded"' in captured


def test_tool_frontier_change_does_not_reuse_previous_allowed_tool():
    _ensure_adapter_guards(llama.LlamaCppAdapter)
    edit_request = GenerationRequest(
        tools=(_tool("apply_source_edit"),),
        tool_choice="required",
        parallel_tool_calls=False,
    )
    recovery_request = GenerationRequest(
        tools=(_tool("search_code_rag"),),
        tool_choice="required",
        parallel_tool_calls=False,
    )
    message = _native_call("apply_source_edit")

    accepted = llama._qwen_tool_generation_response(message, edit_request)
    assert [call.name for call in accepted.tool_calls] == ["apply_source_edit"]

    with pytest.raises(Exception, match="not exposed in the current tool frontier"):
        llama._qwen_tool_generation_response(message, recovery_request)
