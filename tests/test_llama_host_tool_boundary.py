from __future__ import annotations

import minecraft_mod_ai.model_adapters.llama_cpp_adapter as llama_adapter
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest


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


def test_host_tool_request_never_requests_native_grammar() -> None:
    request = llama_adapter._host_tool_request(
        [{"role": "user", "content": "find evidence"}],
    )

    assert request.response_format == "text"
    assert request.response_schema is None
    assert request.tools == ()
    assert request.tool_choice is None
    assert request.parallel_tool_calls is False


def test_host_tool_payload_strips_native_controls_from_wrapped_builder() -> None:
    request = llama_adapter._host_tool_request(
        [{"role": "user", "content": "find evidence"}],
    )

    def polluted_builder(adapter, wire_request):
        del adapter
        assert wire_request.response_format == "text"
        return {
            "model": "local",
            "messages": list(wire_request.messages),
            "response_format": {"type": "json_object"},
            "json_schema": {"type": "object"},
            "grammar": "root ::= object",
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning_effort": "none",
        }

    payload = llama_adapter._host_tool_payload(object(), request, polluted_builder)

    assert payload["model"] == "local"
    assert payload["reasoning_effort"] == "none"
    assert not llama_adapter._NATIVE_GRAMMAR_CONTROL_KEYS.intersection(payload)


def test_host_tool_turn_survives_polluted_payload_builder(monkeypatch) -> None:
    adapter = llama_adapter.LlamaCppAdapter(
        AdapterConfig(role="planner", adapter="llama_cpp", model_id="test-model")
    )
    request = GenerationRequest(
        messages=[{"role": "user", "content": "inspect project"}],
        response_format="json",
        response_schema={"type": "object"},
        tools=(_tool_schema(),),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    captured: dict[str, object] = {}

    def polluted_builder(_adapter, wire_request):
        assert wire_request.response_format == "text"
        assert wire_request.response_schema is None
        assert wire_request.tools == ()
        return {
            "model": "local",
            "messages": list(wire_request.messages),
            "response_format": {"type": "json_object"},
            "json_schema": {"type": "object"},
            "grammar": "root ::= object",
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

    def completion(_server_url, payload):
        captured.update(payload)
        return {"content": '{"kind":"final","content":"done"}'}

    monkeypatch.setattr(llama_adapter, "_completion_message", completion)

    result = adapter._generate_host_tool_turn(
        "http://127.0.0.1:8910/v1",
        request,
        payload_builder=polluted_builder,
    )

    assert result.content == "done"
    assert not llama_adapter._NATIVE_GRAMMAR_CONTROL_KEYS.intersection(captured)
