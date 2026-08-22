from __future__ import annotations

import pytest

from minecraft_mod_ai import causal_tool_frontier_contract
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
    ToolCall,
)


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def _call(name: str, value: str) -> ToolCall:
    raw = '{"query":"' + value + '"}'
    return ToolCall(
        id="call-" + name,
        name=name,
        arguments={"query": value},
        raw_arguments=raw,
    )


def _request(search: dict, edit: dict) -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "repair source"},),
        tools=(search, edit),
        tool_validation_schemas=(search, edit),
        tool_choice={
            "type": "function",
            "function": {"name": "search_code_rag"},
        },
        parallel_tool_calls=False,
    )


def test_runtime_adapter_retries_stale_authorized_call_on_visible_frontier() -> None:
    search = _schema("search_code_rag")
    edit = _schema("apply_source_edit")
    stale = _call("apply_source_edit", "old edit")
    legal = _call("search_code_rag", "refresh source evidence")

    class Inner:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return GenerationResponse(tool_calls=(stale,))
            return GenerationResponse(tool_calls=(legal,))

    inner = Inner()
    adapter = causal_tool_frontier_contract.CausalFrontierAdapter(
        inner,
        stage="generation",
        role="coder",
        require_fresh_evidence=False,
        authorized_surface=(search, edit),
    )

    result = adapter.generate_turn(_request(search, edit))

    assert result.tool_calls == (legal,)
    assert len(inner.requests) == 2
    retry = inner.requests[1]
    assert [item["function"]["name"] for item in retry.tools] == ["search_code_rag"]
    assert retry.tool_choice == {
        "type": "function",
        "function": {"name": "search_code_rag"},
    }
    assert "stale" in str(retry.messages[-1]["content"]).casefold()


def test_runtime_adapter_does_not_multiply_repeated_stale_resync_decodes() -> None:
    search = _schema("search_code_rag")
    edit = _schema("apply_source_edit")
    stale = _call("apply_source_edit", "old edit")

    class Inner:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            return GenerationResponse(tool_calls=(stale,))

    inner = Inner()
    adapter = causal_tool_frontier_contract.CausalFrontierAdapter(
        inner,
        stage="generation",
        role="coder",
        require_fresh_evidence=False,
        authorized_surface=(search, edit),
    )

    with pytest.raises(ModelConfigurationError, match="single causal-frontier"):
        adapter.generate_turn(_request(search, edit))

    # One initial generation plus exactly one stale-frontier re-synchronization.
    assert len(inner.requests) == 2
    retry = inner.requests[1]
    assert [item["function"]["name"] for item in retry.tools] == ["search_code_rag"]
