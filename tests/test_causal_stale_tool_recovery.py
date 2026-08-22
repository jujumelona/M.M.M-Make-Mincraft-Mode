from __future__ import annotations

import pytest

from minecraft_mod_ai import causal_stale_tool_recovery_contract as recovery
from minecraft_mod_ai import causal_tool_frontier_contract
from minecraft_mod_ai.causal_frontier_adapter import FrontierExecutionGate
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


def _request(
    search: dict,
    edit: dict,
    *,
    forced_name: str = "search_code_rag",
) -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "repair source"},),
        tools=(search, edit),
        tool_validation_schemas=(search, edit),
        tool_choice={
            "type": "function",
            "function": {"name": forced_name},
        },
        parallel_tool_calls=False,
    )


def test_runtime_adapter_synthesizes_deterministic_read_without_resync_decode() -> None:
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

    result = adapter.generate_turn(_request(search, edit))

    assert [call.name for call in result.tool_calls] == ["search_code_rag"]
    assert result.tool_calls[0].arguments == {"query": "repair source"}
    # Only the original stale decode ran; the current read action is host-derived.
    assert len(inner.requests) == 1


def test_runtime_adapter_does_not_multiply_repeated_stale_resync_decodes() -> None:
    search = _schema("java_workspace_symbols")
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
        adapter.generate_turn(
            _request(search, edit, forced_name="java_workspace_symbols")
        )

    # One initial generation plus exactly one stale-frontier re-synchronization.
    assert len(inner.requests) == 2
    retry = inner.requests[1]
    assert [item["function"]["name"] for item in retry.tools] == [
        "java_workspace_symbols"
    ]


def test_owner_execution_gate_ignores_nested_context_frontier(monkeypatch) -> None:
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
    gate = FrontierExecutionGate()
    adapter = causal_tool_frontier_contract.CausalFrontierAdapter(
        inner,
        stage="generation",
        role="coder",
        require_fresh_evidence=False,
        execution_gate=gate,
        authorized_surface=(search, edit),
    )
    # Simulate a nested adapter publishing a different ContextVar frontier after this
    # loop already published search_code_rag into its owner-local execution gate.
    monkeypatch.setattr(
        recovery,
        "current_frontier_names",
        lambda: ("apply_source_edit",),
    )

    result = adapter.generate_turn(_request(search, edit))

    assert [call.name for call in result.tool_calls] == ["search_code_rag"]
    assert len(inner.requests) == 1
    assert gate.visible_names() == ("search_code_rag",)
