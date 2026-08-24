from __future__ import annotations

import json
from contextlib import nullcontext

import pytest

from minecraft_mod_ai import model_router
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
    ToolCall,
)
from minecraft_mod_ai.progress_aware_tool_loop import generate_with_tools


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _messages() -> tuple[dict, ...]:
    return (
        {
            "role": "user",
            "content": json.dumps(
                {
                    "phase": "implement_module",
                    "task": "Implement the approved Minecraft/Fabric feature.",
                }
            ),
        },
    )


class _Router:
    _agent_require_fresh_evidence = False

    @staticmethod
    def _generation_scope(config):
        del config
        return nullcontext()


def test_model_selects_read_then_edit_without_host_tool_forcing(monkeypatch) -> None:
    monkeypatch.setattr(model_router, "_agent_tool_round_limit", lambda: 8)
    monkeypatch.setattr(model_router, "_usable_rag_result", lambda result: bool(result))
    monkeypatch.setattr(
        model_router,
        "_execute_tool_waves",
        lambda calls, execute: tuple(execute(call) for call in calls),
    )

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            assert request.tool_choice in {"auto", None}
            if not request.tools:
                return GenerationResponse(content="implemented")
            if len(self.requests) == 1:
                return GenerationResponse(
                    tool_calls=(ToolCall(id="rag", name="search_code_rag", arguments={"query": "feature"}),)
                )
            if len(self.requests) == 2:
                return GenerationResponse(
                    tool_calls=(ToolCall(id="edit", name="apply_source_edit", arguments={}),)
                )
            return GenerationResponse(content="implemented")

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(self, stage: str, name: str, arguments: dict) -> dict:
            assert stage == "generation"
            self.calls.append(name)
            if name == "search_code_rag":
                return {
                    "hits": [{"path": "src/Main.java"}],
                    "receipt": {"result_count": 1, "coverage_score": 1.0, "relevance_score": 1.0},
                }
            if name == "apply_source_edit":
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [{"path": "src/Main.java", "operation": "edit"}],
                }
            raise AssertionError(name)

    adapter = Adapter()
    runtime = Runtime()
    request = GenerationRequest(
        messages=_messages(),
        tools=(_schema("search_code_rag"), _schema("apply_source_edit")),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    result = generate_with_tools(
        _Router(),
        config=object(),
        adapter=adapter,
        request=request,
        runtime=runtime,
        stage="generation",
        role="coder",
    )

    assert result == "implemented"
    assert runtime.calls == ["search_code_rag", "apply_source_edit"]
    assert all(item.tool_choice == "auto" for item in adapter.requests)


def test_prose_only_implementation_fails_closed_without_forced_retry(monkeypatch) -> None:
    monkeypatch.setattr(model_router, "_agent_tool_round_limit", lambda: 8)

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            return GenerationResponse(content="I am done")

    adapter = Adapter()
    request = GenerationRequest(
        messages=_messages(),
        tools=(_schema("search_code_rag"), _schema("apply_source_edit")),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    with pytest.raises(ModelConfigurationError, match="before a reviewed source mutation"):
        generate_with_tools(
            _Router(),
            config=object(),
            adapter=adapter,
            request=request,
            runtime=object(),
            stage="generation",
            role="coder",
        )

    assert len(adapter.requests) == 1
    assert adapter.requests[0].tool_choice == "auto"
