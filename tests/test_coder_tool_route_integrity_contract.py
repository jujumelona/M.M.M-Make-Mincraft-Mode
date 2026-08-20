from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import causal_tool_frontier_contract as causal
from minecraft_mod_ai import small_model_max_agent_contract as small_agent
from minecraft_mod_ai.causal_frontier_adapter import (
    CausalFrontierAdapter,
    remember_authorized_tools,
)
from minecraft_mod_ai.coder_tool_route_integrity_contract import (
    _WritableProgressAdapter,
    _run_with_dynamic_frontier,
)
from minecraft_mod_ai.model_adapters import GenerationRequest, ModelConfigurationError
from minecraft_mod_ai.model_router import ModelRouter


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _implement_messages() -> tuple[dict, ...]:
    return (
        {
            "role": "system",
            "content": "MMM capability routing supports external MCP and other tools.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "phase": "implement_module",
                    "task": "Implement the approved feature in the current project.",
                    "research_context": {"transport": "external MCP may be available"},
                }
            ),
        },
    )


def _request(tools: tuple[dict, ...]) -> GenerationRequest:
    return GenerationRequest(
        messages=_implement_messages(),
        media_paths=(),
        response_format="text",
        response_schema=None,
        tools=tools,
        tool_choice="auto" if tools else None,
        parallel_tool_calls=bool(tools),
    )


def test_final_runtime_recomposes_progress_and_dynamic_causal_frontier() -> None:
    method = ModelRouter._generate_with_tools
    assert getattr(method, "_mmm_progress_aware_causal_composed", False) is True
    assert getattr(method, "_mmm_dynamic_causal_frontier", False) is True
    assert getattr(method, "_mmm_writable_coder_fail_closed", False) is True
    assert getattr(method, "_mmm_writable_coder_progress_forced", False) is True


def test_tool_routing_query_ignores_external_mcp_system_boilerplate() -> None:
    query = small_agent._request_query(_implement_messages())
    assert "Implement the approved feature" in query
    assert "MMM capability routing supports external MCP" not in query


def test_implement_phase_requires_source_edit_terminal_despite_external_metadata() -> None:
    query = small_agent._request_query(_implement_messages())
    assert "external MCP" in query
    assert causal.goals_for_query(query) == ("repair",)


def test_initial_one_tool_frontier_recovers_complete_mutation_surface() -> None:
    initial = (_schema("external_mcp_call"),)
    complete = (
        _schema("inspect_existing_mod"),
        _schema("search_code_rag"),
        _schema("apply_source_patch"),
        _schema("external_mcp_call"),
    )
    remember_authorized_tools(complete, {"external_mcp_call": 0})
    captured = {}

    def final_loop(router, *, adapter, request, runtime, stage, role):
        del router, runtime
        captured["adapter"] = adapter
        captured["tools"] = tuple(
            item["function"]["name"] for item in request.tools
        )
        captured["stage"] = stage
        captured["role"] = role
        return "ok"

    class Router:
        _agent_require_fresh_evidence = True

    try:
        result = _run_with_dynamic_frontier(
            final_loop,
            Router(),
            adapter=object(),
            request=_request(initial),
            runtime=object(),
            stage="generation",
            role="coder",
        )
    finally:
        remember_authorized_tools(())

    assert result == "ok"
    assert isinstance(captured["adapter"], CausalFrontierAdapter)
    assert captured["tools"] == tuple(
        item["function"]["name"] for item in complete
    )
    assert "apply_source_patch" in captured["tools"]
    assert captured["stage"] == "generation"
    assert captured["role"] == "coder"


def test_writable_progress_forces_visible_causal_action() -> None:
    captured = {}

    class Adapter:
        def generate_turn(self, request):
            captured["tool_choice"] = request.tool_choice
            chosen = request.tool_choice["function"]["name"]
            return SimpleNamespace(
                tool_calls=(SimpleNamespace(name=chosen),),
                content="",
            )

    wrapped = _WritableProgressAdapter(Adapter())
    turn = wrapped.generate_turn(
        GenerationRequest(
            messages=_implement_messages(),
            media_paths=(),
            response_format="text",
            response_schema=None,
            tools=(_schema("apply_source_patch"),),
            tool_choice="auto",
            parallel_tool_calls=True,
        )
    )
    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "apply_source_patch"},
    }
    assert turn.tool_calls[0].name == "apply_source_patch"


def test_writable_progress_rejects_prose_only_turn() -> None:
    class Adapter:
        def generate_turn(self, request):
            del request
            return SimpleNamespace(tool_calls=(), content="done")

    wrapped = _WritableProgressAdapter(Adapter())
    with pytest.raises(ModelConfigurationError, match="prose-only implementation turn"):
        wrapped.generate_turn(
            GenerationRequest(
                messages=_implement_messages(),
                media_paths=(),
                response_format="text",
                response_schema=None,
                tools=(_schema("apply_source_patch"),),
                tool_choice="auto",
                parallel_tool_calls=True,
            )
        )


def test_writable_coder_without_mutation_surface_fails_before_model_loop() -> None:
    initial = (_schema("external_mcp_call"),)
    remember_authorized_tools(initial, {"external_mcp_call": 0})
    called = False

    def must_not_run(router, *, adapter, request, runtime, stage, role):
        nonlocal called
        del router, adapter, request, runtime, stage, role
        called = True
        return "unreachable"

    class Router:
        _agent_require_fresh_evidence = True

    try:
        with pytest.raises(ModelConfigurationError, match="no authorized source-mutation tool"):
            _run_with_dynamic_frontier(
                must_not_run,
                Router(),
                adapter=object(),
                request=_request(initial),
                runtime=object(),
                stage="generation",
                role="coder",
            )
    finally:
        remember_authorized_tools(())

    assert called is False
