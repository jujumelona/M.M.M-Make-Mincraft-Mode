from __future__ import annotations

import json
from dataclasses import replace
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
    _source_mutation_applied,
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


def _applied_patch_message() -> dict:
    return {
        "role": "tool",
        "name": "apply_source_patch",
        "tool_call_id": "patch-1",
        "content": json.dumps(
            {
                "ok": True,
                "tool": "apply_source_patch",
                "result": {
                    "structured_content": {
                        "schema_version": "mmm/source-patch-receipt-v1",
                        "status": "APPLIED",
                        "operations": [
                            {
                                "path": "src/main/java/example/Test.java",
                                "operation": "create",
                                "before_sha256": None,
                                "after_sha256": "sha256:" + "1" * 64,
                            }
                        ],
                    }
                },
            }
        ),
    }


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
    assert getattr(method, "_mmm_writable_coder_mutation_completion_invariant", False) is True


def test_tool_routing_query_ignores_external_mcp_system_boilerplate() -> None:
    query = small_agent._request_query(_implement_messages())
    assert "Implement the approved feature" in query
    assert "MMM capability routing supports external MCP" not in query


def test_implement_phase_requires_source_edit_terminal_despite_external_metadata() -> None:
    query = small_agent._request_query(_implement_messages())
    assert "external MCP" not in query
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

    def final_loop(router, *, config, adapter, request, runtime, stage, role):
        del router, runtime
        captured["config"] = config
        captured["adapter"] = adapter
        captured["tools"] = tuple(
            item["function"]["name"] for item in request.tools
        )
        captured["stage"] = stage
        captured["role"] = role
        return "ok"

    class Router:
        _agent_require_fresh_evidence = True

    config = object()
    try:
        result = _run_with_dynamic_frontier(
            final_loop,
            Router(),
            config=config,
            adapter=object(),
            request=_request(initial),
            runtime=object(),
            stage="generation",
            role="coder",
        )
    finally:
        remember_authorized_tools(())

    assert result == "ok"
    assert captured["config"] is config
    assert isinstance(captured["adapter"], CausalFrontierAdapter)
    assert tuple(
        item["function"]["name"]
        for item in captured["adapter"].authorized_surface
    ) == captured["tools"]
    assert captured["adapter"].request_template is not None
    assert tuple(
        item["function"]["name"]
        for item in captured["adapter"].request_template.tool_validation_schemas
    ) == captured["tools"]
    assert captured["tools"] == tuple(
        item["function"]["name"] for item in complete
    )
    assert "apply_source_patch" in captured["tools"]
    assert captured["stage"] == "generation"
    assert captured["role"] == "coder"


def test_live_coder_freezes_surface_across_nested_selector_state_changes() -> None:
    complete = (
        _schema("search_project_rag"),
        _schema("search_code_rag"),
        _schema("apply_source_edit"),
    )
    nested_surface = (_schema("search_project_rag"),)
    rag_observation = {
        "role": "tool",
        "name": "search_code_rag",
        "tool_call_id": "rag-1",
        "content": json.dumps(
            {
                "ok": True,
                "result": {
                    "hits": [{"path": "src/main/java/example/Test.java"}],
                    "receipt": {
                        "result_count": 1,
                        "coverage_score": 1.0,
                        "relevance_score": 1.0,
                    },
                },
            }
        ),
    }
    safe_failed_edit = {
        "role": "tool",
        "name": "apply_source_edit",
        "tool_call_id": "edit-1",
        "content": json.dumps(
            {
                "ok": False,
                "error": (
                    "AgentToolRuntimeError: exact edit did not match "
                    "[mmm-workspace-impact:unchanged]"
                ),
            }
        ),
    }
    model_requests = []

    class Adapter:
        def generate_turn(self, request):
            model_requests.append(request)
            forced = request.tool_choice["function"]["name"]
            return SimpleNamespace(
                tool_calls=(SimpleNamespace(name=forced),),
                content="",
            )

    def exercise(router, *, config, adapter, request, runtime, stage, role):
        del router, config, runtime, stage, role
        first = replace(
            request,
            messages=(*request.messages, rag_observation),
        )
        adapter.generate_turn(first)

        # A nested selector writes compatibility ContextVars during the model/tool
        # loop.  It must not replace this coder loop's frozen authorization surface.
        remember_authorized_tools(nested_surface, {"search_project_rag": 0})
        second = replace(
            request,
            messages=(*first.messages, safe_failed_edit),
        )
        adapter.generate_turn(second)
        return "ok"

    remember_authorized_tools(complete, {"apply_source_edit": 0})
    try:
        result = _run_with_dynamic_frontier(
            exercise,
            SimpleNamespace(_agent_require_fresh_evidence=True),
            config=object(),
            adapter=Adapter(),
            request=_request(complete),
            runtime=object(),
            stage="generation",
            role="coder",
        )
    finally:
        remember_authorized_tools(())

    assert result == "ok"
    assert len(model_requests) == 2
    assert [
        request.tool_choice["function"]["name"] for request in model_requests
    ] == ["apply_source_edit", "apply_source_edit"]


def test_writable_progress_keeps_prerequisite_auto_then_forces_only_after_prose() -> None:
    requests = []

    class Adapter:
        def generate_turn(self, request):
            requests.append(request)
            if request.tool_choice == "auto":
                return SimpleNamespace(tool_calls=(), content="draft before retrieval")
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
            tools=(
                _schema("search_code_rag"),
                _schema("java_workspace_symbols"),
            ),
            tool_choice="auto",
            parallel_tool_calls=True,
        )
    )

    assert len(requests) == 2
    assert requests[0].tool_choice == "auto"
    assert requests[0].parallel_tool_calls is True
    assert requests[1].tool_choice == {
        "type": "function",
        "function": {"name": "search_code_rag"},
    }
    assert requests[1].parallel_tool_calls is False
    assert turn.tool_calls[0].name == "search_code_rag"


def test_writable_progress_does_not_force_cooperative_prerequisite() -> None:
    requests = []

    class Adapter:
        def generate_turn(self, request):
            requests.append(request)
            return SimpleNamespace(
                tool_calls=(SimpleNamespace(name="java_workspace_symbols"),),
                content="",
            )

    wrapped = _WritableProgressAdapter(Adapter())
    turn = wrapped.generate_turn(
        GenerationRequest(
            messages=_implement_messages(),
            tools=(
                _schema("search_code_rag"),
                _schema("java_workspace_symbols"),
            ),
            tool_choice="auto",
            parallel_tool_calls=True,
        )
    )

    assert len(requests) == 1
    assert requests[0].tool_choice == "auto"
    assert turn.tool_calls[0].name == "java_workspace_symbols"


def test_writable_progress_forces_visible_causal_action() -> None:
    captured = {}

    class Adapter:
        def generate_turn(self, request):
            captured["request"] = request
            captured["tool_choice"] = request.tool_choice
            chosen = request.tool_choice["function"]["name"]
            return SimpleNamespace(
                tool_calls=(SimpleNamespace(name=chosen),),
                content="",
            )

    edit = _schema("apply_source_patch")
    historical = _schema("search_code_rag")
    wrapped = _WritableProgressAdapter(Adapter())
    turn = wrapped.generate_turn(
        GenerationRequest(
            messages=_implement_messages(),
            media_paths=(),
            response_format="text",
            response_schema=None,
            tools=(edit,),
            tool_validation_schemas=(edit, historical),
            tool_choice="auto",
            parallel_tool_calls=True,
            task="task-sentinel",
            prompt="prompt-sentinel",
            metadata={"trace": "metadata-sentinel"},
        )
    )
    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "apply_source_patch"},
    }
    assert turn.tool_calls[0].name == "apply_source_patch"
    assert captured["request"].tool_validation_schemas == (edit, historical)
    assert captured["request"].task == "task-sentinel"
    assert captured["request"].prompt == "prompt-sentinel"
    assert captured["request"].metadata == {"trace": "metadata-sentinel"}


def test_writable_progress_rejects_prose_only_mutation_turn() -> None:
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


def test_writable_progress_rejects_final_synthesis_before_applied_patch() -> None:
    class Adapter:
        def generate_turn(self, request):
            del request
            return SimpleNamespace(tool_calls=(), content="done")

    wrapped = _WritableProgressAdapter(Adapter())
    with pytest.raises(ModelConfigurationError, match="before any reviewed source mutation"):
        wrapped.generate_turn(
            GenerationRequest(
                messages=_implement_messages(),
                tools=(),
                tool_choice=None,
                parallel_tool_calls=False,
            )
        )


def test_applied_patch_receipt_unlocks_final_synthesis() -> None:
    messages = (*_implement_messages(), _applied_patch_message())
    assert _source_mutation_applied(messages) is True

    class Adapter:
        def generate_turn(self, request):
            assert request.tools == ()
            return SimpleNamespace(tool_calls=(), content="implemented")

    wrapped = _WritableProgressAdapter(Adapter())
    turn = wrapped.generate_turn(
        GenerationRequest(
            messages=messages,
            tools=(),
            tool_choice=None,
            parallel_tool_calls=False,
        )
    )
    assert turn.content == "implemented"


def test_apply_source_patch_transport_success_without_receipt_is_not_completion() -> None:
    messages = (
        *_implement_messages(),
        {
            "role": "tool",
            "name": "apply_source_patch",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "apply_source_patch",
                    "result": {"status": "PASS"},
                }
            ),
        },
    )
    assert _source_mutation_applied(messages) is False


def test_writable_coder_without_mutation_surface_fails_before_model_loop() -> None:
    initial = (_schema("external_mcp_call"),)
    remember_authorized_tools(initial, {"external_mcp_call": 0})
    called = False

    def must_not_run(router, *, config, adapter, request, runtime, stage, role):
        nonlocal called
        del router, config, adapter, request, runtime, stage, role
        called = True
        return "unreachable"

    class Router:
        _agent_require_fresh_evidence = True

    try:
        with pytest.raises(ModelConfigurationError, match="no authorized source-mutation tool"):
            _run_with_dynamic_frontier(
                must_not_run,
                Router(),
                config=object(),
                adapter=object(),
                request=_request(initial),
                runtime=object(),
                stage="generation",
                role="coder",
            )
    finally:
        remember_authorized_tools(())

    assert called is False
