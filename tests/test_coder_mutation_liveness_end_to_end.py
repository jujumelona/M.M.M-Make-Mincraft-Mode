from __future__ import annotations

import json
from contextlib import nullcontext

from minecraft_mod_ai import model_router
from minecraft_mod_ai.causal_frontier_adapter import (
    CausalFrontierAdapter,
    remember_authorized_tools,
)
from minecraft_mod_ai.coder_tool_route_integrity_contract import (
    _run_with_dynamic_frontier,
    _WritableProgressAdapter,
)
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
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


def _query_schema(name: str) -> dict:
    schema = _schema(name)
    schema["function"]["parameters"] = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1}},
        "required": ["query"],
        "additionalProperties": False,
    }
    return schema


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


class _BaseAdapter:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        if not request.tools:
            return GenerationResponse(content="implemented after source mutation")

        if request.tool_choice == "auto":
            return GenerationResponse(content="I can implement this from the context.")

        name = str(request.tool_choice["function"]["name"])
        if name == "search_code_rag":
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="search-1",
                        name=name,
                        arguments={"query": "approved feature implementation"},
                    ),
                )
            )
        if name == "apply_source_patch":
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="patch-1",
                        name=name,
                        arguments={
                            "files": [
                                {
                                    "path": "src/main/java/example/Test.java",
                                    "content": "package example; public final class Test {}\n",
                                }
                            ]
                        },
                    ),
                )
            )
        raise AssertionError(f"unexpected forced tool: {name}")


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, stage: str, name: str, arguments: dict) -> dict:
        assert stage == "generation"
        self.calls.append(name)
        if name == "search_code_rag":
            return {
                "hits": [{"path": "src/main/java/example/Existing.java"}],
                "receipt": {
                    "result_count": 1,
                    "coverage_score": 1.0,
                    "relevance_score": 1.0,
                },
            }
        if name == "apply_source_patch":
            return {
                "truncated": True,
                "original_bytes": 256_000,
                "preview": "{...large source-patch observation omitted...}",
            }
        raise AssertionError(f"unexpected runtime call: {name} {arguments}")


def test_prose_refusal_cannot_finish_before_source_patch_even_when_result_is_truncated(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_router, "_agent_tool_round_limit", lambda: 8)
    monkeypatch.setattr(model_router, "_usable_rag_result", lambda result: bool(result))
    monkeypatch.setattr(
        model_router,
        "_execute_tool_waves",
        lambda calls, execute: tuple(execute(call) for call in calls),
    )

    base = _BaseAdapter()
    runtime = _Runtime()
    adapter = CausalFrontierAdapter(
        _WritableProgressAdapter(base),
        stage="generation",
        role="coder",
        require_fresh_evidence=False,
        frontier_limit=3,
    )
    request = GenerationRequest(
        messages=_messages(),
        tools=(
            _schema("search_code_rag"),
            _schema("apply_source_patch"),
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    class Router:
        _agent_require_fresh_evidence = False

        @staticmethod
        def _generation_scope(config):
            del config
            return nullcontext()

    config = object()
    remember_authorized_tools(
        request.tools,
        {"search_code_rag": 0, "apply_source_patch": 1},
    )
    try:
        result = generate_with_tools(
            Router(),
            config=config,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage="generation",
            role="coder",
        )
    finally:
        remember_authorized_tools(())

    assert result == "implemented after source mutation"
    assert runtime.calls == ["search_code_rag", "apply_source_patch"]
    choices = [item.tool_choice for item in base.requests]
    assert choices[0] == "auto"
    assert choices[1] == {
        "type": "function",
        "function": {"name": "search_code_rag"},
    }
    assert choices[2] == {
        "type": "function",
        "function": {"name": "apply_source_patch"},
    }
    assert base.requests[-1].tools == ()


def test_unpatched_adapter_gets_canonical_forced_edit_protocol_retry() -> None:
    class ProseThenEditAdapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return GenerationResponse(content="I will edit the source now.")
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="edit-after-correction",
                        name="apply_source_edit",
                        arguments={},
                    ),
                )
            )

    inner = ProseThenEditAdapter()
    request = GenerationRequest(
        messages=_messages(),
        tools=(_schema("apply_source_edit"),),
        tool_choice={
            "type": "function",
            "function": {"name": "apply_source_edit"},
        },
        parallel_tool_calls=False,
    )

    turn = _WritableProgressAdapter(inner).generate_turn(request)

    assert [call.name for call in turn.tool_calls] == ["apply_source_edit"]
    assert len(inner.requests) == 2
    assert all(item.tool_choice == "required" for item in inner.requests)
    assert all(len(item.tools) == 1 for item in inner.requests)
    assert inner.requests[0].parallel_tool_calls is False
    assert inner.requests[1].parallel_tool_calls is False
    assert "previous assistant turn did not satisfy" in str(
        inner.requests[1].messages[-1]["content"]
    ).casefold()


def test_drift_refresh_discards_stale_edit_and_finishes_without_resync_decode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_router, "_agent_tool_round_limit", lambda: 10)
    monkeypatch.setattr(model_router, "_usable_rag_result", lambda result: bool(result))
    monkeypatch.setattr(
        model_router,
        "_execute_tool_waves",
        lambda calls, execute: tuple(execute(call) for call in calls),
    )

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []
            self.emissions: list[str] = []
            self.edit_turn = 0
            self.final_messages: tuple[dict, ...] = ()

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            if not request.tools:
                self.emissions.append("summary")
                self.final_messages = tuple(request.messages)
                return GenerationResponse(content="implemented after drift refresh")

            if request.tool_choice == "auto":
                failed_edit_seen = any(
                    str(message.get("role", "")) == "tool"
                    and str(message.get("name", "")) == "apply_source_edit"
                    and '"ok": false' in str(message.get("content", "")).casefold()
                    for message in request.messages
                )
                name = "apply_source_edit" if failed_edit_seen else "search_code_rag"
            else:
                name = str(request.tool_choice["function"]["name"])

            self.emissions.append(name)
            if name == "search_code_rag":
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id="initial-rag",
                            name=name,
                            arguments={"query": "initial exact source evidence"},
                        ),
                    )
                )
            if name == "apply_source_edit":
                self.edit_turn += 1
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id=f"edit-{self.edit_turn}",
                            name=name,
                            arguments={
                                "operation": "replace_exact",
                                "path": "src/main/java/example/Test.java",
                                "old": "oldValue",
                                "new": "newValue",
                            },
                        ),
                    )
                )
            raise AssertionError(f"unexpected model action: {name}")

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.search_turn = 0
            self.edit_turn = 0
            self.applied_receipt = {
                "schema_version": "mmm/source-patch-receipt-v1",
                "status": "APPLIED",
                "operations": [
                    {
                        "path": "src/main/java/example/Test.java",
                        "operation": "edit",
                        "after_sha256": "sha256:" + "1" * 64,
                    }
                ],
            }

        def call(self, stage: str, name: str, arguments: dict) -> dict:
            assert stage == "generation"
            self.calls.append(name)
            if name == "search_code_rag":
                self.search_turn += 1
                return {
                    "hits": [
                        {
                            "path": "src/main/java/example/Test.java",
                            "revision": self.search_turn,
                        }
                    ],
                    "receipt": {
                        "result_count": 1,
                        "coverage_score": 1.0,
                        "relevance_score": 1.0,
                    },
                    "query": arguments["query"],
                }
            if name == "apply_source_edit":
                self.edit_turn += 1
                if self.edit_turn == 1:
                    raise RuntimeError(
                        "exact source anchor drifted [workspace_impact=drift]"
                    )
                return self.applied_receipt
            raise AssertionError(f"unexpected runtime call: {name}")

    class Router:
        _agent_require_fresh_evidence = True

        @staticmethod
        def _generation_scope(config):
            del config
            return nullcontext()

    tools = (
        _query_schema("search_code_rag"),
        _schema("apply_source_edit"),
    )
    request = GenerationRequest(
        messages=_messages(),
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=True,
        task="Implement the approved Minecraft/Fabric feature.",
    )
    adapter = Adapter()
    runtime = Runtime()
    remember_authorized_tools(
        tools,
        {"search_code_rag": 0, "apply_source_edit": 1},
    )
    try:
        result = _run_with_dynamic_frontier(
            generate_with_tools,
            Router(),
            config=object(),
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage="generation",
            role="coder",
        )
    finally:
        remember_authorized_tools(())

    assert result == "implemented after drift refresh"
    assert runtime.calls == [
        "search_code_rag",
        "apply_source_edit",
        "search_code_rag",
        "apply_source_edit",
    ]
    assert adapter.emissions == [
        "search_code_rag",
        "apply_source_edit",
        "apply_source_edit",
        "apply_source_edit",
        "summary",
    ]
    assert len(adapter.requests) == 5
    assert runtime.edit_turn == 2
    assert any(
        "mmm/source-patch-receipt-v1" in str(message.get("content", ""))
        for message in adapter.final_messages
        if str(message.get("role", "")) == "tool"
        and str(message.get("name", "")) == "apply_source_edit"
    )
