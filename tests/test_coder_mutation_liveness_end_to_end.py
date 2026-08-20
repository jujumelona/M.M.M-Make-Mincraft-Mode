from __future__ import annotations

import json

from minecraft_mod_ai import model_router
from minecraft_mod_ai.causal_frontier_adapter import (
    CausalFrontierAdapter,
    remember_authorized_tools,
)
from minecraft_mod_ai.coder_tool_route_integrity_contract import _WritableProgressAdapter
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
            # Reproduce the recurring small-model failure: the model sees a legal
            # prerequisite tool but initially tries to answer in prose instead.
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
        raise AssertionError(f"unexpected runtime call: {name} {arguments}")


def test_prose_refusal_cannot_finish_before_source_patch(monkeypatch) -> None:
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

    remember_authorized_tools(
        request.tools,
        {"search_code_rag": 0, "apply_source_patch": 1},
    )
    try:
        result = generate_with_tools(
            type("Router", (), {"_agent_require_fresh_evidence": False})(),
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
