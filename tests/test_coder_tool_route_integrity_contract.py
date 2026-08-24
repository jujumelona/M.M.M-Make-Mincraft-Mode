from __future__ import annotations

import json

from minecraft_mod_ai.coder_tool_route_integrity_contract import (
    _WritableProgressAdapter,
    _is_implementation_request,
    _run_with_dynamic_frontier,
    _source_mutation_applied,
    _user_only_request_query,
)
from minecraft_mod_ai.model_adapters import GenerationRequest, GenerationResponse
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


def _messages() -> tuple[dict, ...]:
    return (
        {"role": "system", "content": "injected routing boilerplate"},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "phase": "implement_module",
                    "task": "Implement the approved feature.",
                }
            ),
        },
    )


def test_route_integrity_no_longer_wraps_live_tool_loop() -> None:
    current = ModelRouter._generate_with_tools
    seen: set[int] = set()
    implementations: list[str] = []
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "__code__", None)
        implementations.append(str(getattr(code, "co_filename", "")).replace("\\", "/"))
        current = getattr(current, "__wrapped__", None)
    assert not any(path.endswith("/coder_tool_route_integrity_contract.py") for path in implementations)


def test_structured_implementation_intent_uses_user_turn_only() -> None:
    query = _user_only_request_query(_messages())
    assert "implement_module" in query
    assert "Implement the approved feature" in query
    assert "injected routing boilerplate" not in query
    assert _is_implementation_request(_messages()) is True


def test_writable_adapter_is_transparent_and_keeps_auto_choice() -> None:
    captured: list[GenerationRequest] = []

    class Inner:
        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            captured.append(request)
            return GenerationResponse(content="model-owned decision")

    request = GenerationRequest(
        messages=_messages(),
        tools=(_schema("search_code_rag"), _schema("apply_source_edit")),
        tool_choice="auto",
        parallel_tool_calls=True,
        task="task-sentinel",
        prompt="prompt-sentinel",
        metadata={"trace": "sentinel"},
    )
    response = _WritableProgressAdapter(Inner()).generate_turn(request)

    assert response.content == "model-owned decision"
    assert captured == [request]
    assert captured[0].tool_choice == "auto"
    assert captured[0].parallel_tool_calls is True
    assert captured[0].task == "task-sentinel"
    assert captured[0].metadata == {"trace": "sentinel"}


def test_legacy_dynamic_frontier_entry_is_direct_passthrough() -> None:
    captured = {}

    def current(router, *, config, adapter, request, runtime, stage, role):
        captured.update(
            router=router,
            config=config,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )
        return "ok"

    sentinel = object()
    request = GenerationRequest(messages=_messages(), tools=(_schema("apply_source_edit"),), tool_choice="auto")
    assert _run_with_dynamic_frontier(
        current,
        sentinel,
        config="config",
        adapter="adapter",
        request=request,
        runtime="runtime",
        stage="generation",
        role="coder",
    ) == "ok"
    assert captured["router"] is sentinel
    assert captured["request"] is request
    assert captured["adapter"] == "adapter"


def test_canonical_applied_patch_receipt_is_still_completion_proof() -> None:
    messages = (
        *_messages(),
        {
            "role": "tool",
            "name": "apply_source_edit",
            "tool_call_id": "edit-1",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "apply_source_edit",
                    "result": {
                        "schema_version": "mmm/source-patch-receipt-v1",
                        "status": "APPLIED",
                        "operations": [{"path": "src/Main.java", "operation": "edit"}],
                    },
                }
            ),
        },
    )
    assert _source_mutation_applied(messages) is True
