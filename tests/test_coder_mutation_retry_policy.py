from __future__ import annotations

import json

from minecraft_mod_ai.coder_tool_route_integrity_contract import _WritableProgressAdapter
from minecraft_mod_ai.model_adapters import GenerationRequest, GenerationResponse, ToolCall


def _tool(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _messages() -> tuple[dict[str, object], ...]:
    return ({"role": "user", "content": json.dumps({"phase": "implement_module"})},)


def test_writable_compat_adapter_does_not_override_model_tool_choice() -> None:
    requests: list[GenerationRequest] = []

    class Inner:
        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            requests.append(request)
            return GenerationResponse(
                tool_calls=(ToolCall(id="model-choice", name="apply_source_edit"),)
            )

    request = GenerationRequest(
        messages=_messages(),
        tools=(_tool("apply_source_edit"), _tool("apply_source_patch")),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    turn = _WritableProgressAdapter(Inner()).generate_turn(request)

    assert len(requests) == 1
    assert requests[0] is request
    assert requests[0].tool_choice == "auto"
    assert requests[0].parallel_tool_calls is True
    assert [call.name for call in turn.tool_calls] == ["apply_source_edit"]


def test_writable_compat_adapter_owns_no_retry_or_failover_policy() -> None:
    requests: list[GenerationRequest] = []

    class Inner:
        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            requests.append(request)
            return GenerationResponse(content="model decided not to call a tool")

    request = GenerationRequest(
        messages=_messages(),
        tools=(_tool("apply_source_patch"), _tool("apply_source_edit")),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    response = _WritableProgressAdapter(Inner()).generate_turn(request)

    assert response.content == "model decided not to call a tool"
    assert requests == [request]
