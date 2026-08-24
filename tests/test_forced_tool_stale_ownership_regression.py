from __future__ import annotations

import pytest

from minecraft_mod_ai.forced_tool_execution_contract import _install_adapter_class
from minecraft_mod_ai.model_adapters.base import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
    ToolCall,
)


def _tool_schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def _turn(name: str) -> GenerationResponse:
    return GenerationResponse(
        tool_calls=(
            ToolCall(
                id=f"call_{name}",
                name=name,
                arguments={"query": "evidence"},
                raw_arguments='{"query":"evidence"}',
            ),
        )
    )


def _request(*, forced: str, stale: str) -> GenerationRequest:
    forced_schema = _tool_schema(forced)
    stale_schema = _tool_schema(stale)
    return GenerationRequest(
        messages=({"role": "user", "content": "apply the required action"},),
        tools=(forced_schema,),
        tool_validation_schemas=(forced_schema, stale_schema),
        tool_choice={"type": "function", "function": {"name": forced}},
        parallel_tool_calls=False,
    )


def test_validation_only_stale_call_inside_forced_action_uses_one_local_correction() -> None:
    forced = "required_workspace_action"
    stale = "stale_workspace_action"

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []
            self.responses = [_turn(stale), _turn(forced)]

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            return self.responses[len(self.requests) - 1]

    _install_adapter_class(
        Adapter,
        transport_name="Regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    result = adapter.generate_turn(_request(forced=forced, stale=stale))

    assert [call.name for call in result.tool_calls] == [forced]
    assert len(adapter.requests) == 2
    assert all(request.tool_choice == "required" for request in adapter.requests)
    assert all(
        [schema["function"]["name"] for schema in request.tools] == [forced]
        for request in adapter.requests
    )
    assert adapter.requests[1].messages[-1]["role"] == "system"
    assert "only available function" in adapter.requests[1].messages[-1]["content"]


def test_forced_action_stale_mismatch_remains_bounded_after_one_correction() -> None:
    forced = "required_workspace_action"
    stale = "stale_workspace_action"

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            return _turn(stale)

    _install_adapter_class(
        Adapter,
        transport_name="Regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    with pytest.raises(ModelConfigurationError, match="after one protocol correction"):
        adapter.generate_turn(_request(forced=forced, stale=stale))

    assert len(adapter.requests) == 2
