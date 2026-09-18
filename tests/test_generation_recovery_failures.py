from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import GenerationRequest, ToolDefinition
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    _native_tool_generation_response,
)


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )


def test_all_rejected_non_visible_tool_call_is_recoverable_observation() -> None:
    request = GenerationRequest(
        tools=(_tool("recover_context"),),
        tool_choice="required",
        parallel_tool_calls=False,
    )
    message = {
        "content": "",
        "tool_calls": [
            {
                "id": "call_stale_edit",
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "arguments": "{}",
                },
            }
        ],
    }

    response = _native_tool_generation_response(message, request)

    assert len(response.tool_calls) == 1
    rejection = response.tool_calls[0]
    assert rejection.name == "__mmm_rejected_tool_call__"
    assert rejection.arguments["original_tool"] == "apply_source_edit"
    assert rejection.arguments["failure_code"] == "TOOL_NOT_VISIBLE"
    assert "non-visible tool" in str(rejection.arguments["error"])
