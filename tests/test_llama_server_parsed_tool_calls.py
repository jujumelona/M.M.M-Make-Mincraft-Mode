from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    _native_tool_generation_response,
)


def _strict_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
                "additionalProperties": False,
            },
        },
    }


def _request() -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "look it up"},),
        tools=(_strict_tool(),),
        tool_choice="required",
        parallel_tool_calls=False,
    )


def _native_message(arguments: str, *, name: str = "lookup") -> dict[str, object]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call_7",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def test_server_parsed_tool_call_is_normalized_after_host_validation() -> None:
    turn = _native_tool_generation_response(
        _native_message('{"q":"x"}'),
        _request(),
    )

    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_7"
    assert turn.tool_calls[0].name == "lookup"
    assert turn.tool_calls[0].arguments == {"q": "x"}


def test_server_parsed_schema_invalid_call_is_non_executable_rejection() -> None:
    turn = _native_tool_generation_response(
        _native_message('{"q":7}'),
        _request(),
    )

    assert len(turn.tool_calls) == 1
    rejection = turn.tool_calls[0]
    assert rejection.name == "__mmm_rejected_tool_call__"
    assert rejection.arguments["failure_code"] == "TOOL_SCHEMA_INVALID"
    assert rejection.arguments["original_tool"] == "lookup"


def test_server_parsed_unexposed_tool_is_non_executable_rejection() -> None:
    turn = _native_tool_generation_response(
        _native_message('{"q":"x"}', name="not_visible"),
        _request(),
    )

    assert len(turn.tool_calls) == 1
    rejection = turn.tool_calls[0]
    assert rejection.name == "__mmm_rejected_tool_call__"
    assert rejection.arguments["failure_code"] == "TOOL_NOT_VISIBLE"
    assert rejection.arguments["original_tool"] == "not_visible"


def test_qwen_markup_fallback_uses_same_host_validation_surface() -> None:
    message = {
        "content": (
            "<tool_call><function=lookup>"
            "<parameter=q>registry</parameter>"
            "</function></tool_call>"
        )
    }

    turn = _native_tool_generation_response(message, _request())

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "lookup"
    assert turn.tool_calls[0].arguments == {"q": "registry"}
