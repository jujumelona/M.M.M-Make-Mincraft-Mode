from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    _qwen_tool_generation_response,
    _reject_partial_server_tool_calls,
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


def test_server_parsed_tool_call_is_normalized_only_after_host_validation() -> None:
    turn = _qwen_tool_generation_response(
        _native_message('{"q":"x"}'),
        _request(),
    )

    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_7"
    assert turn.tool_calls[0].name == "lookup"
    assert turn.tool_calls[0].arguments == {"q": "x"}


def test_server_parsed_tool_call_rejects_schema_invalid_arguments() -> None:
    with pytest.raises(RuntimeError, match="schema-invalid arguments"):
        _qwen_tool_generation_response(
            _native_message('{"q":7}'),
            _request(),
        )


def test_server_parsed_tool_call_preserves_unexposed_tool_for_host_phase_validation() -> None:
    turn = _qwen_tool_generation_response(
        _native_message('{"q":"x"}', name="not_visible"),
        _request(),
    )
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "not_visible"
    assert turn.tool_calls[0].arguments == {"q": "x"}


def test_incomplete_server_parsed_tool_call_remains_non_executable() -> None:
    with pytest.raises(RuntimeError, match="partial tool actions are never executable"):
        _reject_partial_server_tool_calls(_native_message('{"q":"x"}'))
