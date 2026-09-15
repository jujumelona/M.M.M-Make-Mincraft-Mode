from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    _native_tool_generation_response,
)
from minecraft_mod_ai.model_adapters.qwen_tool_parser import ToolCallValidationError


_APPLY_SOURCE_EDIT = {
    "type": "function",
    "function": {
        "name": "apply_source_edit",
        "description": "Apply one source edit.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


def _request(
    *,
    tool_choice: str | dict = "required",
    parallel_tool_calls: bool = True,
) -> GenerationRequest:
    return GenerationRequest(
        tools=(_APPLY_SOURCE_EDIT,),
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )


def _named_apply_source_edit_choice() -> dict:
    return {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }


def _valid_call(call_id: str, path: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "arguments": '{"path": ' + repr(path).replace("'", '"') + "}",
        },
    }


def test_named_tool_choice_does_not_turn_prose_only_completion_into_adapter_failure() -> None:
    response = _native_tool_generation_response(
        {"content": "I cannot make a safe edit from the current evidence."},
        _request(tool_choice=_named_apply_source_edit_choice()),
    )

    assert response.content == "I cannot make a safe edit from the current evidence."
    assert response.tool_calls == ()


def test_required_tool_choice_allows_empty_native_completion_to_reach_orchestrator() -> None:
    response = _native_tool_generation_response(
        {"content": "", "tool_calls": []},
        _request(tool_choice="required"),
    )

    assert response.content == ""
    assert response.reasoning_content == ""
    assert response.tool_calls == ()


def test_malformed_actual_native_tool_call_still_fails_structural_validation() -> None:
    with pytest.raises(ToolCallValidationError, match="invalid JSON arguments"):
        _native_tool_generation_response(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "broken",
                        "type": "function",
                        "function": {
                            "name": "apply_source_edit",
                            "arguments": "{",
                        },
                    }
                ],
            },
            _request(),
        )


def test_parallel_native_calls_remain_rejected_when_parallel_calls_are_disabled() -> None:
    with pytest.raises(ToolCallValidationError, match="parallel tool calls"):
        _native_tool_generation_response(
            {
                "content": "",
                "tool_calls": [
                    _valid_call("call_a", "a.py"),
                    _valid_call("call_b", "b.py"),
                ],
            },
            _request(parallel_tool_calls=False),
        )
