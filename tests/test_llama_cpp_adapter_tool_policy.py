from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    _native_tool_generation_response,
)

_APPLY_SOURCE_EDIT = {
    "type": "function",
    "function": {
        "name": "apply_source_edit",
        "description": "Apply one source edit.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
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
    return {"type": "function", "function": {"name": "apply_source_edit"}}


def _valid_call(call_id: str, path: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "arguments": '{"path": ' + repr(path).replace("'", '"') + "}",
        },
    }


def _assert_rejection(response, failure_code: str) -> None:
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "__mmm_rejected_tool_call__"
    assert call.arguments["failure_code"] == failure_code


def test_named_tool_choice_returns_missing_required_rejection_for_prose_only_completion() -> None:
    response = _native_tool_generation_response(
        {"content": "I cannot make a safe edit from the current evidence."},
        _request(tool_choice=_named_apply_source_edit_choice()),
    )
    assert response.content == "I cannot make a safe edit from the current evidence."
    _assert_rejection(response, "REQUIRED_TOOL_MISSING")


def test_required_tool_choice_returns_missing_required_rejection_for_empty_completion() -> None:
    response = _native_tool_generation_response(
        {"content": "", "tool_calls": []},
        _request(tool_choice="required"),
    )
    assert response.content == ""
    assert response.reasoning_content == ""
    _assert_rejection(response, "REQUIRED_TOOL_MISSING")


def test_malformed_native_tool_call_is_non_executed_rejection() -> None:
    response = _native_tool_generation_response(
        {
            "content": "",
            "tool_calls": [{
                "id": "broken",
                "type": "function",
                "function": {"name": "apply_source_edit", "arguments": "{"},
            }],
        },
        _request(),
    )
    _assert_rejection(response, "TOOL_CALL_MALFORMED")


def test_parallel_native_calls_are_non_executed_rejection_when_disabled() -> None:
    response = _native_tool_generation_response(
        {
            "content": "",
            "tool_calls": [
                _valid_call("call_a", "a.py"),
                _valid_call("call_b", "b.py"),
            ],
        },
        _request(parallel_tool_calls=False),
    )
    _assert_rejection(response, "PARALLEL_TOOL_CALLS_DISABLED")
