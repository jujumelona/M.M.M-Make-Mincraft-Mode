"""Deterministic MiMo-native tool-call transport replay fixtures.

These are host-boundary simulations, not claims about live model inference.
MiMo never uses the legacy Qwen3-Coder XML fallback.
"""
from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _native_tool_generation_response


def _request(*, required=True, parallel=False):
    return GenerationRequest(
        messages=({"role": "user", "content": "lookup example"},),
        tools=({
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                    "additionalProperties": False,
                },
            },
        },),
        tool_choice="required" if required else "auto",
        parallel_tool_calls=parallel,
    )


def _call(arguments='{"q":"ok"}', name="lookup", ident="call_1"):
    return {
        "type": "function",
        "id": ident,
        "function": {"name": name, "arguments": arguments},
    }


def _parse(message, *, required=True, parallel=False):
    return _native_tool_generation_response(
        message, _request(required=required, parallel=parallel),
        runtime_contract="mimo",
    )


def test_mimo_structured_native_tool_call_admitted():
    turn = _parse({"content": "", "tool_calls": [_call()]})
    assert [(c.name, c.arguments) for c in turn.tool_calls] == [
        ("lookup", {"q": "ok"})
    ]


def test_mimo_reasoning_and_tool_call_are_separate():
    turn = _parse({
        "content": "",
        "reasoning_content": "reasoned",
        "tool_calls": [_call()],
    })
    assert turn.reasoning_content == "reasoned"
    assert turn.content == ""
    assert turn.tool_calls[0].name == "lookup"


@pytest.mark.parametrize("args", ['{"q":7}', '{"x":"unexpected"}', '{"q":"ok","extra":1}', '{"q":'])
def test_mimo_invalid_argument_rejected_without_execution(args):
    turn = _parse({"content": "", "tool_calls": [_call(args)]})
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "__mmm_rejected_tool_call__"


def test_mimo_unregistered_tool_rejected():
    turn = _parse({"content": "", "tool_calls": [_call(name="unknown")]})
    assert turn.tool_calls[0].name == "__mmm_rejected_tool_call__"


def test_mimo_missing_required_tool_rejected():
    turn = _parse({"content": "no action"})
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "__mmm_rejected_tool_call__"


def test_mimo_no_tools_allowed_for_optional_action():
    turn = _parse({"content": "finished"}, required=False)
    assert turn.tool_calls == ()
    assert turn.content == "finished"


def test_mimo_multiple_calls_fail_closed_when_parallel_disabled():
    turn = _parse({"content": "", "tool_calls": [
        _call(ident="one"), _call(ident="two")
    ]})
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "__mmm_rejected_tool_call__"


@pytest.mark.parametrize("raw", [
    "<tool_call><function=lookup><parameter=q>ok</parameter></function></tool_call>",
    '<tool_call><function=lookup>{"q":"ok"}</function></tool_call>',
    "<function=lookup><parameter=q>ok</parameter></function>",
])
def test_mimo_never_executes_unparsed_legacy_xml(raw):
    with pytest.raises(RuntimeError, match="MIMO_NATIVE_TOOL_CALLS_REQUIRED"):
        _parse({"content": raw})


def test_mimo_native_tool_call_roundtrip_after_prior_tool_result():
    request = _request()
    prior_messages = tuple(request.messages) + (
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call(ident="call_one")],
        },
        {
            "role": "tool",
            "tool_call_id": "call_one",
            "name": "lookup",
            "content": '{"result":"ok"}',
        },
    )
    from dataclasses import replace
    continued = replace(request, messages=prior_messages)
    turn = _native_tool_generation_response(
        {"content": "", "tool_calls": [_call('{"q":"again"}', ident="call_two")]},
        continued,
        runtime_contract="mimo",
    )
    assert turn.tool_calls[0].arguments == {"q": "again"}
