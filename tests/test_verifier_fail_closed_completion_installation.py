from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.verifier_fail_closed_completion_installation import (
    install,
    latest_verifier_outcome,
)


class _ConfigError(RuntimeError):
    pass


def _tool(name: str):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def _module(turn):
    calls = []

    def generate(_router, **kwargs):
        calls.append(kwargs)
        return turn

    module = SimpleNamespace(
        _VERIFY_TOOLS=frozenset({"java_diagnostics", "run_gradle_build"}),
        _MUTATION_ACT_TOOLS=frozenset({"apply_source_edit"}),
        _tool_name=lambda schema: schema.get("function", {}).get("name", ""),
        _verification_outcome=lambda _name, payload: str(
            payload.get("result", {}).get("status", "PASS")
        ).upper(),
        _generate_turn_with_context_recovery=generate,
        ModelConfigurationError=_ConfigError,
    )
    install(module)
    return module, calls


def _messages(status: str):
    return [
        {
            "role": "tool",
            "name": "java_diagnostics",
            "tool_call_id": "verify-1",
            "content": json.dumps({"ok": True, "result": {"status": status}}),
        }
    ]


def _invoke(module, *, messages, tools):
    request = SimpleNamespace(tools=tools)
    return module._generate_turn_with_context_recovery(
        object(),
        config=object(),
        adapter=object(),
        request=request,
        messages=messages,
        media_paths=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )


def test_latest_verifier_outcome_uses_newest_verifier_observation():
    messages = [*_messages("FAIL"), *_messages("PASS")]
    module, _calls = _module(SimpleNamespace(content="done", tool_calls=()))
    assert latest_verifier_outcome(module, messages) == "PASS"


def test_failed_verifier_cannot_enter_no_tool_finalization():
    module, calls = _module(SimpleNamespace(content="done", tool_calls=()))
    with pytest.raises(_ConfigError, match="VERIFICATION_FAILED_FIXED_POINT"):
        _invoke(module, messages=_messages("FAIL"), tools=())
    assert calls == []


def test_failed_verifier_rejects_prose_in_mutation_phase():
    module, calls = _module(SimpleNamespace(content="implementation complete", tool_calls=()))
    with pytest.raises(_ConfigError, match="VERIFICATION_FAILED_PROSE_REJECTED"):
        _invoke(
            module,
            messages=_messages("FAIL"),
            tools=(_tool("apply_source_edit"),),
        )
    assert len(calls) == 1


def test_failed_verifier_allows_actual_corrective_tool_call():
    turn = SimpleNamespace(
        content="",
        tool_calls=(SimpleNamespace(name="apply_source_edit"),),
    )
    module, calls = _module(turn)
    assert (
        _invoke(
            module,
            messages=_messages("FAIL"),
            tools=(_tool("apply_source_edit"),),
        )
        is turn
    )
    assert len(calls) == 1


def test_verifier_pass_allows_final_summary():
    turn = SimpleNamespace(content="done", tool_calls=())
    module, calls = _module(turn)
    assert _invoke(module, messages=_messages("PASS"), tools=()) is turn
    assert len(calls) == 1
