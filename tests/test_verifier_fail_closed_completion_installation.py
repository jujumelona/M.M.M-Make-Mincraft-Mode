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


def _applied_create_messages(*, status: str = "FAIL"):
    path = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    create_args = {
        "operation": "create_file",
        "path": path,
        "content": "package dev.mmm.debugfixture; public class DebugToken {}",
    }
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "mutate-1",
                    "type": "function",
                    "function": {
                        "name": "apply_source_edit",
                        "arguments": json.dumps(create_args),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "apply_source_edit",
            "tool_call_id": "mutate-1",
            "content": json.dumps(
                {
                    "ok": True,
                    "result": {
                        "schema_version": "mmm/source-patch-receipt-v1",
                        "status": "APPLIED",
                        "operations": [
                            {
                                "path": path,
                                "before_sha256": None,
                                "after_sha256": "sha256:created",
                            }
                        ],
                    },
                }
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "verify-1",
                    "type": "function",
                    "function": {
                        "name": "java_diagnostics",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "java_diagnostics",
            "tool_call_id": "verify-1",
            "content": json.dumps(
                {
                    "ok": True,
                    "result": {
                        "status": status,
                        "diagnostics": {
                            f"file:///{path}": [
                                {
                                    "severity": 1,
                                    "code": "UndefinedType",
                                    "message": "RegistryWrapper cannot be resolved to a type",
                                    "range": {
                                        "start": {"line": 4, "character": 7},
                                        "end": {"line": 4, "character": 22},
                                    },
                                }
                            ]
                        },
                    },
                }
            ),
        },
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


def test_failed_verifier_injects_diagnostics_and_existing_file_repair_context():
    turn = SimpleNamespace(
        content="",
        tool_calls=(
            SimpleNamespace(
                name="apply_source_edit",
                arguments={
                    "operation": "replace_exact",
                    "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
                    "old": "RegistryWrapper",
                    "new": "RegistryEntry",
                },
            ),
        ),
    )
    module, calls = _module(turn)
    messages = _applied_create_messages()

    assert _invoke(module, messages=messages, tools=(_tool("apply_source_edit"),)) is turn
    assert len(calls) == 1
    repair_messages = calls[0]["messages"]
    guidance = next(
        message["content"]
        for message in repair_messages
        if message.get("role") == "system"
        and str(message.get("content", "")).startswith("MMM_VERIFIER_REPAIR_CONTEXT_V1")
    )
    assert "RegistryWrapper cannot be resolved to a type" in guidance
    assert "src/main/java/dev/mmm/debugfixture/DebugToken.java" in guidance
    assert "do not call create_file/create/write" in guidance


def test_failed_verifier_rejects_recreating_a_receipt_proven_created_path():
    path = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    turn = SimpleNamespace(
        content="",
        tool_calls=(
            SimpleNamespace(
                name="apply_source_edit",
                arguments={
                    "operation": "create_file",
                    "path": path,
                    "content": "package dev.mmm.debugfixture; public class DebugToken {}",
                },
            ),
        ),
    )
    module, calls = _module(turn)

    with pytest.raises(_ConfigError, match="VERIFICATION_REPAIR_CREATE_CONFLICT"):
        _invoke(
            module,
            messages=_applied_create_messages(),
            tools=(_tool("apply_source_edit"),),
        )
    assert len(calls) == 1


def test_initial_create_is_not_blocked_without_applied_create_receipt():
    path = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    turn = SimpleNamespace(
        content="",
        tool_calls=(
            SimpleNamespace(
                name="apply_source_edit",
                arguments={
                    "operation": "create_file",
                    "path": path,
                    "content": "package dev.mmm.debugfixture; public class DebugToken {}",
                },
            ),
        ),
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
