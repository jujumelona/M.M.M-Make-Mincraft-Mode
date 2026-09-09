from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import mutation_authority_final_guard as guard


def _fake_loop(calls: list[dict[str, object]]) -> SimpleNamespace:
    class FakeContext:
        def merge(self, other):
            return other if other is not None else self

    class FakeModelConfigurationError(RuntimeError):
        pass

    def generate(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "generated"

    return SimpleNamespace(
        TargetMutationContext=FakeContext,
        ModelConfigurationError=FakeModelConfigurationError,
        _generate_turn_with_context_recovery=generate,
        is_mutation_ready=lambda messages, state: True,
        _canonical_mutation_path=lambda value: str(value or "").strip(),
        _planir_owned_anchor_sets=lambda payload: ((), ()),
        _extract_mutation_context_from_payload=lambda payload: None,
        emit_root_cause=lambda *args, **kwargs: None,
    )


@pytest.mark.parametrize(
    "failure_code",
    [
        "MUTATION_TARGET_DRIFT",
        "MUTATION_TARGET_UNBOUND",
        "MUTATION_TARGET_CREATION_CONFLICT",
        "PATH_OUTSIDE_WRITABLE_SET",
        "MUTATION_AUTHORITY_CONFLICT",
        "WRITE_SCOPE_REJECTED",
        "PHASE_PROTOCOL_VIOLATION",
    ],
)
def test_post_argument_semantic_failure_never_regenerates_arguments(failure_code: str) -> None:
    calls: list[dict[str, object]] = []
    loop = _fake_loop(calls)
    guard.install(loop)

    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": json.dumps(
                {
                    "ok": False,
                    "failure_code": failure_code,
                    "error": "host semantic rejection",
                }
            ),
        },
    ]

    with pytest.raises(
        loop.ModelConfigurationError,
        match=rf"POST_ARGUMENT_SEMANTIC_FAILURE: {failure_code}",
    ):
        loop._generate_turn_with_context_recovery(
            object(),
            config=None,
            adapter=None,
            request=None,
            messages=messages,
            media_paths=(),
            tool_choice=None,
            parallel_tool_calls=False,
        )

    assert calls == []


def test_argument_generation_failures_still_reach_native_generation_recovery() -> None:
    calls: list[dict[str, object]] = []
    loop = _fake_loop(calls)
    guard.install(loop)

    result = loop._generate_turn_with_context_recovery(
        object(),
        config=None,
        adapter=None,
        request=None,
        messages=[{"role": "assistant", "content": "incomplete tool arguments"}],
        media_paths=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )

    assert result == "generated"
    assert len(calls) == 1


def test_argument_error_code_is_not_misclassified_as_semantic_execution_failure() -> None:
    calls: list[dict[str, object]] = []
    loop = _fake_loop(calls)
    guard.install(loop)

    result = loop._generate_turn_with_context_recovery(
        object(),
        config=None,
        adapter=None,
        request=None,
        messages=[
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "ok": False,
                        "failure_code": "ARGUMENT_JSON_INVALID",
                        "error": "partial JSON",
                    }
                ),
            }
        ],
        media_paths=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )

    assert result == "generated"
    assert len(calls) == 1


def test_post_argument_phase_violation_allows_host_directed_turn() -> None:
    calls: list[dict[str, object]] = []
    loop = _fake_loop(calls)
    guard.install(loop)

    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": json.dumps(
                {
                    "ok": False,
                    "tool": "apply_source_edit",
                    "failure_code": "PHASE_PROTOCOL_VIOLATION",
                    "error": "Agent attempted tool 'apply_source_edit' outside its allowed phase 'OBSERVE'",
                }
            ),
        },
    ]

    result = loop._generate_turn_with_context_recovery(
        object(),
        config=None,
        adapter=None,
        request=None,
        messages=messages,
        media_paths=(),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
        parallel_tool_calls=False,
    )

    assert result == "generated"
    assert len(calls) == 1


def test_post_argument_semantic_failure_rejects_even_when_host_directed() -> None:
    calls: list[dict[str, object]] = []
    loop = _fake_loop(calls)
    guard.install(loop)

    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": json.dumps(
                {
                    "ok": False,
                    "tool": "apply_source_edit",
                    "failure_code": "MUTATION_TARGET_DRIFT",
                    "error": "host semantic rejection",
                }
            ),
        },
    ]

    with pytest.raises(
        loop.ModelConfigurationError,
        match=r"POST_ARGUMENT_SEMANTIC_FAILURE: MUTATION_TARGET_DRIFT",
    ):
        loop._generate_turn_with_context_recovery(
            object(),
            config=None,
            adapter=None,
            request=None,
            messages=messages,
            media_paths=(),
            tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
            parallel_tool_calls=False,
        )

    assert calls == []
