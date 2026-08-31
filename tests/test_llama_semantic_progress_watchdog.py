from __future__ import annotations

import pytest

from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract
from minecraft_mod_ai.llama_completion_liveness_contract import (
    LlamaSemanticProgressTimeout,
    _semantic_progress_from_sse_line,
    _SemanticProgressWatchdog,
)


def _clock(*values: float):
    iterator = iter(values)
    return lambda: next(iterator)


def test_sse_ping_is_transport_liveness_not_semantic_progress() -> None:
    watchdog = _SemanticProgressWatchdog(
        120.0,
        clock=_clock(0.0, 30.0, 60.0, 121.0),
    )

    watchdog.observe(": ping - 1")
    watchdog.observe(": ping - 2")
    with pytest.raises(LlamaSemanticProgressTimeout, match="no semantic"):
        watchdog.observe(": ping - 3")


def test_prompt_progress_resets_semantic_deadline_only_when_counter_advances() -> None:
    watchdog = _SemanticProgressWatchdog(
        120.0,
        clock=_clock(0.0, 90.0, 180.0, 205.0),
    )

    watchdog.observe('data: {"prompt_progress":{"processed":1024}}')
    watchdog.observe('data: {"prompt_progress":{"processed":1024}}')
    watchdog.observe('data: {"prompt_progress":{"processed":2048}}')

    # 2048 advanced at t=205, so this sequence proves the duplicate 1024 event did
    # not become a synthetic progress reset while the real counter advance did.


def test_nonadvancing_prompt_progress_cannot_keep_request_alive() -> None:
    watchdog = _SemanticProgressWatchdog(
        120.0,
        clock=_clock(0.0, 10.0, 70.0, 131.0),
    )

    watchdog.observe('data: {"prompt_progress":{"processed":1024}}')
    watchdog.observe('data: {"prompt_progress":{"processed":1024}}')
    with pytest.raises(LlamaSemanticProgressTimeout):
        watchdog.observe('data: {"prompt_progress":{"processed":1024}}')


def test_content_reasoning_tool_fragments_and_done_are_semantic_progress() -> None:
    cases = (
        'data: {"choices":[{"delta":{"content":"x"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"r"}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"x\\""}}]}}]}',
        'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}',
        "data: [DONE]",
    )
    prompt = None
    for line in cases:
        progressed, prompt = _semantic_progress_from_sse_line(
            line,
            last_prompt_processed=prompt,
        )
        assert progressed is True


def test_ping_usage_and_role_only_events_are_not_semantic_progress() -> None:
    cases = (
        ": ping - 1",
        'data: {"usage":{"prompt_tokens":1,"completion_tokens":0}}',
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[]}',
    )
    for line in cases:
        progressed, _ = _semantic_progress_from_sse_line(
            line,
            last_prompt_processed=None,
        )
        assert progressed is False


def test_runtime_uses_semantic_sse_owner_and_disables_slot_poll_reporter() -> None:
    assert getattr(
        stream_contract._StreamingCompletionClient.post,
        "_mmm_progress_aware_completion_transport_v1",
        False,
    )
    assert getattr(
        stream_contract._StreamingCompletionClient.stream,
        "_mmm_progress_aware_completion_stream_v1",
        False,
    )
    assert getattr(
        stream_contract._StreamingCompletionClient.__init__,
        "_mmm_semantic_progress_client_v1",
        False,
    )
    assert not hasattr(stream_contract, "_native_tool_liveness_reporter")
    assert not hasattr(stream_contract, "_probe_native_tool_progress")
    assert not hasattr(stream_contract, "_slot_progress_from_payload")
