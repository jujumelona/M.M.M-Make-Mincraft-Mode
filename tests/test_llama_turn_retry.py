from __future__ import annotations

import httpx
import pytest

from minecraft_mod_ai.model_adapters.base import ModelBackendError
from minecraft_mod_ai.model_adapters.llama_turn_retry import (
    _is_retriable_turn_failure,
    install_llama_turn_retry,
)


class _RetryAdapter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_turn(self, request):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _backend_error(cause: BaseException) -> ModelBackendError:
    return ModelBackendError(role="planner", model_id="local", cause=cause)


def _read_error() -> httpx.ReadError:
    return httpx.ReadError(
        "connection reset during SSE read",
        request=httpx.Request("POST", "http://localhost/v1/chat/completions"),
    )


def test_transient_stream_failure_retries_same_turn_once(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_BACKOFF_SECONDS", "0")
    adapter_type = type("RetryOnceAdapter", (_RetryAdapter,), {})
    install_llama_turn_retry(adapter_type)
    complete = {"content": "complete", "tool_calls": ()}
    adapter = adapter_type([_backend_error(_read_error()), complete])

    assert adapter.generate_turn(object()) is complete
    assert adapter.calls == 2


def test_incomplete_sse_partial_result_is_discarded_before_retry(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_BACKOFF_SECONDS", "0")
    adapter_type = type("PartialAdapter", (_RetryAdapter,), {})
    install_llama_turn_retry(adapter_type)
    incomplete = _backend_error(
        RuntimeError("llama server stream ended before the [DONE] marker")
    )
    complete = {
        "content": "final",
        "tool_calls": ({"name": "safe_complete_call"},),
    }
    adapter = adapter_type([incomplete, complete])

    result = adapter.generate_turn(object())

    assert result == complete
    assert adapter.calls == 2
    assert result["tool_calls"] == ({"name": "safe_complete_call"},)


def test_partial_tool_call_failure_never_escapes_as_executable_result(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_BACKOFF_SECONDS", "0")
    adapter_type = type("ToolFragmentAdapter", (_RetryAdapter,), {})
    install_llama_turn_retry(adapter_type)
    adapter = adapter_type(
        [
            _backend_error(
                RuntimeError("llama server stream ended before the [DONE] marker")
            ),
            {"content": "done", "tool_calls": ()},
        ]
    )

    result = adapter.generate_turn(object())

    assert result["tool_calls"] == ()
    assert adapter.calls == 2


def test_permanent_failure_is_not_retried(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_BACKOFF_SECONDS", "0")
    adapter_type = type("PermanentAdapter", (_RetryAdapter,), {})
    install_llama_turn_retry(adapter_type)
    error = _backend_error(ValueError("tool schema is invalid"))
    adapter = adapter_type([error])

    with pytest.raises(ModelBackendError) as raised:
        adapter.generate_turn(object())

    assert raised.value is error
    assert adapter.calls == 1


def test_retries_are_bounded(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_BACKOFF_SECONDS", "0")
    adapter_type = type("BoundedAdapter", (_RetryAdapter,), {})
    install_llama_turn_retry(adapter_type)
    errors = [_backend_error(_read_error()) for _ in range(3)]
    adapter = adapter_type(errors)

    with pytest.raises(ModelBackendError):
        adapter.generate_turn(object())

    assert adapter.calls == 3


def test_http_5xx_is_retriable_but_4xx_is_not():
    assert _is_retriable_turn_failure(
        _backend_error(RuntimeError("llama server returned HTTP 503: unavailable"))
    )
    assert not _is_retriable_turn_failure(
        _backend_error(RuntimeError("llama server returned HTTP 400: bad request"))
    )


def test_install_is_idempotent(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_BACKOFF_SECONDS", "0")
    adapter_type = type("IdempotentAdapter", (_RetryAdapter,), {})
    install_llama_turn_retry(adapter_type)
    first = adapter_type.generate_turn
    install_llama_turn_retry(adapter_type)

    assert adapter_type.generate_turn is first
