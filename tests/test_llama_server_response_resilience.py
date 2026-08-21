from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.llama_server_response_resilience import (
    _transient_completion_failure,
    install,
)


def test_transient_http_failures_are_classified_without_retrying_request_errors() -> None:
    assert _transient_completion_failure(
        RuntimeError("llama server returned HTTP 503: model temporarily unavailable")
    )
    assert _transient_completion_failure(
        RuntimeError("llama server returned HTTP 500: no available slot")
    )
    assert _transient_completion_failure(
        RuntimeError("llama server returned HTTP 500: internal server error")
    )
    assert not _transient_completion_failure(
        RuntimeError("llama server returned HTTP 500: context length exceeds token limit")
    )
    assert not _transient_completion_failure(
        RuntimeError("llama server returned HTTP 400: invalid request")
    )


def test_transient_completion_retries_exactly_once(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_TRANSIENT_RETRY_DELAY_SECONDS", "0")
    calls: list[dict] = []

    def completion(server_url, payload):
        calls.append(dict(payload))
        if len(calls) == 1:
            raise RuntimeError("llama server returned HTTP 502: upstream reset")
        return {"role": "assistant", "content": "ok", "server": server_url}

    module = SimpleNamespace(_completion_message=completion)
    install(module)

    result = module._completion_message("http://127.0.0.1:8910/v1", {"max_tokens": 8})
    assert result["content"] == "ok"
    assert calls == [{"max_tokens": 8}, {"max_tokens": 8}]


def test_permanent_completion_failure_is_not_retried(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_TRANSIENT_RETRY_DELAY_SECONDS", "0")
    calls = 0

    def completion(server_url, payload):
        nonlocal calls
        del server_url, payload
        calls += 1
        raise RuntimeError("llama server returned HTTP 500: invalid request: context length")

    module = SimpleNamespace(_completion_message=completion)
    install(module)

    with pytest.raises(RuntimeError, match="invalid request"):
        module._completion_message("http://127.0.0.1:8910/v1", {})
    assert calls == 1
