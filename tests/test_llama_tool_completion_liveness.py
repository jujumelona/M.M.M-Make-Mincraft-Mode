from __future__ import annotations

from typing import Any

import httpx

from minecraft_mod_ai import llama_stream_efficiency_contract as contract


class _ImmediateRawClient:
    def __init__(self) -> None:
        self.timeout: httpx.Timeout | None = None
        self.response = object()

    def post(self, _url: str, **kwargs: Any) -> object:
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, httpx.Timeout)
        self.timeout = timeout
        return self.response


def _timeout(read: float) -> httpx.Timeout:
    return httpx.Timeout(connect=30.0, read=read, write=30.0, pool=30.0)


def test_native_tool_turn_enforces_bounded_idle_read_timeout(monkeypatch) -> None:
    raw = _ImmediateRawClient()
    client = contract._StreamingCompletionClient(raw)
    monkeypatch.delenv("MMM_LLAMA_TOOL_IDLE_TIMEOUT_SECONDS", raising=False)

    response = client.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={"tools": [{"type": "function"}], "max_tokens": 4096},
        timeout=_timeout(600.0),
    )

    assert response is raw.response
    assert raw.timeout is not None
    assert raw.timeout.read == 120.0


def test_explicit_tool_completion_timeout_is_honored(monkeypatch) -> None:
    raw = _ImmediateRawClient()
    client = contract._StreamingCompletionClient(raw)
    monkeypatch.setenv("MMM_LLAMA_TOOL_IDLE_TIMEOUT_SECONDS", "75")

    client.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={"tools": [{"type": "function"}], "max_tokens": 4096},
        timeout=_timeout(600.0),
    )

    assert raw.timeout is not None
    assert raw.timeout.read == 75.0


def test_slot_poll_liveness_api_stays_removed() -> None:
    assert not hasattr(contract, "_native_tool_liveness_reporter")
    assert not hasattr(contract, "_probe_native_tool_progress")
    assert not hasattr(contract, "_slot_progress_from_payload")
