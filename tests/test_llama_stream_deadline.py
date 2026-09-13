from __future__ import annotations

import time

import httpx
import pytest

from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract
from minecraft_mod_ai.model_concurrency import (
    ModelExecutionDeadlineExceeded,
    bind_model_execution_deadline,
)


def test_bounded_timeout_clamps_every_http_phase_to_model_deadline() -> None:
    with bind_model_execution_deadline(time.monotonic() + 0.2):
        timeout = stream_contract._bounded_timeout(
            httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
            read_seconds=120.0,
        )

    for value in (timeout.connect, timeout.read, timeout.write, timeout.pool):
        assert value is not None
        assert 0.0 < float(value) <= 0.2


def test_bounded_timeout_fails_closed_when_model_deadline_already_expired() -> None:
    with bind_model_execution_deadline(time.monotonic() - 1.0):
        with pytest.raises(ModelExecutionDeadlineExceeded):
            stream_contract._bounded_timeout(None, read_seconds=120.0)


def test_non_completion_post_also_receives_deadline_bounded_timeout() -> None:
    captured: dict[str, object] = {}

    class RawClient:
        def post(self, url: str, **kwargs):
            captured["url"] = url
            captured["timeout"] = kwargs.get("timeout")
            return object()

    client = stream_contract._StreamingCompletionClient(RawClient())
    with bind_model_execution_deadline(time.monotonic() + 0.2):
        client.post(
            "http://localhost:8080/apply-template",
            json={"messages": []},
            timeout=httpx.Timeout(120.0),
        )

    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    for value in (timeout.connect, timeout.read, timeout.write, timeout.pool):
        assert value is not None
        assert 0.0 < float(value) <= 0.2
