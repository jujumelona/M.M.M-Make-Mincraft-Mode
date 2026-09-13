from __future__ import annotations

import time

import httpx
import pytest

from minecraft_mod_ai.model_adapters import llama_turn_retry as retry
from minecraft_mod_ai.model_concurrency import (
    ModelExecutionDeadlineExceeded,
    bind_model_execution_deadline,
)


def test_model_execution_deadline_is_never_retriable() -> None:
    assert retry._is_retriable_turn_failure(
        ModelExecutionDeadlineExceeded("expired")
    ) is False


def test_transactional_turn_retry_stops_immediately_on_model_deadline(monkeypatch) -> None:
    calls = 0

    class Adapter:
        def generate_turn(self, request):
            nonlocal calls
            calls += 1
            raise ModelExecutionDeadlineExceeded("expired")

    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_ATTEMPTS", "5")
    retry.install_llama_turn_retry(Adapter)

    with pytest.raises(ModelExecutionDeadlineExceeded):
        Adapter().generate_turn(object())

    assert calls == 1


def test_retry_backoff_cannot_outlive_model_deadline(monkeypatch) -> None:
    calls = 0
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")

    class Adapter:
        def generate_turn(self, turn_request):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("transient timeout", request=request)

    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("MMM_LLAMA_TURN_RETRY_BACKOFF_SECONDS", "1")
    retry.install_llama_turn_retry(Adapter)

    with bind_model_execution_deadline(time.monotonic() + 0.05):
        with pytest.raises(ModelExecutionDeadlineExceeded):
            Adapter().generate_turn(object())

    assert calls == 1
