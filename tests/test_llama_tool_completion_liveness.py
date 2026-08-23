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


class _JsonResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _ProbeClient:
    def __init__(self, slots: Any) -> None:
        self.slots = slots
        self.urls: list[str] = []

    def get(self, url: str, *, timeout: float) -> _JsonResponse:
        assert timeout > 0
        self.urls.append(url)
        if url.endswith("/slots"):
            return _JsonResponse(200, self.slots)
        return _JsonResponse(200, {"status": "ok"})


def _timeout(read: float) -> httpx.Timeout:
    return httpx.Timeout(connect=30.0, read=read, write=30.0, pool=30.0)


def test_unbounded_native_tool_turn_disables_implicit_read_deadline(
    monkeypatch,
    capsys,
) -> None:
    raw = _ImmediateRawClient()
    client = contract._StreamingCompletionClient(raw)
    monkeypatch.delenv("MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS", raising=False)

    response = client.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={"tools": [{"type": "function"}], "max_tokens": -1},
        timeout=_timeout(600.0),
    )

    assert response is raw.response
    assert raw.timeout is not None
    assert raw.timeout.read is None
    output = capsys.readouterr().out
    assert "effective_read_timeout=none" in output
    assert "reason=max_tokens=-1" in output


def test_explicit_tool_completion_timeout_is_not_overridden(monkeypatch) -> None:
    raw = _ImmediateRawClient()
    client = contract._StreamingCompletionClient(raw)
    monkeypatch.setenv("MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS", "75")

    client.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={"tools": [{"type": "function"}], "max_tokens": -1},
        timeout=_timeout(75.0),
    )

    assert raw.timeout is not None
    assert raw.timeout.read == 75.0


def test_bounded_tool_completion_keeps_read_deadline(monkeypatch) -> None:
    raw = _ImmediateRawClient()
    client = contract._StreamingCompletionClient(raw)
    monkeypatch.delenv("MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS", raising=False)

    client.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={"tools": [{"type": "function"}], "max_tokens": 512},
        timeout=_timeout(600.0),
    )

    assert raw.timeout is not None
    assert raw.timeout.read == 600.0


def test_slot_progress_reports_only_counters_not_generated_text() -> None:
    snapshot = contract._slot_progress_from_payload(
        [
            {
                "id": 0,
                "is_processing": True,
                "n_prompt_tokens_processed": 4096,
                "n_decoded": 73,
                "generated": "secret model output must not be logged",
            },
            {"id": 1, "is_processing": False, "n_decoded": 999},
        ]
    )

    assert snapshot == {
        "processing_slots": 1,
        "decoded": 73,
        "prompt_processed": 4096,
    }
    assert "generated" not in snapshot


def test_live_probe_uses_origin_slots_endpoint() -> None:
    client = _ProbeClient(
        [
            {
                "id": 0,
                "is_processing": True,
                "n_prompt_tokens_processed": 4550,
                "n_decoded": 128,
            }
        ]
    )

    snapshot = contract._probe_native_tool_progress(
        client,
        "http://127.0.0.1:8080/v1/chat/completions",
    )

    assert snapshot == {
        "state": "slots",
        "processing_slots": 1,
        "decoded": 128,
        "prompt_processed": 4550,
    }
    assert client.urls == ["http://127.0.0.1:8080/slots"]
