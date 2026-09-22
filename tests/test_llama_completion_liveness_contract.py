from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_completion_liveness_contract as contract
from minecraft_mod_ai.llama_sse_protocol import LlamaSseServerError
from minecraft_mod_ai.model_adapters import llama_cpp_adapter


def test_progress_payload_requests_prompt_events_and_bounded_ping() -> None:
    stream_module = SimpleNamespace(
        _tool_idle_timeout_seconds=lambda: 120.0,
        _stream_idle_timeout_seconds=lambda: 120.0,
    )
    original = {"model": "local", "messages": [], "tools": [{"type": "function"}]}

    result = contract._progress_aware_payload(stream_module, original)

    assert result is not original
    assert "return_progress" not in original
    assert result["return_progress"] is True
    assert result["sse_ping_interval"] == 30


def test_semantic_progress_ignores_transport_ping_and_tracks_prompt_progress() -> None:
    progressed, processed = contract._semantic_progress_from_sse_line(
        ": ping", last_prompt_processed=None
    )
    assert progressed is False
    assert processed is None

    progressed, processed = contract._semantic_progress_from_sse_line(
        'data: {"prompt_progress":{"processed":64}}',
        last_prompt_processed=None,
    )
    assert progressed is True
    assert processed == 64


def test_progress_response_raises_server_error_before_watchdog() -> None:
    response = SimpleNamespace(
        iter_lines=lambda: iter(
            ['data: {"error":{"code":400,"message":"context overflow"}}']
        )
    )
    wrapped = contract._ProgressCheckedResponse(
        response,
        0.001,
        request_id="test-server-error",
        started_at=0.0,
    )

    with pytest.raises(LlamaSseServerError, match="context overflow"):
        list(wrapped.iter_lines())


def test_install_wraps_nonstream_chat_completion_without_changing_timeout() -> None:
    calls: list[tuple[str, dict]] = []

    class FakeClient:
        def post(self, url: str, **kwargs):
            calls.append((url, kwargs))
            return "ok"

    stream_module = SimpleNamespace(
        _StreamingCompletionClient=FakeClient,
        _tool_idle_timeout_seconds=lambda: 12.0,
        _stream_idle_timeout_seconds=lambda: 120.0,
    )

    contract.install(stream_module)
    client = FakeClient()
    timeout = object()
    result = client.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={"messages": [], "tools": [{"type": "function"}]},
        timeout=timeout,
    )

    assert result == "ok"
    assert calls[0][1]["timeout"] is timeout
    assert calls[0][1]["json"]["return_progress"] is True
    assert calls[0][1]["json"]["sse_ping_interval"] == 4


def test_liveness_install_has_no_reporter_or_slot_polling_dependency() -> None:
    class FakeClient:
        def __init__(self, _client=None):
            self._client = _client

        def post(self, _url: str, **_kwargs):
            return "ok"

        def stream(self, method: str, url: str, **kwargs):
            return method, url, kwargs

    stream_module = SimpleNamespace(
        _StreamingCompletionClient=FakeClient,
        _CLIENTS={},
        _tool_idle_timeout_seconds=lambda: 12.0,
        _stream_idle_timeout_seconds=lambda: 120.0,
    )

    contract.install(stream_module)

    assert not hasattr(stream_module, "_native_tool_liveness_reporter")
    assert not hasattr(stream_module, "_probe_native_tool_progress")


def test_runtime_completion_transport_has_one_progress_aware_owner() -> None:
    assert getattr(
        llama_cpp_adapter._post_completion,
        "_mmm_single_progress_aware_completion_owner_v1",
        False,
    )


def test_semantic_progress_refreshes_execution_deadline(monkeypatch) -> None:
    refreshed: list[float] = []
    monkeypatch.setattr(
        contract,
        "refresh_model_execution_deadline",
        lambda seconds: refreshed.append(float(seconds)),
    )
    response = SimpleNamespace(
        iter_lines=lambda: iter(
            [
                'data: {"choices":[{"delta":{"content":"x"}}]}',
                ": ping",
                "data: [DONE]",
            ]
        )
    )
    wrapped = contract._ProgressCheckedResponse(
        response,
        120.0,
        request_id="test-refresh-deadline",
        started_at=0.0,
    )

    assert list(wrapped.iter_lines())[-1] == "data: [DONE]"
    assert refreshed == [120.0, 120.0]



def test_transient_protocol_disconnect_replays_once_before_model_failure() -> None:
    calls: list[dict] = []

    class Client:
        def post(self, _url: str, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise llama_cpp_adapter.httpx.RemoteProtocolError(
                    "peer closed connection without complete body"
                )
            return "recovered"

    result = contract._post_completion_with_transport_replay(
        Client(),
        "http://127.0.0.1:8080/v1/chat/completions",
        payload={"messages": [], "tools": [{"type": "function"}]},
        timeout=llama_cpp_adapter.httpx.Timeout(120.0),
        httpx_module=llama_cpp_adapter.httpx,
        request_id="llama-first",
    )

    assert result == "recovered"
    assert len(calls) == 2
    assert calls[0]["headers"]["X-MMM-Request-Id"] == "llama-first"
    assert calls[0]["headers"]["X-MMM-Request-Id"] != calls[1]["headers"]["X-MMM-Request-Id"]


def test_semantic_progress_timeout_replays_once_before_model_failure() -> None:
    calls: list[dict] = []

    class Client:
        def post(self, _url: str, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise contract.LlamaSemanticProgressTimeout("semantic stall")
            return "recovered"

    result = contract._post_completion_with_transport_replay(
        Client(),
        "http://127.0.0.1:8080/v1/chat/completions",
        payload={"messages": [], "tools": [{"type": "function"}]},
        timeout=llama_cpp_adapter.httpx.Timeout(120.0),
        httpx_module=llama_cpp_adapter.httpx,
        request_id="llama-first",
    )

    assert result == "recovered"
    assert len(calls) == 2
    assert calls[0]["headers"]["X-MMM-Request-Id"] == "llama-first"
    assert calls[0]["headers"]["X-MMM-Request-Id"] != calls[1]["headers"]["X-MMM-Request-Id"]


def test_second_semantic_progress_timeout_is_not_retried_forever() -> None:
    calls = 0

    class Client:
        def post(self, _url: str, **_kwargs):
            nonlocal calls
            calls += 1
            raise contract.LlamaSemanticProgressTimeout("semantic stall")

    with pytest.raises(contract.LlamaSemanticProgressTimeout, match="semantic stall"):
        contract._post_completion_with_transport_replay(
            Client(),
            "http://127.0.0.1:8080/v1/chat/completions",
            payload={"messages": []},
            timeout=llama_cpp_adapter.httpx.Timeout(120.0),
            httpx_module=llama_cpp_adapter.httpx,
            request_id="llama-first",
        )

    assert calls == 2


def test_second_protocol_disconnect_is_not_retried_forever() -> None:
    calls = 0

    class Client:
        def post(self, _url: str, **_kwargs):
            nonlocal calls
            calls += 1
            raise llama_cpp_adapter.httpx.RemoteProtocolError("disconnect")

    with pytest.raises(llama_cpp_adapter.httpx.RemoteProtocolError, match="disconnect"):
        contract._post_completion_with_transport_replay(
            Client(),
            "http://127.0.0.1:8080/v1/chat/completions",
            payload={"messages": []},
            timeout=llama_cpp_adapter.httpx.Timeout(120.0),
            httpx_module=llama_cpp_adapter.httpx,
            request_id="llama-first",
        )

    assert calls == 2
