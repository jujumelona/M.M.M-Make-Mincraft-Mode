from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from minecraft_mod_ai import llama_completion_liveness_contract as contract
from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract
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


def test_tool_semantic_idle_scales_to_output_budget_without_changing_ping_policy(monkeypatch) -> None:
    stream_module = SimpleNamespace(
        _tool_idle_timeout_seconds=lambda: 120.0,
        _stream_idle_timeout_seconds=lambda: 120.0,
    )
    for name in (
        "MMM_LLAMA_TOOL_SEMANTIC_TPS_FLOOR",
        "MMM_LLAMA_TOOL_SEMANTIC_GRACE_SECONDS",
        "MMM_LLAMA_TOOL_SEMANTIC_MAX_IDLE_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    short_tool = {"tools": [{"type": "function"}], "max_tokens": 512}
    long_tool = {"tools": [{"type": "function"}], "max_tokens": 4096}
    huge_tool = {"tools": [{"type": "function"}], "max_tokens": 32768}

    assert contract._semantic_idle_timeout_seconds(stream_module, short_tool) == 120.0
    assert contract._semantic_idle_timeout_seconds(stream_module, long_tool) == pytest.approx(439.6)
    assert contract._semantic_idle_timeout_seconds(stream_module, huge_tool) == 480.0
    assert contract._semantic_idle_timeout_seconds(
        stream_module, {"max_tokens": 4096}
    ) == 120.0
    assert contract._ping_interval_seconds(stream_module, long_tool) == 30


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
        started_at=contract.time.monotonic(),
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


def test_runtime_client_owns_semantic_progress_without_install(monkeypatch) -> None:
    created = []

    class RawClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def stream(self, method, url, **kwargs):
            return method, url, kwargs

        def close(self):
            return None

    monkeypatch.setattr(httpx, "Client", RawClient)
    stream_contract._CLIENTS.clear()

    client = stream_contract._client("http://127.0.0.1:18080")

    assert created
    assert getattr(client._client, "_mmm_semantic_progress_client_v1", False) is True


def test_stream_payload_requests_semantic_progress_without_install() -> None:
    captured = {}

    class RawClient:
        def stream(self, method, url, **kwargs):
            captured["method"] = method
            captured["url"] = url
            captured["kwargs"] = kwargs
            return object()

    client = stream_contract._StreamingCompletionClient(RawClient())
    client.stream(
        "POST",
        "http://127.0.0.1:8080/chat/completions",
        json={"stream": True, "messages": []},
    )

    payload = captured["kwargs"]["json"]
    assert payload["return_progress"] is True
    assert payload["sse_ping_interval"] <= 30


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
        started_at=contract.time.monotonic(),
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


def test_active_decode_reports_periodic_semantic_progress(monkeypatch, capsys) -> None:
    ticks = iter([0.0, 0.1, 0.2, 20.0, 20.1, 20.2, 20.3, 20.4])
    monkeypatch.setattr(contract.time, "monotonic", lambda: next(ticks))
    monkeypatch.setenv("MMM_LLAMA_PROGRESS_LOG_INTERVAL_SECONDS", "15")

    response = SimpleNamespace(
        iter_lines=lambda: iter(
            [
                'data: {"choices":[{"delta":{"content":"a"}}]}',
                'data: {"choices":[{"delta":{"content":"b"}}]}',
                "data: [DONE]",
            ]
        )
    )
    wrapped = contract._ProgressCheckedResponse(
        response,
        120.0,
        request_id="periodic-progress",
        started_at=0.0,
    )

    list(wrapped.iter_lines())
    output = capsys.readouterr().out

    assert "first semantic progress" in output
    assert "llama server: semantic progress" in output
    assert "events=2" in output


def test_completion_wall_deadline_does_not_refresh_with_progress(monkeypatch) -> None:
    ticks = iter([0.0, 10.0, 301.0, 301.1])
    monkeypatch.setattr(contract.time, "monotonic", lambda: next(ticks))
    response = SimpleNamespace(
        iter_lines=lambda: iter(
            [
                'data: {"choices":[{"delta":{"content":"a"}}]}',
                'data: {"choices":[{"delta":{"content":"b"}}]}',
            ]
        )
    )
    wrapped = contract._ProgressCheckedResponse(
        response,
        120.0,
        request_id="hard-wall",
        started_at=0.0,
        wall_seconds=300.0,
    )

    with pytest.raises(contract.LlamaSemanticProgressTimeout, match="wall-clock ceiling"):
        list(wrapped.iter_lines())
