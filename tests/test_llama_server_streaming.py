from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from minecraft_mod_ai import llama_stream_efficiency_contract as stream_runtime
from minecraft_mod_ai.llama_server_hardware_policy import _strict_server_generate


class _Adapter:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            role="planner",
            model_id="test/model",
            max_new_tokens=8192,
        )


class _StreamingResponse:
    status_code = 200
    text = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b""

    def iter_lines(self):
        yield ': ping - connection liveness only'
        yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"{\\"ok\\":"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"true}"}}]}'
        yield 'data: [DONE]'


class _Client:
    def __init__(self, response, captured: dict[str, object]) -> None:
        self.response = response
        self.captured = captured

    def stream(self, method, url, *, json):
        self.captured["method"] = method
        self.captured["url"] = url
        self.captured["json"] = json
        return self.response


def test_persistent_client_has_bounded_idle_read_timeout() -> None:
    client = stream_runtime._client("http://127.0.0.1:18910/v1")
    timeout = client.timeout
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 30.0
    assert timeout.read == 120.0
    assert timeout.write == 30.0
    assert timeout.pool == 30.0


def test_local_native_generation_uses_persistent_sse_client(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        stream_runtime,
        "_client",
        lambda _server_url: _Client(_StreamingResponse(), captured),
    )

    adapter = _Adapter()
    request = SimpleNamespace(
        messages=({"role": "user", "content": "return json"},),
        response_format="json",
    )

    result = _strict_server_generate(
        adapter,
        request,
        "http://127.0.0.1:8910/v1",
    )

    assert result == '{"ok":true}'
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8910/v1/chat/completions"
    assert captured["json"]["stream"] is True
    assert captured["json"]["stream_options"] == {"include_usage": True}


def test_local_native_stream_requires_done_marker(monkeypatch) -> None:
    class _BrokenResponse(_StreamingResponse):
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'

    monkeypatch.setattr(
        stream_runtime,
        "_client",
        lambda _server_url: _Client(_BrokenResponse(), {}),
    )

    adapter = _Adapter()
    request = SimpleNamespace(
        messages=({"role": "user", "content": "x"},),
        response_format="text",
    )

    with pytest.raises(RuntimeError, match=r"stream ended before the \[DONE\] marker"):
        _strict_server_generate(adapter, request, "http://127.0.0.1:8910/v1")
