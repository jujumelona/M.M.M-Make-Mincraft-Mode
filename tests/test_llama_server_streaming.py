from __future__ import annotations

from types import SimpleNamespace

import httpx

from minecraft_mod_ai.llama_server_hardware_policy import _strict_server_generate


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


def test_local_mtp_generation_uses_sse_without_fixed_read_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_stream(method, url, *, json, timeout):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _StreamingResponse()

    monkeypatch.setattr(httpx, "stream", fake_stream)

    adapter = SimpleNamespace(
        config=SimpleNamespace(
            role="planner",
            model_id="test/model",
            max_new_tokens=8192,
        )
    )
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
    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 30.0
    assert timeout.read is None
    assert timeout.write == 30.0
    assert timeout.pool == 30.0


def test_local_mtp_stream_requires_done_marker(monkeypatch) -> None:
    class _BrokenResponse(_StreamingResponse):
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'

    monkeypatch.setattr(httpx, "stream", lambda *args, **kwargs: _BrokenResponse())

    adapter = SimpleNamespace(
        config=SimpleNamespace(
            role="planner",
            model_id="test/model",
            max_new_tokens=8192,
        )
    )
    request = SimpleNamespace(
        messages=({"role": "user", "content": "x"},),
        response_format="text",
    )

    try:
        _strict_server_generate(adapter, request, "http://127.0.0.1:8910/v1")
    except Exception as exc:
        assert "stream ended before the [DONE] marker" in str(exc)
    else:
        raise AssertionError("truncated SSE stream must fail closed")
