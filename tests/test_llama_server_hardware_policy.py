from __future__ import annotations

import sys
from types import SimpleNamespace

from minecraft_mod_ai import llama_server_hardware_policy as policy


class _FakeResponse:
    status_code = 200

    def __init__(self, *, text: str = "", lines: tuple[str, ...] = ()) -> None:
        self.text = text
        self._lines = lines

    def read(self) -> bytes:
        return b""

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        return False


class _FakeClient:
    def __init__(self) -> None:
        self.metrics_gets = 0
        self.stream_calls = 0
        self.closed = False

    def get(self, url: str, *, timeout: float) -> _FakeResponse:
        assert url.endswith("/metrics")
        self.metrics_gets += 1
        return _FakeResponse(text="unexpected auxiliary telemetry request")

    def stream(self, method: str, url: str, **_kwargs) -> _FakeResponse:
        assert method == "POST"
        assert url.endswith("/chat/completions")
        self.stream_calls += 1
        return _FakeResponse(
            lines=(
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                "data: [DONE]",
            )
        )

    def close(self) -> None:
        self.closed = True


def test_strict_server_generate_reuses_one_http_client_without_auxiliary_metrics(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_AUXILIARY_TELEMETRY", raising=False)
    client = _FakeClient()
    created: list[_FakeClient] = []
    fake_httpx = SimpleNamespace(
        Client=lambda **_kwargs: created.append(client) or client,
        Timeout=lambda **_kwargs: object(),
        Limits=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setattr(
        policy,
        "_TELEMETRY_TOTALS",
        {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "generation_seconds": 0.0,
            "requests": 0,
        },
    )

    class Adapter:
        _reported_server_url = None
        config = SimpleNamespace(
            role="code_generator",
            model_id="test-model",
            max_new_tokens=-1,
        )

    request = SimpleNamespace(
        messages=({"role": "user", "content": "x"},),
        response_format="text",
        tools=(),
    )

    result = policy._strict_server_generate(
        Adapter(), request, "http://127.0.0.1:8080/v1"
    )

    assert result == "ok"
    assert created == [client]
    assert client.metrics_gets == 0
    assert client.stream_calls == 1
    assert not client.closed


def test_auxiliary_native_telemetry_is_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_AUXILIARY_TELEMETRY", raising=False)
    assert policy._auxiliary_native_telemetry_enabled() is False
    monkeypatch.setenv("MMM_LLAMA_AUXILIARY_TELEMETRY", "true")
    assert policy._auxiliary_native_telemetry_enabled() is True
