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
        snapshots = (
            "llamacpp:prompt_tokens_total 10\n"
            "llamacpp:tokens_predicted_total 5\n"
            "llamacpp:tokens_predicted_seconds_total 1\n",
            "llamacpp:prompt_tokens_total 12\n"
            "llamacpp:tokens_predicted_total 6\n"
            "llamacpp:tokens_predicted_seconds_total 1.5\n",
        )
        response = _FakeResponse(text=snapshots[self.metrics_gets])
        self.metrics_gets += 1
        return response

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


def test_strict_server_generate_reuses_one_http_client(monkeypatch) -> None:
    client = _FakeClient()
    created: list[_FakeClient] = []
    fake_httpx = SimpleNamespace(
        Client=lambda: created.append(client) or client,
        Timeout=lambda **_kwargs: object(),
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
    assert client.metrics_gets == 2
    assert client.stream_calls == 1
    assert client.closed
