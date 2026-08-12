from __future__ import annotations

import threading
from types import SimpleNamespace

from minecraft_mod_ai import llama_stream_efficiency_contract as contract


class _Response:
    status_code = 200
    text = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b""

    def iter_lines(self):
        return iter(
            [
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
                "data: [DONE]",
            ]
        )


class _Client:
    def __init__(self) -> None:
        self.calls = []

    def stream(self, method, endpoint, *, json):
        self.calls.append((method, endpoint, json))
        return _Response()


class _Adapter:
    _reported_server_url = None

    def __init__(self) -> None:
        self.config = SimpleNamespace(role="planner", model_id="model")


def _hardware(old):
    def parts(choice):
        delta = choice.get("delta") or {}
        return str(delta.get("reasoning_content") or ""), str(delta.get("content") or "")

    return SimpleNamespace(
        _strict_server_generate=old,
        _server_payload=lambda _adapter, _request: {
            "model": "local",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 8,
            "temperature": 0.0,
        },
        _request_content_chars=lambda payload: len(payload["messages"][0]["content"]),
        _stream_delta_parts=parts,
        _TELEMETRY_LOCK=threading.Lock(),
        _TELEMETRY_TOTALS={
            "prompt_tokens": 0,
            "output_tokens": 0,
            "generation_seconds": 0.0,
            "requests": 0,
        },
    )


def test_stream_usage_avoids_auxiliary_telemetry_http(monkeypatch) -> None:
    old_calls = []

    def old(*_args):
        old_calls.append(True)
        return "old"

    hardware = _hardware(old)
    client = _Client()
    monkeypatch.setattr(contract, "_client", lambda _url: client)
    monkeypatch.delenv("MMM_LLAMA_DETAILED_TELEMETRY", raising=False)

    contract.install(hardware)
    result = hardware._strict_server_generate(
        _Adapter(),
        SimpleNamespace(),
        "http://127.0.0.1:8080/v1",
    )

    assert result == "ok"
    assert not old_calls
    assert len(client.calls) == 1
    payload = client.calls[0][2]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert hardware._TELEMETRY_TOTALS["prompt_tokens"] == 5
    assert hardware._TELEMETRY_TOTALS["output_tokens"] == 2
    assert hardware._TELEMETRY_TOTALS["requests"] == 1


def test_detailed_telemetry_opt_in_preserves_original_path(monkeypatch) -> None:
    hardware = _hardware(lambda *_args: "detailed")
    monkeypatch.setenv("MMM_LLAMA_DETAILED_TELEMETRY", "1")

    contract.install(hardware)

    assert hardware._strict_server_generate(
        _Adapter(),
        SimpleNamespace(),
        "http://127.0.0.1:8080/v1",
    ) == "detailed"
