from __future__ import annotations

import httpx

from minecraft_mod_ai import llama_stream_efficiency_contract
from minecraft_mod_ai.model_adapters import llama_cpp_adapter


class _RecordingClient:
    def __init__(self) -> None:
        self.timeout: httpx.Timeout | None = None
        self.headers: dict[str, str] | None = None
        self.response = object()

    def post(
        self,
        endpoint: str,
        *,
        json: object,
        timeout: httpx.Timeout,
        headers: dict[str, str] | None = None,
    ):
        assert endpoint == "http://127.0.0.1:8080/chat/completions"
        assert json == {"messages": []}
        self.timeout = timeout
        self.headers = headers
        return self.response


def test_native_tool_completion_has_bounded_idle_read_timeout(monkeypatch) -> None:
    client = _RecordingClient()
    monkeypatch.setattr(
        llama_stream_efficiency_contract,
        "_client",
        lambda _server_url: client,
    )
    monkeypatch.delenv("MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS", raising=False)

    response = llama_cpp_adapter._post_completion(
        "http://127.0.0.1:8080",
        {"messages": []},
    )

    assert response is client.response
    assert client.timeout is not None
    assert client.timeout.read == 120.0
    assert client.timeout.connect == 30.0
    assert client.timeout.write == 30.0
    assert client.timeout.pool == 30.0
    assert client.headers is not None
    assert client.headers["X-MMM-Request-Id"].startswith("llama-")


def test_native_tool_completion_honors_explicit_idle_timeout(monkeypatch) -> None:
    client = _RecordingClient()
    monkeypatch.setattr(
        llama_stream_efficiency_contract,
        "_client",
        lambda _server_url: client,
    )
    monkeypatch.setenv("MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS", "75")

    llama_cpp_adapter._post_completion(
        "http://127.0.0.1:8080",
        {"messages": []},
    )

    assert client.timeout is not None
    assert client.timeout.read == 75.0
    assert client.headers is not None
    assert client.headers["X-MMM-Request-Id"].startswith("llama-")
