from __future__ import annotations

import httpx

from minecraft_mod_ai import llama_stream_efficiency_contract
from minecraft_mod_ai.model_adapters import llama_cpp_adapter


class _RecordingClient:
    def __init__(self) -> None:
        self.timeout: httpx.Timeout | None = None
        self.response = object()

    def post(self, endpoint: str, *, json: object, timeout: httpx.Timeout):
        assert endpoint == "http://127.0.0.1:8080/chat/completions"
        assert json == {"messages": []}
        self.timeout = timeout
        return self.response


def test_native_tool_completion_does_not_inherit_sse_read_deadline(monkeypatch) -> None:
    client = _RecordingClient()
    monkeypatch.setattr(
        llama_stream_efficiency_contract,
        "_client",
        lambda _server_url: client,
    )

    response = llama_cpp_adapter._post_completion(
        "http://127.0.0.1:8080",
        {"messages": []},
    )

    assert response is client.response
    assert client.timeout is not None
    assert client.timeout.read is None
    assert client.timeout.connect == 30.0
    assert client.timeout.write == 30.0
    assert client.timeout.pool == 30.0
