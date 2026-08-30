from __future__ import annotations

import httpx

from minecraft_mod_ai import llama_sse_error_contract as contract
from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract


class _FakeResponse:
    status_code = 200

    def __init__(self, lines: tuple[str, ...]) -> None:
        self.headers: dict[str, str] = {}
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        yield from self._lines


class _FakeRawClient:
    def __init__(self, lines: tuple[str, ...]) -> None:
        self.lines = lines

    def stream(self, method: str, url: str, **kwargs):
        assert method == "POST"
        assert url.endswith("/chat/completions")
        return _FakeResponse(self.lines)

    def close(self):
        return None


def test_current_openai_compatible_sse_error_is_detected() -> None:
    parsed = contract._sse_error_from_line(
        'data: {"error":{"code":400,"message":"the request exceeds the available context size","type":"invalid_request_error"}}'
    )
    assert parsed is not None
    status, error = parsed
    assert status == 400
    assert "exceeds the available context size" in error["message"]


def test_legacy_llama_sse_error_field_is_detected() -> None:
    parsed = contract._sse_error_from_line(
        'error: {"code":400,"message":"the request exceeds the available context size","type":"invalid_request_error"}'
    )
    assert parsed is not None
    assert parsed[0] == 400


def test_streamed_error_becomes_http_like_error_for_finish_reason_classifier() -> None:
    raw = _FakeRawClient(
        (
            'data: {"error":{"code":400,"message":"the request exceeds the available context size","type":"invalid_request_error"}}',
            "data: [DONE]",
        )
    )
    client = stream_contract._StreamingCompletionClient(raw)
    response = client.post(
        "http://127.0.0.1:8080/chat/completions",
        json={"messages": []},
        timeout=httpx.Timeout(120.0),
    )

    assert response.status_code == 400
    assert "exceeds the available context size" in response.text
    assert '"error"' in response.text


def test_runtime_installs_sse_error_preservation() -> None:
    assert getattr(
        stream_contract._StreamingCompletionClient.post,
        "_mmm_sse_server_error_response_v1",
        False,
    )
