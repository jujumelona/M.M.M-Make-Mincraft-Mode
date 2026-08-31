from __future__ import annotations

import json
from types import TracebackType
from typing import Any, Self

import httpx

from minecraft_mod_ai.llama_stream_efficiency_contract import _StreamingCompletionClient


class _FakeStreamResponse:
    def __init__(self, lines: list[str], *, status_code: int = 200, body: bytes = b"") -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._lines = lines
        self._body = body

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def iter_lines(self):
        yield from self._lines

    def read(self) -> bytes:
        return self._body


class _FakeClient:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self.response = response
        self.stream_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.post_calls: list[tuple[str, dict[str, Any]]] = []

    def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        self.stream_calls.append((method, url, kwargs))
        return self.response

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.post_calls.append((url, kwargs))
        return httpx.Response(204, request=httpx.Request("POST", url))

    def close(self) -> None:
        return None


def _event(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, separators=(",", ":"))


def test_non_completion_post_still_delegates_normally() -> None:
    raw = _FakeClient(_FakeStreamResponse([]))
    client = _StreamingCompletionClient(raw)

    result = client.post("http://127.0.0.1:8080/health", json={"probe": True})

    assert result.status_code == 204
    assert len(raw.post_calls) == 1
    assert not raw.stream_calls


def test_plain_text_stream_must_reach_done_marker() -> None:
    response = _FakeStreamResponse(
        [
            _event(
                {
                    "choices": [
                        {"delta": {"content": "partial"}, "finish_reason": None}
                    ]
                }
            )
        ]
    )
    client = _StreamingCompletionClient(_FakeClient(response))

    try:
        client.post(
            "http://127.0.0.1:8080/chat/completions",
            json={"messages": []},
        )
    except RuntimeError as exc:
        assert "before the [DONE] marker" in str(exc)
    else:
        raise AssertionError("incomplete SSE stream must not be accepted")
