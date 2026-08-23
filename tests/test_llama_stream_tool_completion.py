from __future__ import annotations

import json
from typing import Any

import httpx

from minecraft_mod_ai.llama_stream_efficiency_contract import _StreamingCompletionClient


class _FakeStreamResponse:
    def __init__(self, lines: list[str], *, status_code: int = 200, body: bytes = b"") -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._lines = lines
        self._body = body

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
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


def test_tool_completion_post_is_received_as_sse_and_reassembled() -> None:
    response = _FakeStreamResponse(
        [
            _event(
                {
                    "choices": [
                        {
                            "delta": {"role": "assistant", "reasoning_content": "inspect "},
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            _event(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": (
                                    "<tool_call><function=apply_source_edit>"
                                    "<parameter=path>src/A.java</parameter>"
                                    "</function></tool_call>"
                                )
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            _event(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                }
            ),
            "data: [DONE]",
        ]
    )
    raw = _FakeClient(response)
    client = _StreamingCompletionClient(raw)
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "repair"}],
        "tools": [{"type": "function", "function": {"name": "apply_source_edit"}}],
        "max_tokens": -1,
    }

    result = client.post(
        "http://127.0.0.1:8080/chat/completions",
        json=payload,
        timeout=httpx.Timeout(600.0),
    )

    assert not raw.post_calls
    assert len(raw.stream_calls) == 1
    method, _, kwargs = raw.stream_calls[0]
    assert method == "POST"
    assert kwargs["json"]["stream"] is True
    assert kwargs["json"]["stream_options"] == {"include_usage": True}
    assert payload.get("stream") is None

    data = result.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["reasoning_content"] == "inspect "
    assert "<function=apply_source_edit>" in choice["message"]["content"]
    assert data["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}


def test_non_completion_post_still_delegates_normally() -> None:
    raw = _FakeClient(_FakeStreamResponse([]))
    client = _StreamingCompletionClient(raw)

    result = client.post("http://127.0.0.1:8080/health", json={"probe": True})

    assert result.status_code == 204
    assert len(raw.post_calls) == 1
    assert not raw.stream_calls


def test_stream_must_reach_done_marker() -> None:
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
            json={"messages": [], "tools": [{}]},
        )
    except RuntimeError as exc:
        assert "before the [DONE] marker" in str(exc)
    else:
        raise AssertionError("incomplete SSE stream must not be accepted")
