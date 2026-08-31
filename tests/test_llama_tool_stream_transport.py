from __future__ import annotations

import json

import httpx
import pytest

from minecraft_mod_ai.llama_stream_efficiency_contract import _StreamingCompletionClient


class _StreamResponse:
    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self.headers: dict[str, str] = {}
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def iter_lines(self):
        return iter(self._lines)


class _RawClient:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.post_calls = 0
        self.stream_calls = 0
        self.stream_payload: dict[str, object] | None = None

    def post(self, url: str, **kwargs):
        del url, kwargs
        self.post_calls += 1
        raise AssertionError("production tool completion must not use native blocking post")

    def stream(self, method: str, url: str, **kwargs):
        del method, url
        self.stream_calls += 1
        self.stream_payload = dict(kwargs["json"])
        return _StreamResponse(self.lines)

    def get(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("probe not needed for a completed synthetic stream")


def _event(payload: dict[str, object]) -> str:
    return "data: " + json.dumps(payload, separators=(",", ":"))


def test_tool_completion_uses_sse_and_reassembles_structured_tool_call() -> None:
    raw = _RawClient(
        [
            _event(
                {
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_7",
                                        "type": "function",
                                        "function": {
                                            "name": "look",
                                            "arguments": "{",
                                        },
                                    }
                                ],
                            },
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
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "up",
                                            "arguments": '\"q\":\"x\"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
            _event(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                }
            ),
            "data: [DONE]",
        ]
    )
    client = _StreamingCompletionClient(raw)

    response = client.post(
        "http://llama.test/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "lookup x"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "stream": False,
        },
        timeout=httpx.Timeout(connect=1.0, read=120.0, write=1.0, pool=1.0),
    )

    assert raw.post_calls == 0
    assert raw.stream_calls == 1
    assert raw.stream_payload is not None
    assert raw.stream_payload["stream"] is True
    assert raw.stream_payload["stream_options"] == {"include_usage": True}

    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["tool_calls"] == [
        {
            "id": "call_7",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    assert data["usage"]["completion_tokens"] == 3


def test_tool_stream_without_done_marker_is_never_promoted_to_completed_message() -> None:
    raw = _RawClient(
        [
            _event(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_partial",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": "{",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            )
        ]
    )
    client = _StreamingCompletionClient(raw)

    with pytest.raises(RuntimeError, match=r"before the \[DONE\] marker"):
        client.post(
            "http://llama.test/v1/chat/completions",
            json={
                "messages": [],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            },
            timeout=httpx.Timeout(connect=1.0, read=120.0, write=1.0, pool=1.0),
        )
