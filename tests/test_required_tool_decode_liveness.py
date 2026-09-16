from __future__ import annotations

import json

import httpx

from minecraft_mod_ai.llama_stream_efficiency_contract import _StreamingCompletionClient
from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _native_tool_generation_response


def _apply_source_edit_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "Apply one atomic source edit.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["operation", "path"],
                "properties": {
                    "operation": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    }


def _sse(delta: dict[str, object], *, finish_reason=None) -> str:
    return "data: " + json.dumps(
        {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    )


class _FakeStreamResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, lines: list[str], *, forbid_after: int | None = None) -> None:
        self._lines = lines
        self._forbid_after = forbid_after
        self.lines_requested = 0
        self.saw_done = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return b""

    def iter_lines(self):
        for line in self._lines:
            self.lines_requested += 1
            if self._forbid_after is not None and self.lines_requested > self._forbid_after:
                raise AssertionError("required-tool semantic guard failed to abort immediately")
            if line == "data: [DONE]":
                self.saw_done = True
            yield line


class _FakeStreamingClient:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self.response = response

    def stream(self, method: str, url: str, **kwargs):
        return self.response


def _post(response: _FakeStreamResponse, *, tool_choice="required") -> dict[str, object]:
    tool = _apply_source_edit_tool()
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "perform the edit"}],
        "max_tokens": 24930,
        "tools": [tool],
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
    }
    return _StreamingCompletionClient(_FakeStreamingClient(response)).post(
        "http://llama.local/v1/chat/completions",
        json=payload,
        timeout=httpx.Timeout(120.0),
    ).json()


def test_required_tool_reasoning_is_rejected_on_first_semantic_delta() -> None:
    response = _FakeStreamResponse(
        [
            _sse({"reasoning_content": "I need to think first."}),
            _sse({"reasoning_content": "this must never be requested"}),
        ],
        forbid_after=1,
    )

    data = _post(response)
    choice = data["choices"][0]

    assert choice["finish_reason"] == "stop"
    assert choice["message"].get("tool_calls") in (None, [])
    assert choice["message"]["reasoning_content"] == "I need to think first."
    assert response.lines_requested == 1
    assert response.saw_done is False

    tool = _apply_source_edit_tool()
    request = GenerationRequest(
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice={
            "type": "function",
            "function": {"name": "apply_source_edit"},
        },
        parallel_tool_calls=False,
    )
    generation = _native_tool_generation_response(choice["message"], request)

    assert len(generation.tool_calls) == 1
    rejected = generation.tool_calls[0]
    assert rejected.name == "__mmm_rejected_tool_call__"
    assert rejected.arguments["failure_code"] == "REQUIRED_TOOL_MISSING"


def test_required_tool_fragmented_text_marker_is_not_rejected() -> None:
    response = _FakeStreamResponse(
        [
            _sse({"content": "   <"}),
            _sse({"content": "tool_"}),
            _sse({"content": "call>"}),
            _sse({"content": '{"name":"apply_source_edit"}'}),
            "data: [DONE]",
        ]
    )

    data = _post(response)
    choice = data["choices"][0]

    assert response.saw_done is True
    assert choice["message"]["content"].lstrip().startswith("<tool_call>")


def test_required_native_tool_allows_large_arguments_after_invocation_starts() -> None:
    large_content = "x" * 20000
    arguments = json.dumps(
        {
            "operation": "write",
            "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
            "content": large_content,
        }
    )
    response = _FakeStreamResponse(
        [
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "apply_source_edit",
                                "arguments": arguments,
                            },
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
    )

    data = _post(response)
    message = data["choices"][0]["message"]

    assert response.saw_done is True
    tool_call = message["tool_calls"][0]
    decoded = json.loads(tool_call["function"]["arguments"])
    assert decoded["content"] == large_content


def test_optional_tool_turn_preserves_reasoning_stream() -> None:
    response = _FakeStreamResponse(
        [
            _sse({"reasoning_content": "Reasoning is allowed when no tool is required."}),
            _sse({"content": "No edit needed."}),
            "data: [DONE]",
        ]
    )

    data = _post(response, tool_choice="auto")
    message = data["choices"][0]["message"]

    assert response.saw_done is True
    assert message["reasoning_content"].startswith("Reasoning is allowed")
    assert message["content"] == "No edit needed."


def test_explicit_function_choice_uses_required_tool_semantics() -> None:
    response = _FakeStreamResponse(
        [
            _sse({"content": "I will explain before editing."}),
            "data: [DONE]",
        ],
        forbid_after=1,
    )
    explicit = {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }

    data = _post(response, tool_choice=explicit)

    assert data["choices"][0]["finish_reason"] == "stop"
    assert response.lines_requested == 1
    assert response.saw_done is False
