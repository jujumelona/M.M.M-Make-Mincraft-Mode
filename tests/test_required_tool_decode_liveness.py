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

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
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


def _native_call(arguments: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "arguments": json.dumps(arguments or {
                        "operation": "replace_exact",
                        "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
                    }),
                },
            }
        ]
    }


def test_required_tool_reasoning_can_precede_native_tool_call() -> None:
    response = _FakeStreamResponse(
        [
            _sse({"reasoning_content": "I need to inspect the repair first."}),
            _sse(_native_call()),
            "data: [DONE]",
        ]
    )

    data = _post(response)
    choice = data["choices"][0]

    assert response.saw_done is True
    assert response.lines_requested == 3
    assert choice["message"]["reasoning_content"].startswith("I need to inspect")
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "apply_source_edit"

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
    assert generation.tool_calls[0].name == "apply_source_edit"


def test_required_tool_completed_without_tool_is_rejected_after_done() -> None:
    response = _FakeStreamResponse(
        [
            _sse({"reasoning_content": "I considered the edit."}),
            _sse({"content": "No tool call."}),
            "data: [DONE]",
        ]
    )

    data = _post(response)
    choice = data["choices"][0]
    assert response.saw_done is True

    tool = _apply_source_edit_tool()
    request = GenerationRequest(
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
        parallel_tool_calls=False,
    )
    generation = _native_tool_generation_response(choice["message"], request)
    assert len(generation.tool_calls) == 1
    rejected = generation.tool_calls[0]
    assert rejected.name == "__mmm_rejected_tool_call__"
    assert rejected.arguments["failure_code"] == "REQUIRED_TOOL_MISSING"


def test_required_tool_long_semantic_preface_is_rejected_before_full_decode() -> None:
    response = _FakeStreamResponse(
        [
            _sse({"content": "x" * 1200}),
            _sse(_native_call()),
            "data: [DONE]",
        ]
    )

    data = _post(response)
    choice = data["choices"][0]

    assert response.saw_done is False
    assert response.lines_requested == 1
    tool = _apply_source_edit_tool()
    request = GenerationRequest(
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
        parallel_tool_calls=False,
    )
    generation = _native_tool_generation_response(choice["message"], request)
    assert generation.tool_calls[0].name == "__mmm_rejected_tool_call__"
    assert generation.tool_calls[0].arguments["failure_code"] == "REQUIRED_TOOL_MISSING"


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
    response = _FakeStreamResponse(
        [
            _sse(_native_call({
                "operation": "write",
                "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
                "content": large_content,
            })),
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


def test_explicit_function_choice_allows_preface_then_native_tool() -> None:
    response = _FakeStreamResponse(
        [
            _sse({"content": "I will repair the existing file."}),
            _sse(_native_call()),
            "data: [DONE]",
        ]
    )
    explicit = {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }

    data = _post(response, tool_choice=explicit)

    assert response.saw_done is True
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "apply_source_edit"
