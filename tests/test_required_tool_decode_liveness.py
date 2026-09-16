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


class _FakeStreamResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return b""

    def iter_lines(self):
        for text in ("abcdefghij", "klmnopqrst"):
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"reasoning_content": text}}]}
            )
        raise AssertionError("required-tool preface guard failed to abort the stream")


class _FakeStreamingClient:
    def stream(self, method: str, url: str, **kwargs):
        return _FakeStreamResponse()


def test_required_tool_reasoning_preface_is_aborted_into_nonexecuting_rejection(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_REQUIRED_TOOL_PREFACE_CHARS", "16")
    tool = _apply_source_edit_tool()
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "perform the edit"}],
        "max_tokens": 24930,
        "tools": [tool],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }

    response = _StreamingCompletionClient(_FakeStreamingClient()).post(
        "http://llama.local/v1/chat/completions",
        json=payload,
        timeout=httpx.Timeout(120.0),
    )
    data = response.json()
    choice = data["choices"][0]

    assert choice["finish_reason"] == "stop"
    assert choice["message"].get("tool_calls") in (None, [])
    assert choice["message"]["reasoning_content"] == "abcdefghijklmnopqrst"

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
