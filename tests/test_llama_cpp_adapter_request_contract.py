from __future__ import annotations

import httpx
import pytest

from minecraft_mod_ai.model_adapters.base import (
    AdapterConfig,
    GenerationRequest,
    ModelBackendError,
)
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


class _HealthResponse:
    status_code = 200

    @staticmethod
    def raise_for_status() -> None:
        return None


class _CompletionResponse:
    def __init__(self, *, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _adapter() -> LlamaCppAdapter:
    return LlamaCppAdapter(
        AdapterConfig(
            role="planner",
            adapter="llama_cpp",
            model_id="test/qwen35",
            max_new_tokens=512,
        )
    )


def test_generate_turn_accepts_direct_final_json_without_native_tool_transport(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())

    def post(url, *, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _CompletionResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": '{"game_design":{}}'}}]},
        )

    monkeypatch.setattr(httpx, "post", post)
    request = GenerationRequest(
        messages=({"role": "user", "content": "plan"},),
        response_format="json",
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    turn = _adapter().generate_turn(request)

    assert turn.content == '{"game_design":{}}'
    payload = captured["payload"]
    for forbidden in (
        "tools", "tool_choice", "parallel_tool_calls",
        "response_format", "json_schema", "grammar",
    ):
        assert forbidden not in payload
    rendered = "\n".join(str(message.get("content", "")) for message in payload["messages"])
    assert "REVIEWED_TOOL_CATALOG" in rendered
    assert "lookup" in rendered


def test_generate_turn_parses_host_tool_envelope_without_native_tool_fields(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())

    def post(url, *, json, timeout):
        captured["payload"] = json
        return _CompletionResponse(
            status_code=200,
            payload={
                "choices": [{"message": {"content": (
                    '{"kind":"tool_calls","calls":['
                    '{"id":"call_7","name":"lookup","arguments":{"q":"x"}}]}'
                )}}]
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    request = GenerationRequest(
        messages=({"role": "user", "content": "look it up"},),
        response_format="json",
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    turn = _adapter().generate_turn(request)
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_7"
    assert turn.tool_calls[0].name == "lookup"
    assert turn.tool_calls[0].arguments == {"q": "x"}
    for forbidden in (
        "tools", "tool_choice", "parallel_tool_calls",
        "response_format", "json_schema", "grammar",
    ):
        assert forbidden not in captured["payload"]


def test_generate_turn_preserves_llama_server_400_body_without_prompt(monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: _CompletionResponse(
            status_code=400,
            text='{"error":{"message":"unsupported request field"}}',
        ),
    )

    with pytest.raises(ModelBackendError) as caught:
        _adapter().generate_turn(
            GenerationRequest(
                messages=({"role": "user", "content": "SECRET_PROMPT_SENTINEL"},),
                response_format="json",
            )
        )

    message = str(caught.value)
    assert "HTTP 400" in message
    assert "unsupported request field" in message
    assert "SECRET_PROMPT_SENTINEL" not in message


def test_generate_turn_keeps_detailed_schema_host_side_for_non_tool_json(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())

    def post(url, *, json, timeout):
        captured["payload"] = json
        return _CompletionResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": '{"value":"ok"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", post)
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    request = GenerationRequest(
        messages=({"role": "user", "content": "structured"},),
        response_format="json",
        response_schema=schema,
    )
    turn = _adapter().generate_turn(request)

    assert turn.content == '{"value":"ok"}'
    assert request.response_schema == schema
    assert "response_format" not in captured["payload"]
    assert "json_schema" not in captured["payload"]
    assert "grammar" not in captured["payload"]
