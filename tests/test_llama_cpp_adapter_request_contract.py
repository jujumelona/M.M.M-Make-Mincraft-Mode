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
            model_id="test/qwen3.5-9b",
            max_new_tokens=512,
        )
    )


def _tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_generate_turn_accepts_final_content_while_native_tools_are_available(monkeypatch) -> None:
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
        tools=(_tool(),),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    turn = _adapter().generate_turn(request)

    assert turn.content == '{"game_design":{}}'
    payload = captured["payload"]
    assert payload["tools"] == [_tool()]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    for forbidden in ("response_format", "json_schema", "grammar"):
        assert forbidden not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["presence_penalty"] == 1.5


def test_generate_turn_parses_native_openai_tool_calls(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())

    def post(url, *, json, timeout):
        captured["payload"] = json
        return _CompletionResponse(
            status_code=200,
            payload={
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "call_7",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q":"x"}',
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    request = GenerationRequest(
        messages=({"role": "user", "content": "look it up"},),
        response_format="json",
        tools=(_tool(),),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    turn = _adapter().generate_turn(request)
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_7"
    assert turn.tool_calls[0].name == "lookup"
    assert turn.tool_calls[0].arguments == {"q": "x"}
    assert captured["payload"]["tools"] == [_tool()]
    rendered = "\n".join(
        str(message.get("content", ""))
        for message in captured["payload"]["messages"]
    )
    assert "mmm/host-tool-envelope" not in rendered


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
