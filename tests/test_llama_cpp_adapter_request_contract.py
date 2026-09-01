from __future__ import annotations

import httpx
import pytest

from minecraft_mod_ai import llama_exact_context, llama_server_autotune
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


@pytest.fixture(autouse=True)
def _stub_canonical_server_owner(monkeypatch) -> None:
    monkeypatch.setattr(
        llama_server_autotune,
        "ensure_tuned_server",
        lambda _config, _request: "http://127.0.0.1:8910/v1",
    )
    monkeypatch.setattr(
        llama_exact_context,
        "capacity_safe_payload",
        lambda _url, payload: dict(payload),
    )


def _adapter() -> LlamaCppAdapter:
    return LlamaCppAdapter(
        AdapterConfig(
            role="planner",
            adapter="llama_cpp",
            model_id="test/qwen3.5-9b",
            max_new_tokens=512,
            extra={
                "runtime_contract": "qwen",
                "qwen_family": "qwen3.5",
                "qwen_tool_markup": "qwen3_coder_xml",
                "qwen_action_thinking_control": "enable_thinking_false",
                "qwen_preserve_thinking": False,
                "qwen_reasoning_effort": False,
                "qwen_assistant_prefill": True,
                "request_policy": "task_aware_sampling",
                "sampling_profiles": {
                    "non_thinking": {
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 20,
                        "min_p": 0.0,
                        "presence_penalty": 1.5,
                        "repeat_penalty": 1.0,
                    }
                },
            },
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


def _source_edit_tool(*, include_action: bool = False) -> dict[str, object]:
    properties: dict[str, object] = {
        "operation": {
            "type": "string",
            "enum": ["replace_exact", "insert_before", "insert_after"],
        },
        "path": {"type": "string"},
    }
    if include_action:
        properties["action"] = {"type": "string"}
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "apply one exact source edit",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["operation", "path"],
                "additionalProperties": False,
            },
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
    assert "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 1.5
    assert payload["repeat_penalty"] == 1.0


def test_generate_turn_accepts_host_validated_server_parsed_openai_tool_calls(monkeypatch) -> None:
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
    assert captured["payload"]["tool_choice"] == "auto"


def test_reasoning_only_turn_is_completed_once_into_a_semantic_action(monkeypatch) -> None:
    payloads: list[dict[str, object]] = []
    responses = [
        _CompletionResponse(
            status_code=200,
            payload={
                "choices": [{
                    "message": {
                        "content": "",
                        "reasoning_content": "I need exact evidence before answering.",
                    }
                }]
            },
        ),
        _CompletionResponse(
            status_code=200,
            payload={
                "choices": [{
                    "message": {
                        "content": (
                            "<tool_call><function=lookup>"
                            "<parameter=q>exact api</parameter>"
                            "</function></tool_call>"
                        ),
                    }
                }]
            },
        ),
    ]
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")

    def post(url, *, json, timeout):
        payloads.append(json)
        return responses.pop(0)

    monkeypatch.setattr(httpx, "post", post)
    turn = _adapter().generate_turn(
        GenerationRequest(
            messages=({"role": "user", "content": "inspect then act"},),
            tools=(_tool(),),
            tool_choice="auto",
        )
    )

    assert len(payloads) == 2
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "lookup"
    assert turn.tool_calls[0].arguments == {"q": "exact api"}
    assert turn.reasoning_content == "I need exact evidence before answering."
    continuation_messages = payloads[1]["messages"]
    assert continuation_messages[-2]["role"] == "assistant"
    assert continuation_messages[-2]["reasoning_content"] == "I need exact evidence before answering."
    assert continuation_messages[-1]["role"] == "user"
    assert "Do not return another reasoning-only response" in continuation_messages[-1]["content"]
    assert payloads[1]["tools"] == [_tool()]
    assert payloads[1]["tool_choice"] == "auto"


def test_pure_content_qwen_reasoning_is_split_before_host_tool_parse(monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: _CompletionResponse(
            status_code=200,
            payload={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>inspect the current schema</think>\n"
                                "<tool_call><function=lookup>"
                                "<parameter=q>exact api</parameter>"
                                "</function></tool_call>"
                            )
                        }
                    }
                ]
            },
        ),
    )

    turn = _adapter().generate_turn(
        GenerationRequest(
            messages=({"role": "user", "content": "inspect then act"},),
            tools=(_tool(),),
            tool_choice="auto",
        )
    )

    assert turn.content == ""
    assert turn.reasoning_content == "inspect the current schema"
    assert [call.name for call in turn.tool_calls] == ["lookup"]
    assert turn.tool_calls[0].arguments == {"q": "exact api"}


def test_apply_source_edit_action_alias_is_one_decode_local_recovery(monkeypatch) -> None:
    posts = 0

    def post(url, *, json, timeout):
        nonlocal posts
        posts += 1
        return _CompletionResponse(
            status_code=200,
            payload={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<tool_call><function=apply_source_edit>"
                                "<parameter=action>replace_exact</parameter>"
                                "<parameter=path>src/main/java/Example.java</parameter>"
                                "</function></tool_call>"
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", post)

    turn = _adapter().generate_turn(
        GenerationRequest(
            messages=({"role": "user", "content": "apply one edit"},),
            tools=(_source_edit_tool(),),
            tool_choice={
                "type": "function",
                "function": {"name": "apply_source_edit"},
            },
        )
    )

    assert posts == 1
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].arguments == {
        "operation": "replace_exact",
        "path": "src/main/java/Example.java",
    }
    assert "action" not in turn.tool_calls[0].arguments


@pytest.mark.parametrize(
    "parameters",
    [
        (
            "<parameter=action>replace_exact</parameter>"
            "<parameter=operation>replace_exact</parameter>"
        ),
        (
            "<parameter=operation>replace_exact</parameter>"
            "<parameter=action>replace_exact</parameter>"
        ),
    ],
)
def test_apply_source_edit_rejects_canonical_alias_collision(
    monkeypatch,
    parameters: str,
) -> None:
    posts = 0

    def post(url, *, json, timeout):
        nonlocal posts
        posts += 1
        return _CompletionResponse(
            status_code=200,
            payload={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<tool_call><function=apply_source_edit>"
                                + parameters
                                + "<parameter=path>Example.java</parameter>"
                                "</function></tool_call>"
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ModelBackendError, match="conflicting sources.*parameter 'operation'"):
        _adapter().generate_turn(
            GenerationRequest(
                messages=({"role": "user", "content": "apply one edit"},),
                tools=(_source_edit_tool(),),
                tool_choice="auto",
            )
        )
    assert posts == 1


@pytest.mark.parametrize(
    ("tool_name", "include_action", "parameter", "expected"),
    [
        ("lookup", False, "action", "unknown parameter 'action'"),
        ("apply_source_edit", False, "strategy", "unknown parameter 'strategy'"),
    ],
)
def test_action_alias_does_not_weaken_other_schema_boundaries(
    monkeypatch,
    tool_name: str,
    include_action: bool,
    parameter: str,
    expected: str,
) -> None:
    schema = _source_edit_tool(include_action=include_action)
    schema["function"]["name"] = tool_name
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: _CompletionResponse(
            status_code=200,
            payload={
                "choices": [
                    {
                        "message": {
                            "content": (
                                f"<tool_call><function={tool_name}>"
                                f"<parameter={parameter}>replace_exact</parameter>"
                                "<parameter=path>Example.java</parameter>"
                                "</function></tool_call>"
                            )
                        }
                    }
                ]
            },
        ),
    )

    with pytest.raises(ModelBackendError, match=expected):
        _adapter().generate_turn(
            GenerationRequest(
                messages=({"role": "user", "content": "one tool"},),
                tools=(schema,),
                tool_choice="auto",
            )
        )


def test_repeated_reasoning_only_turn_fails_closed_after_one_continuation(monkeypatch) -> None:
    calls = 0
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")

    def post(url, *, json, timeout):
        nonlocal calls
        calls += 1
        return _CompletionResponse(
            status_code=200,
            payload={
                "choices": [{
                    "message": {
                        "content": "",
                        "reasoning_content": f"reasoning pass {calls}",
                    }
                }]
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ModelBackendError) as caught:
        _adapter().generate_turn(
            GenerationRequest(
                messages=({"role": "user", "content": "inspect then act"},),
                tools=(_tool(),),
                tool_choice="auto",
            )
        )

    assert calls == 2
    assert "reasoning-only tool continuation without a semantic action" in str(caught.value)


def test_fully_empty_native_turn_still_fails_immediately(monkeypatch) -> None:
    calls = 0
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")

    def post(url, *, json, timeout):
        nonlocal calls
        calls += 1
        return _CompletionResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": ""}}]},
        )

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ModelBackendError) as caught:
        _adapter().generate_turn(
            GenerationRequest(
                messages=({"role": "user", "content": "inspect then act"},),
                tools=(_tool(),),
                tool_choice="auto",
            )
        )

    assert calls == 1
    assert "neither visible content, reasoning, nor Qwen tool calls" in str(caught.value)


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


def test_generate_turn_keeps_detailed_schema_host_side(monkeypatch) -> None:
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
    for forbidden in ("response_format", "json_schema", "grammar"):
        assert forbidden not in captured["payload"]
