from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import game_design
from minecraft_mod_ai.llama_server_hardware_policy import (
    _server_payload,
    _stream_delta_parts,
)


def _adapter(
    max_new_tokens: int = 8192,
    model_id: str = "generic/model",
    *,
    qwen_family: str = "",
):
    extra = {}
    if qwen_family:
        extra = {
            "runtime_contract": "qwen",
            "qwen_family": qwen_family,
            "qwen_tool_markup": "qwen3_coder_xml",
            "qwen_action_thinking_control": "enable_thinking_false",
            "qwen_preserve_thinking": qwen_family in {"qwen3.6", "qwen3.8"},
            "qwen_reasoning_effort": qwen_family == "qwen3.8",
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
        }
    return SimpleNamespace(
        config=SimpleNamespace(
            max_new_tokens=max_new_tokens,
            model_id=model_id,
            extra=extra,
        )
    )


def test_json_request_uses_server_schema_and_keeps_host_validation() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    request = SimpleNamespace(
        messages=(
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Plan it."},
        ),
        response_format="json",
        response_schema=schema,
        tools=(),
    )
    payload = _server_payload(_adapter(), request)
    assert request.response_schema == schema
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "mmm_host_action_arguments",
            "strict": True,
            "schema": schema,
        },
    }
    assert "grammar" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "parallel_tool_calls" not in payload
    assert payload["max_tokens"] == 4096


def test_native_tool_request_keeps_tools_visible_with_structured_action_page() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "lookup evidence",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }
    request = SimpleNamespace(
        messages=({"role": "user", "content": "inspect then plan"},),
        response_format="json",
        response_schema=schema,
        tools=(tool,),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    payload = _server_payload(
        _adapter(
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            qwen_family="qwen3.5",
        ),
        request,
    )

    assert payload["tools"] == [tool]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "mmm_host_action_arguments",
            "strict": True,
            "schema": schema,
        },
    }
    assert "grammar" not in payload
    assert "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 1.5
    assert payload["repeat_penalty"] == 1.0
    assert payload["max_tokens"] == 4096


def test_text_request_does_not_force_reasoning_policy() -> None:
    request = SimpleNamespace(
        messages=({"role": "user", "content": "hello"},),
        response_format="text",
    )
    payload = _server_payload(_adapter(64), request)
    assert "response_format" not in payload
    assert "reasoning_effort" not in payload


def test_reasoning_delta_is_progress_but_not_visible_content() -> None:
    reasoning, content = _stream_delta_parts(
        {"delta": {"reasoning_content": "hidden reasoning"}}
    )
    assert reasoning == "hidden reasoning"
    assert content == ""


def test_content_delta_is_visible_content() -> None:
    reasoning, content = _stream_delta_parts(
        {"delta": {"content": '{"ok":true}'}}
    )
    assert reasoning == ""
    assert content == '{"ok":true}'


def test_legacy_text_choice_is_still_supported() -> None:
    reasoning, content = _stream_delta_parts({"text": "legacy"})
    assert reasoning == ""
    assert content == "legacy"


def test_planner_page_budget_uses_selected_native_model_context() -> None:
    config = SimpleNamespace(max_context=32768, max_new_tokens=8192)

    class Registry:
        @staticmethod
        def role(profile, role):
            assert profile == "Qwen3.5-9B_6GB"
            assert role == "planner"
            return config

    router = SimpleNamespace(profile="Qwen3.5-9B_6GB", registry=Registry())
    budget = game_design._request_page_bytes(router)
    assert budget == 64 * 1024
