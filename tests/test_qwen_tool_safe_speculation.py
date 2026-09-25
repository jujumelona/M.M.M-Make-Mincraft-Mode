from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import llama_server_hardware_policy as hardware_policy
from minecraft_mod_ai.model_adapters.base import GenerationRequest

_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_source_edit",
        "description": "Apply one semantic source edit.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["operation", "path"],
        },
    },
}


def _config():
    return SimpleNamespace(
        model_id="vendor/qwen-runtime",
        role="coder",
        max_context=32768,
        max_new_tokens=8192,
        extra={
            "runtime_contract": "qwen",
            "qwen_family": "qwen3.5",
            "qwen_tool_markup": "qwen3_coder_xml",
            "qwen_action_thinking_control": "enable_thinking_false",
            "qwen_preserve_thinking": False,
            "qwen_reasoning_effort": False,
            "qwen_assistant_prefill": True,
            "agent_thinking": True,
            "sampling_profiles": {
                "non_thinking": {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 1.5,
                    "repeat_penalty": 1.0,
                },
                "precise_coding": {"temperature": 0.6},
                "general_thinking": {"temperature": 1.0},
            },
        },
    )


def _request(*, tools=(_TOOL,)):
    return GenerationRequest(
        messages=({"role": "user", "content": "Implement the requested edit."},),
        tools=tuple(tools),
        tool_choice="required" if tools else None,
        parallel_tool_calls=False,
    )


def test_tool_page_safety_is_direct_payload_policy_without_runtime_restart(monkeypatch) -> None:
    monkeypatch.setattr(
        autotune,
        "_launch_selected",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("payload construction must not restart llama-server")
        ),
    )

    request = _request()
    payload = hardware_policy._server_payload(
        SimpleNamespace(config=_config()),
        request,
    )

    assert payload["tools"] == [_TOOL]
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20


def test_plain_coder_page_keeps_family_thinking_policy_without_server_mutation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        autotune,
        "_launch_selected",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("request policy must not mutate server ownership")
        ),
    )

    payload = hardware_policy._server_payload(
        SimpleNamespace(config=_config()),
        _request(tools=()),
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["temperature"] == 0.6
    assert "tools" not in payload
