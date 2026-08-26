from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_server_hardware_policy as hardware
from minecraft_mod_ai.forced_tool_execution_contract import _single_tool_request
from minecraft_mod_ai.model_adapters.base import (
    GenerationRequest,
    GenerationResponse,
    ToolCall,
)
from minecraft_mod_ai.qwen_agent_family_contract import (
    _inject_reasoning_history,
    _remember_reasoning,
)

_TOOL = {
    "type": "function",
    "function": {
        "name": "read_project_file",
        "description": "Read one project file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

_SAMPLING = {
    "general_thinking": {
        "temperature": 0.31,
        "top_p": 0.71,
        "top_k": 17,
        "min_p": 0.03,
        "presence_penalty": 0.21,
        "repeat_penalty": 0.91,
    },
    "precise_coding": {
        "temperature": 0.23,
        "top_p": 0.67,
        "top_k": 13,
        "min_p": 0.02,
        "presence_penalty": 0.12,
        "repeat_penalty": 0.89,
    },
    "non_thinking": {
        "temperature": 0.11,
        "top_p": 0.61,
        "top_k": 7,
        "min_p": 0.01,
        "presence_penalty": 0.04,
        "repeat_penalty": 0.83,
    },
}


class _Adapter:
    def __init__(
        self,
        *,
        role: str = "coder_safe",
        enabled: bool = True,
        reasoning_effort: str = "",
        family: str = "qwen3.6",
    ) -> None:
        extra = {}
        if enabled:
            extra.update(
                {
                    "runtime_contract": "qwen",
                    "qwen_family": family,
                    "qwen_tool_markup": "qwen3_coder_xml",
                    "qwen_action_thinking_control": "enable_thinking_false",
                    "qwen_preserve_thinking": family in {"qwen3.6", "qwen3.8"},
                    "qwen_reasoning_effort": family == "qwen3.8",
                    "qwen_assistant_prefill": True,
                    "agent_thinking": True,
                    "sampling_profiles": _SAMPLING,
                }
            )
            if reasoning_effort:
                extra["thinking_reasoning_effort"] = reasoning_effort
        self.config = SimpleNamespace(
            model_id="vendor/arbitrary-runtime-model",
            role=role,
            max_new_tokens=2048,
            extra=extra,
        )


def _request(
    *,
    tools=(_TOOL,),
    tool_choice="auto",
    messages=({"role": "user", "content": "Implement the feature."},),
) -> GenerationRequest:
    return GenerationRequest(
        messages=messages,
        tools=tuple(tools),
        tool_choice=tool_choice,
        parallel_tool_calls=True,
    )


def test_registry_declared_auto_tool_action_disables_thinking() -> None:
    payload = hardware._server_payload(_Adapter(), _request())

    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == 0.11
    assert payload["top_p"] == 0.61
    assert payload["top_k"] == 7
    assert payload["min_p"] == 0.01
    assert payload["presence_penalty"] == 0.04
    assert payload["repeat_penalty"] == 0.83
    assert "repetition_penalty" not in payload


def test_registry_metadata_not_model_name_selects_agent_policy() -> None:
    enabled = hardware._server_payload(_Adapter(enabled=True), _request())
    disabled = hardware._server_payload(_Adapter(enabled=False), _request())

    assert enabled["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert "reasoning_effort" not in enabled
    assert enabled["temperature"] == 0.11
    assert disabled["chat_template_kwargs"] == {"enable_thinking": False}
    assert disabled["reasoning_effort"] == "none"


def test_qwen38_action_drops_planning_reasoning_effort() -> None:
    payload = hardware._server_payload(
        _Adapter(role="researcher", reasoning_effort="xhigh", family="qwen3.8"),
        _request(),
    )

    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == 0.11


def test_family_wrapper_preserves_existing_payload_contract_markers() -> None:
    assert getattr(hardware._server_payload, "_mmm_active_cache_reuse", False)
    assert not getattr(hardware._server_payload, "_mmm_qwen35_request_policy_v2", False)
    assert getattr(hardware._server_payload, "_mmm_qwen_family_agent_policy", False)


def test_family_wrapper_accepts_request_without_tools_attribute() -> None:
    request = SimpleNamespace(
        messages=({"role": "user", "content": "plain text"},),
        response_format="text",
    )
    payload = hardware._server_payload(_Adapter(enabled=False), request)

    assert payload["messages"] == [{"role": "user", "content": "plain text"}]
    assert payload["temperature"] == 0.0


def test_forced_return_function_stays_owned_by_transport_layer() -> None:
    payload = hardware._server_payload(
        _Adapter(),
        _request(
            tool_choice={
                "type": "function",
                "function": {"name": "read_project_file"},
            }
        ),
    )

    assert payload["temperature"] == 0.0
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert "reasoning_effort" not in payload


def test_required_single_tool_turn_does_not_restore_agent_thinking() -> None:
    original = _request(
        tool_choice={
            "type": "function",
            "function": {"name": "read_project_file"},
        }
    )
    local = _single_tool_request(original, "read_project_file")

    payload = hardware._server_payload(_Adapter(), local)

    assert local.tool_choice == "required"
    assert local.parallel_tool_calls is False
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert "reasoning_effort" not in payload


def test_final_agent_continuation_keeps_thinking_without_tools() -> None:
    messages = (
        {"role": "user", "content": "Implement the feature."},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I need to inspect the source first.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_project_file",
                        "arguments": '{"path":"src/main/java/A.java"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "read_project_file",
            "content": '{"ok":true}',
        },
    )
    payload = hardware._server_payload(
        _Adapter(role="researcher"),
        _request(tools=(), tool_choice=None, messages=messages),
    )

    assert payload["chat_template_kwargs"]["preserve_thinking"] is True
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert "reasoning_effort" not in payload


def test_reasoning_trace_is_restored_for_next_tool_turn() -> None:
    adapter = _Adapter()
    response = GenerationResponse(
        content="",
        reasoning_content="Inspect the exact Java API before editing.",
        tool_calls=(
            ToolCall(
                id="call_7",
                name="read_project_file",
                arguments={"path": "src/main/java/example/Hook.java"},
                raw_arguments='{"path":"src/main/java/example/Hook.java"}',
            ),
        ),
    )
    _remember_reasoning(adapter, response)

    continuation = _request(
        messages=(
            {"role": "user", "content": "Implement the feature."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_7",
                        "type": "function",
                        "function": {
                            "name": "read_project_file",
                            "arguments": '{"path":"src/main/java/example/Hook.java"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_7",
                "name": "read_project_file",
                "content": '{"ok":true}',
            },
        )
    )
    prepared = _inject_reasoning_history(adapter, continuation)

    assistant = prepared.messages[1]
    assert assistant["reasoning_content"] == "Inspect the exact Java API before editing."


def test_fresh_agent_request_does_not_leak_prior_reasoning() -> None:
    adapter = _Adapter()
    _remember_reasoning(
        adapter,
        GenerationResponse(
            content="old answer",
            reasoning_content="old private reasoning",
        ),
    )

    fresh = _inject_reasoning_history(adapter, _request())

    assert fresh.messages == _request().messages
    assert all(
        not str(message.get("reasoning_content") or "").strip()
        for message in fresh.messages
    )
