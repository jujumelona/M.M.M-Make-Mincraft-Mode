from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
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


class _Adapter:
    def __init__(self, model_id: str) -> None:
        self.config = SimpleNamespace(model_id=model_id, max_new_tokens=8192)


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


def test_qwen36_auto_tool_loop_enables_thinking_preservation() -> None:
    payload = _server_payload(
        _Adapter("unsloth/Qwen3.6-27B-MTP-GGUF"),
        _request(),
    )

    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 0.0
    assert payload["repetition_penalty"] == 1.0


def test_qwen35_keeps_non_thinking_tool_agent_policy() -> None:
    payload = _server_payload(
        _Adapter("unsloth/Qwen3.5-9B-MTP-GGUF"),
        _request(),
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "none"
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 1.5
    assert payload["repetition_penalty"] == 1.0
    assert "preserve_thinking" not in payload["chat_template_kwargs"]


def test_qwen36_forced_return_function_stays_non_thinking() -> None:
    payload = _server_payload(
        _Adapter("unsloth/Qwen3.6-27B-MTP-GGUF"),
        _request(
            tool_choice={
                "type": "function",
                "function": {"name": "read_project_file"},
            }
        ),
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "none"
    assert payload["temperature"] == 0.0
    assert "preserve_thinking" not in payload["chat_template_kwargs"]


def test_qwen36_final_agent_continuation_keeps_thinking_without_tools() -> None:
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
    payload = _server_payload(
        _Adapter("unsloth/Qwen3.6-27B-MTP-GGUF"),
        _request(tools=(), tool_choice=None, messages=messages),
    )

    assert payload["chat_template_kwargs"]["preserve_thinking"] is True
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert "reasoning_effort" not in payload


def test_qwen36_reasoning_trace_is_restored_for_next_tool_turn() -> None:
    adapter = _Adapter("unsloth/Qwen3.6-27B-MTP-GGUF")
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


def test_fresh_qwen36_agent_request_does_not_leak_prior_reasoning() -> None:
    adapter = _Adapter("unsloth/Qwen3.6-27B-MTP-GGUF")
    _remember_reasoning(
        adapter,
        GenerationResponse(
            content="old answer",
            reasoning_content="old private reasoning",
        ),
    )

    fresh = _inject_reasoning_history(adapter, _request())

    assert fresh.messages == _request().messages
    assert not getattr(adapter, "_mmm_qwen36_reasoning_traces")
