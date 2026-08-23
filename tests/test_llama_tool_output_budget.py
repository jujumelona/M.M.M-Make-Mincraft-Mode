from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_length_resilience
from minecraft_mod_ai import llama_server_hardware_policy as hardware_policy
from minecraft_mod_ai.llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    LlamaCompletionBoundaryError,
)


def _tool(name: str = "apply_source_edit") -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _adapter() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            role="coder",
            adapter="llama_cpp",
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            extra={
                "gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
                "runtime_contract": "qwen",
                "qwen_family": "qwen3.5",
                "qwen_tool_markup": "qwen3_coder_xml",
                "qwen_action_thinking_control": "enable_thinking_false",
                "qwen_preserve_thinking": False,
                "qwen_reasoning_effort": False,
                "qwen_assistant_prefill": True,
                "decode_hotpath": "t4_mtp",
            },
            max_new_tokens=-1,
        )
    )


def _request(tool_name: str, *, tool_choice: object = "auto") -> SimpleNamespace:
    return SimpleNamespace(
        messages=({"role": "user", "content": "repair the source"},),
        response_format="text",
        response_schema=None,
        tools=(_tool(tool_name),),
        tool_choice=tool_choice,
        parallel_tool_calls=False,
    )


def test_read_tool_turn_keeps_native_runtime_output_allowance() -> None:
    payload = hardware_policy._server_payload(_adapter(), _request("search_code_rag"))

    assert payload["max_tokens"] == -1
    assert payload["tool_choice"] == "auto"


def test_source_mutation_turn_keeps_native_runtime_output_allowance() -> None:
    payload = hardware_policy._server_payload(_adapter(), _request("apply_source_edit"))

    assert payload["max_tokens"] == -1
    assert payload["tool_choice"] == "auto"


def test_named_required_action_uses_required_wire_contract_without_token_cap() -> None:
    choice = {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }
    payload = hardware_policy._server_payload(
        _adapter(),
        _request("apply_source_edit", tool_choice=choice),
    )

    assert payload["max_tokens"] == -1
    assert payload["tool_choice"] == "required"
    assert payload["temperature"] == 0.0
    assert len(payload["tools"]) == 1


def test_length_recovery_preserves_existing_runtime_output_allowance(monkeypatch) -> None:
    attempts: list[dict[str, object]] = []

    def completion(_server_url: str, payload: dict[str, object]):
        attempts.append(dict(payload))
        if len(attempts) == 1:
            raise LlamaCompletionBoundaryError(
                "synthetic context pressure",
                kind=CONTEXT_PRESSURE,
            )
        return {"content": "ok"}

    fake = SimpleNamespace(_completion_message=completion)
    monkeypatch.setattr(
        llama_length_resilience,
        "emergency_fit_messages",
        lambda _messages, *, budget_bytes: ({"role": "user", "content": "compact"},),
    )
    llama_length_resilience.install(fake)

    result = fake._completion_message(
        "http://127.0.0.1:8910/v1",
        {
            "messages": [
                {"role": "user", "content": "x" * 50_000},
                {"role": "tool", "content": "y" * 10_000},
            ],
            "max_tokens": -1,
            "tools": [_tool("search_code_rag")],
        },
    )

    assert result == {"content": "ok"}
    assert len(attempts) == 2
    assert attempts[1]["max_tokens"] == -1
    assert len(str(attempts[1]["messages"])) < len(str(attempts[0]["messages"]))
