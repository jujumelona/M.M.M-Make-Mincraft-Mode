from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_length_resilience
from minecraft_mod_ai import llama_server_hardware_policy as hardware_policy
from minecraft_mod_ai.llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    LlamaCompletionBoundaryError,
)
from minecraft_mod_ai.llama_tool_output_budget import tool_output_budget


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


def _request(tool_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        messages=({"role": "user", "content": "repair the source"},),
        response_format="json",
        response_schema=None,
        tools=(_tool(tool_name),),
        tool_choice="auto",
        parallel_tool_calls=False,
    )


def test_fully_composed_qwen_tool_turn_uses_native_unlimited_output(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_MAX_OUTPUT_TOKENS", "-1")
    monkeypatch.delenv("MMM_LLAMA_TOOL_MAX_TOKENS", raising=False)
    adapter = _adapter()

    payload = hardware_policy._server_payload(adapter, _request("search_code_rag"))

    assert tool_output_budget(adapter.config) == 4096
    assert payload["max_tokens"] == -1


def test_payload_heavy_source_mutation_keeps_model_runtime_budget(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_MAX_OUTPUT_TOKENS", "-1")
    monkeypatch.delenv("MMM_LLAMA_TOOL_MAX_TOKENS", raising=False)
    adapter = _adapter()

    payload = hardware_policy._server_payload(adapter, _request("apply_source_edit"))

    # The compact 4K policy must never become a second owner for a source payload.
    assert payload["max_tokens"] == -1


def test_tool_output_override_remains_hard_bounded(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_TOOL_MAX_TOKENS", "999999")
    config = SimpleNamespace(max_new_tokens=32768)

    assert tool_output_budget(config) == 16384


def test_length_recovery_preserves_existing_tool_output_budget(monkeypatch) -> None:
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
            "max_tokens": 4096,
            "tools": [_tool("search_code_rag")],
        },
    )

    assert result == {"content": "ok"}
    assert len(attempts) == 2
    assert attempts[1]["max_tokens"] == 4096
    assert len(str(attempts[1]["messages"])) < len(str(attempts[0]["messages"]))
