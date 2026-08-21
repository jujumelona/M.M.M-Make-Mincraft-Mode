from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_length_resilience
from minecraft_mod_ai import llama_server_hardware_policy as hardware_policy
from minecraft_mod_ai.llama_tool_output_budget import tool_output_budget


def _tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "edit source",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_fully_composed_qwen_tool_turn_uses_configured_coder_budget(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MMM_QWEN35_MAX_OUTPUT_TOKENS", "-1")
    monkeypatch.delenv("MMM_LLAMA_TOOL_MAX_TOKENS", raising=False)
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            extra={
                "gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
                "runtime_contract": "qwen",
                "decode_hotpath": "t4_mtp",
            },
            max_new_tokens=8192,
        )
    )
    request = SimpleNamespace(
        messages=({"role": "user", "content": "repair the source"},),
        response_format="json",
        response_schema=None,
        tools=(_tool(),),
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    payload = hardware_policy._server_payload(adapter, request)

    assert tool_output_budget(adapter.config) == 8192
    assert payload["max_tokens"] == 8192


def test_tool_output_override_remains_hard_bounded(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_TOOL_MAX_TOKENS", "999999")
    config = SimpleNamespace(max_new_tokens=32768)

    assert tool_output_budget(config) == 16384


def test_length_recovery_preserves_bounded_tool_output_budget(monkeypatch) -> None:
    attempts: list[dict[str, object]] = []

    def completion(_server_url: str, payload: dict[str, object]):
        attempts.append(dict(payload))
        if len(attempts) == 1:
            raise RuntimeError(
                "native llama-server reached its model/server context boundary before "
                "the assistant turn completed"
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
            "max_tokens": 8192,
            "tools": [_tool()],
        },
    )

    assert result == {"content": "ok"}
    assert len(attempts) == 2
    assert attempts[1]["max_tokens"] == 8192
    assert len(str(attempts[1]["messages"])) < len(str(attempts[0]["messages"]))
