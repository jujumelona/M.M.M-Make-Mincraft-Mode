from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.model_context_budget import request_message_budget


def _tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_qwen35_prompt_budget_uses_runtime_slot_and_reserves_tool_decode(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TOOL_MAX_TOKENS", raising=False)
    config = SimpleNamespace(
        adapter="llama_cpp",
        max_context=262144,
        max_new_tokens=8192,
        extra={
            "runtime_contract": "qwen",
            "decode_hotpath": "t4_mtp",
            "runtime_context_default": 32768,
        },
    )

    plain_budget = request_message_budget(config)
    tool_budget = request_message_budget(config, (_tool(),))

    assert plain_budget < 64 * 1024
    assert 40 * 1024 < tool_budget < 48 * 1024
    assert tool_budget < plain_budget


def test_runtime_context_override_is_shared_with_prompt_fitting(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "24576")
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TOOL_MAX_TOKENS", raising=False)
    config = SimpleNamespace(
        adapter="llama_cpp",
        max_context=262144,
        max_new_tokens=8192,
        extra={
            "runtime_contract": "qwen",
            "decode_hotpath": "t4_mtp",
            "runtime_context_default": 32768,
        },
    )

    budget = request_message_budget(config, (_tool(),))

    assert 24 * 1024 < budget < 32 * 1024
