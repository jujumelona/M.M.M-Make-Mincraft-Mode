from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.model_context_budget as context_budget
from minecraft_mod_ai import small_model_context_compaction
from minecraft_mod_ai.model_context_budget import (
    _BYTES_PER_TOKEN_BUDGET,
    _CONTEXT_TOKEN_GUARD,
    _canonical_bytes,
    _effective_context_tokens,
    request_message_budget,
)


def _tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _expected_tool_budget(config: SimpleNamespace, tool: dict[str, object]) -> int:
    runtime_tokens = _effective_context_tokens(config)
    plain = (runtime_tokens - _CONTEXT_TOKEN_GUARD) * _BYTES_PER_TOKEN_BUDGET
    return plain - len(_canonical_bytes((tool,)))


def _config(*, max_new_tokens: int = 8192) -> SimpleNamespace:
    return SimpleNamespace(
        adapter="llama_cpp",
        max_context=262144,
        max_new_tokens=max_new_tokens,
        extra={
            "runtime_contract": "qwen",
            "decode_hotpath": "t4_mtp",
            "runtime_context_default": 32768,
        },
    )


def test_qwen35_prompt_budget_uses_runtime_slot_without_arbitrary_tool_decode_reserve(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_BYTES", str(512 * 1024))
    config = _config()
    tool = _tool()

    runtime_tokens = _effective_context_tokens(config)
    plain_budget = request_message_budget(config)
    tool_budget = request_message_budget(config, (tool,))

    assert runtime_tokens == 32768
    assert plain_budget == (
        runtime_tokens - _CONTEXT_TOKEN_GUARD
    ) * _BYTES_PER_TOKEN_BUDGET
    assert tool_budget == _expected_tool_budget(config, tool)
    assert plain_budget - tool_budget == len(_canonical_bytes((tool,)))


def test_runtime_context_override_is_shared_with_prompt_fitting(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "24576")
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_BYTES", str(512 * 1024))
    config = _config()
    tool = _tool()

    runtime_tokens = _effective_context_tokens(config)
    plain_budget = request_message_budget(config)
    tool_budget = request_message_budget(config, (tool,))

    assert runtime_tokens == 24576
    assert plain_budget == (
        runtime_tokens - _CONTEXT_TOKEN_GUARD
    ) * _BYTES_PER_TOKEN_BUDGET
    assert tool_budget == _expected_tool_budget(config, tool)


def test_llama_tool_prompt_budget_does_not_change_with_decode_allowance(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_BYTES", str(512 * 1024))
    tool = _tool()

    assert request_message_budget(_config(max_new_tokens=1024), (tool,)) == request_message_budget(
        _config(max_new_tokens=65536),
        (tool,),
    )


def test_tool_compaction_avoids_repeated_full_history_serialization(monkeypatch) -> None:
    messages = tuple(
        {
            "role": "tool",
            "name": f"tool-{index}",
            "content": '{"result":{"payload":"' + ("x" * 12000) + '"}}',
        }
        for index in range(6)
    )
    monkeypatch.setattr(
        small_model_context_compaction,
        "_archive_transcript",
        lambda _messages: {"available": True, "path": "memory://test"},
    )
    full_history_scans = 0
    original_canonical_bytes = context_budget._canonical_bytes

    def counting_canonical_bytes(value):
        nonlocal full_history_scans
        if isinstance(value, list) and len(value) == len(messages):
            full_history_scans += 1
        return original_canonical_bytes(value)

    monkeypatch.setattr(context_budget, "_canonical_bytes", counting_canonical_bytes)

    compacted = context_budget._compact_tool_messages(
        messages,
        budget=40 * 1024,
        preview_bytes=2048,
    )

    assert compacted != messages
    assert full_history_scans == 1
