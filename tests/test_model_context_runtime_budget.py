from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.llama_tool_output_budget import tool_output_budget
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
    available_input_tokens = max(
        2048,
        runtime_tokens - tool_output_budget(config) - _CONTEXT_TOKEN_GUARD,
    )
    return (
        available_input_tokens * _BYTES_PER_TOKEN_BUDGET
        - len(_canonical_bytes((tool,)))
    )


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
    tool = _tool()

    runtime_tokens = _effective_context_tokens(config)
    plain_budget = request_message_budget(config)
    tool_budget = request_message_budget(config, (tool,))

    assert runtime_tokens == 32768
    assert plain_budget == (
        runtime_tokens - _CONTEXT_TOKEN_GUARD
    ) * _BYTES_PER_TOKEN_BUDGET
    assert tool_budget == _expected_tool_budget(config, tool)
    assert plain_budget - tool_budget == (
        tool_output_budget(config) * _BYTES_PER_TOKEN_BUDGET
        + len(_canonical_bytes((tool,)))
    )


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
    tool = _tool()

    runtime_tokens = _effective_context_tokens(config)
    plain_budget = request_message_budget(config)
    tool_budget = request_message_budget(config, (tool,))

    assert runtime_tokens == 24576
    assert plain_budget == (
        runtime_tokens - _CONTEXT_TOKEN_GUARD
    ) * _BYTES_PER_TOKEN_BUDGET
    assert tool_budget == _expected_tool_budget(config, tool)
