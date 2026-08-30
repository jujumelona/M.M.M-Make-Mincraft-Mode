from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import model_context_budget
from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.llama_context_safety_contract import (
    ContextPackingError,
    _protocol_safe_minimal_fit,
)
from minecraft_mod_ai.model_adapters import AdapterConfig
from minecraft_mod_ai.source_mutation_contract import mutation_history_applied


def _config(*, runtime_context: int = 32_768) -> AdapterConfig:
    return AdapterConfig(
        role="coder",
        adapter="llama_cpp",
        model_id="test/qwen",
        max_context=262_144,
        max_new_tokens=8_192,
        extra={"runtime_context_default": runtime_context},
    )


def _tool_schema() -> tuple[dict[str, object], ...]:
    return (
        {
            "type": "function",
            "function": {
                "name": "apply_source_patch",
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    )


def _assistant(call_id: str, name: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _tool(call_id: str, name: str, payload: object) -> dict[str, object]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(payload, sort_keys=True),
    }


def test_hard_capacity_never_reinflates_tiny_runtime_to_12k_floor() -> None:
    with pytest.raises(ContextPackingError, match="consume the active llama context"):
        model_context_budget.request_message_budget(
            _config(runtime_context=4096),
            _tool_schema(),
        )


def test_incompressible_authored_task_fails_before_oversized_inference() -> None:
    messages = (
        {"role": "system", "content": "authority"},
        {"role": "user", "content": "TASK:" + ("x" * 100_000)},
    )
    with pytest.raises(ContextPackingError, match="Refusing silent task/protocol truncation"):
        model_context_budget.fit_messages_to_context(
            messages,
            config=_config(),
            tools=_tool_schema(),
        )


def test_protocol_minimal_fit_keeps_all_leading_authority_task_and_mutation() -> None:
    messages = (
        {"role": "system", "content": "system-a"},
        {"role": "system", "content": "system-b"},
        {"role": "user", "content": "original authored task"},
        _assistant("old", "search_code_rag"),
        _tool("old", "search_code_rag", {"ok": True, "result": "old" * 2000}),
        _assistant("mutation", "apply_source_patch"),
        _tool(
            "mutation",
            "apply_source_patch",
            {
                "ok": True,
                "_mmm_source_mutation": {
                    "tool": "apply_source_patch",
                    "status": "APPLIED_BY_HOST_RUNTIME",
                },
                "result": {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "before_sha256": "sha256:before",
                            "after_sha256": "sha256:after",
                        }
                    ],
                },
            },
        ),
        _assistant("recent", "search_code_rag"),
        _tool("recent", "search_code_rag", {"ok": True, "result": "recent"}),
    )

    fitted = _protocol_safe_minimal_fit(
        model_context_budget,
        messages,
        budget=8 * 1024,
    )

    assert [item["content"] for item in fitted[:3]] == [
        "system-a",
        "system-b",
        "original authored task",
    ]
    assert mutation_history_applied(fitted)
    call_ids = {
        call["id"]
        for item in fitted
        if item.get("role") == "assistant"
        for call in item.get("tool_calls", ())
    }
    result_ids = {
        item["tool_call_id"]
        for item in fitted
        if item.get("role") == "tool"
    }
    assert call_ids == result_ids
    assert "recent" in call_ids


def test_emergency_context_pressure_never_retries_identical_history() -> None:
    messages = (
        {"role": "system", "content": "system-a"},
        {"role": "system", "content": "system-b"},
        {"role": "user", "content": "original task"},
        _assistant("old", "search_code_rag"),
        _tool("old", "search_code_rag", {"ok": True, "result": "old evidence"}),
        _assistant("recent", "search_code_rag"),
        _tool("recent", "search_code_rag", {"ok": True, "result": "recent evidence"}),
    )
    original_size = model_context_budget._canonical_size(messages)

    fitted = model_context_budget.emergency_fit_messages(
        messages,
        budget_bytes=40 * 1024,
    )

    assert fitted != messages
    assert model_context_budget._canonical_size(fitted) < original_size
    assert any(item.get("content") == "original task" for item in fitted)
    assert any(item.get("tool_call_id") == "recent" for item in fitted)


def test_ultra_context_fallback_never_silently_drops_authored_task() -> None:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "system-a"},
        {"role": "system", "content": "system-b"},
        {"role": "user", "content": "original authored task"},
        _assistant("old", "search_code_rag"),
        _tool("old", "search_code_rag", {"ok": True, "result": "old"}),
        _assistant("recent", "search_code_rag"),
        _tool("recent", "search_code_rag", {"ok": True, "result": "recent"}),
    ]
    before = [dict(message) for message in messages]
    legacy_three = (messages[0], messages[-2], messages[-1])

    with pytest.raises(ContextPackingError, match="Refusing silent task/protocol truncation"):
        tool_loop._replace_live_messages(messages, legacy_three)

    assert messages == before
    assert any(item.get("content") == "original authored task" for item in messages)


def test_runtime_context_safety_wrappers_are_installed() -> None:
    assert getattr(
        model_context_budget.request_message_budget,
        "_mmm_hard_context_capacity_v1",
        False,
    )
    assert getattr(
        model_context_budget.fit_messages_to_context,
        "_mmm_hard_context_fit_v1",
        False,
    )
    assert getattr(
        model_context_budget.emergency_fit_messages,
        "_mmm_protocol_safe_emergency_fit_v1",
        False,
    )
    assert getattr(
        tool_loop._replace_live_messages,
        "_mmm_protocol_safe_live_replace_v1",
        False,
    )
