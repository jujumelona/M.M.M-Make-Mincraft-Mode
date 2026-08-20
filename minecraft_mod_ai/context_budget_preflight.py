from __future__ import annotations

"""Model-free regression checks for context-window resilience."""

import json
from types import SimpleNamespace
from typing import Any


class ContextBudgetPreflightError(RuntimeError):
    pass


def _encoded_size(messages: Any) -> int:
    return len(
        json.dumps(
            list(messages),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def run_context_budget_preflight() -> None:
    from . import llama_server_hardware_policy
    from . import small_model_context_compaction as archive_module
    from .model_adapters import llama_cpp_adapter
    from .model_context_budget import fit_messages_to_context, request_message_budget

    if (
        getattr(
            llama_server_hardware_policy._server_payload,
            "_mmm_unbounded_llama_completion_v1",
            False,
        )
        is not True
    ):
        raise ContextBudgetPreflightError(
            "llama-server payload is missing the native unbounded completion policy"
        )
    synthetic_payload = llama_server_hardware_policy._server_payload(
        SimpleNamespace(
            config=SimpleNamespace(max_new_tokens=8192, model_id="synthetic")
        ),
        SimpleNamespace(
            messages=({"role": "user", "content": "test"},),
            tools=(),
            response_format="text",
        ),
    )
    if synthetic_payload.get("max_tokens") != -1:
        raise ContextBudgetPreflightError(
            "llama-server production payload is still capped by max_new_tokens"
        )

    if (
        getattr(
            llama_cpp_adapter._completion_message,
            "_mmm_bounded_length_recovery_v1",
            False,
        )
        is not True
    ):
        raise ContextBudgetPreflightError(
            "llama completion path is missing bounded finish_reason=length recovery"
        )

    large_result = json.dumps(
        {
            "ok": True,
            "tool": "search_code_rag",
            "result": {
                "receipt": {
                    "result_count": 8,
                    "coverage_score": 1.0,
                    "relevance_score": 1.0,
                },
                "preview": "e" * (46 * 1024),
            },
        },
        separators=(",", ":"),
    )
    messages = (
        {"role": "system", "content": "coder"},
        {"role": "user", "content": "task\n" + "r" * (20 * 1024)},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "rag",
                    "type": "function",
                    "function": {"name": "search_code_rag", "arguments": "{}"},
                },
                {
                    "id": "symbols",
                    "type": "function",
                    "function": {"name": "java_workspace_symbols", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "rag",
            "name": "search_code_rag",
            "content": large_result,
        },
        {
            "role": "tool",
            "tool_call_id": "symbols",
            "name": "java_workspace_symbols",
            "content": large_result.replace("search_code_rag", "java_workspace_symbols", 1),
        },
    )
    config = SimpleNamespace(max_context=32768, max_new_tokens=8192)
    tools = (
        {"type": "function", "function": {"name": "apply_source_edit"}},
    )
    budget = request_message_budget(config, tools)
    if _encoded_size(messages) <= budget:
        raise ContextBudgetPreflightError(
            "synthetic first-tool-round fixture no longer exceeds its context budget"
        )

    original_archive = archive_module._archive_transcript
    archive_module._archive_transcript = lambda values: {
        "available": True,
        "sha256": "sha256:" + "0" * 64,
        "bytes": _encoded_size(values),
        "path": "/synthetic/context.json",
        "format": "canonical-json",
    }
    try:
        fitted = fit_messages_to_context(messages, config=config, tools=tools)
    finally:
        archive_module._archive_transcript = original_archive

    if _encoded_size(fitted) > budget:
        raise ContextBudgetPreflightError(
            "first assistant/tool exchange still exceeds the model message budget"
        )
    tool_messages = [
        item for item in fitted if str(item.get("role", "")) == "tool"
    ]
    if len(tool_messages) != 2:
        raise ContextBudgetPreflightError("context fitting broke tool protocol cardinality")
    if not all(
        "_mmm_context_compaction" in str(item.get("content", ""))
        for item in tool_messages
    ):
        raise ContextBudgetPreflightError(
            "large first-round tool observations were not replaced by recoverable summaries"
        )


__all__ = ["ContextBudgetPreflightError", "run_context_budget_preflight"]
