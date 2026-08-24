from __future__ import annotations

"""Model-free regression checks for context-window resilience."""

import json
import os
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
    from . import llama_server_hardware_policy, model_context_budget
    from .model_context_budget import (
        fit_messages_to_context,
        request_message_budget,
        tool_action_token_budget,
    )

    if (
        getattr(
            llama_server_hardware_policy._server_payload,
            "_mmm_finite_generation_budget_v1",
            False,
        )
        is not True
    ):
        raise ContextBudgetPreflightError(
            "llama-server payload is missing the finite generation budget policy"
        )

    synthetic_adapter = SimpleNamespace(
        config=SimpleNamespace(max_new_tokens=8192, model_id="synthetic", extra={})
    )
    synthetic_payload = llama_server_hardware_policy._server_payload(
        synthetic_adapter,
        SimpleNamespace(
            messages=({"role": "user", "content": "test"},),
            tools=(),
            tool_choice=None,
            parallel_tool_calls=False,
            response_format="text",
        ),
    )
    if synthetic_payload.get("max_tokens") != 8192:
        raise ContextBudgetPreflightError(
            "llama-server non-tool production payload is not using the finite text budget"
        )

    tool_schema = {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "path": {"type": "string"},
                },
            },
        },
    }
    synthetic_tool_payload = llama_server_hardware_policy._server_payload(
        synthetic_adapter,
        SimpleNamespace(
            messages=({"role": "user", "content": "edit"},),
            tools=(tool_schema,),
            tool_choice="required",
            parallel_tool_calls=False,
            response_format="json",
        ),
    )
    if synthetic_tool_payload.get("max_tokens") != tool_action_token_budget(
        synthetic_adapter.config
    ):
        raise ContextBudgetPreflightError(
            "tool turns must preserve the bounded tool-action budget"
        )
    if synthetic_tool_payload.get("tool_choice") != "required":
        raise ContextBudgetPreflightError(
            "required tool payload did not preserve native llama-server tool semantics"
        )
    if synthetic_tool_payload.get("parallel_tool_calls") is not False:
        raise ContextBudgetPreflightError(
            "single required tool payload unexpectedly enabled parallel tool calls"
        )
    if [item.get("function", {}).get("name") for item in synthetic_tool_payload.get("tools", [])] != [
        "apply_source_edit"
    ]:
        raise ContextBudgetPreflightError(
            "required tool payload did not expose exactly the selected function schema"
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
    tools = (tool_schema,)

    previous_qwen_ctx = os.environ.pop("MMM_QWEN35_MTP_CTX", None)
    previous_server_ctx = os.environ.pop("MMM_LLAMA_SERVER_CTX", None)
    try:
        budget = request_message_budget(config, tools)
    finally:
        if previous_qwen_ctx is not None:
            os.environ["MMM_QWEN35_MTP_CTX"] = previous_qwen_ctx
        if previous_server_ctx is not None:
            os.environ["MMM_LLAMA_SERVER_CTX"] = previous_server_ctx

    if budget >= 64 * 1024:
        raise ContextBudgetPreflightError(
            "llama input budget is using model capacity instead of the 32K runtime slot"
        )
    if _encoded_size(messages) <= budget:
        raise ContextBudgetPreflightError(
            "synthetic first-tool-round fixture no longer exceeds its context budget"
        )

    original_archive = model_context_budget.archive_transcript
    model_context_budget.archive_transcript = lambda values: {
        "available": True,
        "sha256": "sha256:" + "0" * 64,
        "bytes": _encoded_size(values),
        "path": "/synthetic/context.json",
        "format": "canonical-json",
    }
    try:
        fitted = fit_messages_to_context(messages, config=config, tools=tools)
    finally:
        model_context_budget.archive_transcript = original_archive

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
