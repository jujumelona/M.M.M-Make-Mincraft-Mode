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
    from . import llama_server_hardware_policy
    from . import small_model_context_compaction as archive_module
    from .llama_tool_output_budget import tool_output_budget
    from .model_adapters import llama_cpp_adapter
    from .model_context_budget import fit_messages_to_context, request_message_budget

    if (
        getattr(
            llama_server_hardware_policy._server_payload,
            "_mmm_unbounded_llama_completion_v2",
            False,
        )
        is not True
    ):
        raise ContextBudgetPreflightError(
            "llama-server payload is missing the native unbounded completion policy"
        )
    if (
        getattr(
            llama_server_hardware_policy._server_payload,
            "_mmm_llama_tool_output_budget_v2",
            False,
        )
        is not True
    ):
        raise ContextBudgetPreflightError(
            "llama-server payload is missing the bounded native tool-output policy"
        )

    synthetic_adapter = SimpleNamespace(
        config=SimpleNamespace(max_new_tokens=8192, model_id="synthetic")
    )
    synthetic_payload = llama_server_hardware_policy._server_payload(
        synthetic_adapter,
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

    synthetic_tool_payload = llama_server_hardware_policy._server_payload(
        synthetic_adapter,
        SimpleNamespace(
            messages=({"role": "user", "content": "search"},),
            tools=({"type": "function", "function": {"name": "search_code_rag"}},),
            response_format="json",
        ),
    )
    expected_tool_budget = tool_output_budget(synthetic_adapter.config)
    if synthetic_tool_payload.get("max_tokens") != expected_tool_budget:
        raise ContextBudgetPreflightError(
            "llama-server compact tool payload disagrees with the authoritative bounded "
            f"tool-output policy: expected {expected_tool_budget}"
        )
    if expected_tool_budget >= synthetic_adapter.config.max_new_tokens:
        raise ContextBudgetPreflightError(
            "default scalar tool-output budget must remain below full coder generation budget"
        )

    synthetic_mutation_payload = llama_server_hardware_policy._server_payload(
        synthetic_adapter,
        SimpleNamespace(
            messages=({"role": "user", "content": "edit"},),
            tools=({"type": "function", "function": {"name": "apply_source_edit"}},),
            response_format="json",
        ),
    )
    if synthetic_mutation_payload.get("max_tokens") != synthetic_payload.get("max_tokens"):
        raise ContextBudgetPreflightError(
            "llama-server source mutation payload was incorrectly capped by the compact "
            "tool-output policy"
        )

    if (
        getattr(
            llama_cpp_adapter._completion_message,
            "_mmm_bounded_length_recovery_v2",
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
    tools = (
        {"type": "function", "function": {"name": "apply_source_edit"}},
    )

    # Exercise the repository default rather than an operator's explicit runtime
    # override. The production function itself still honors both environment knobs.
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