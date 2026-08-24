from __future__ import annotations

"""Context-budget fitting for tool-capable small/local model turns.

Large projects remain host-indexed and retrieval-driven; the model only sees the
bounded source/evidence needed for the current action. This module is the shared
request-budget owner for both live history fitting and native tool action decode.
"""

import hashlib
import json
import os
from typing import Any, Mapping, Sequence

_DEFAULT_CONTEXT_BYTES = 96 * 1024
_MIN_CONTEXT_BYTES = 12 * 1024
_MAX_CONTEXT_BYTES = 512 * 1024
_BYTES_PER_TOKEN_BUDGET = 2
_CONTEXT_TOKEN_GUARD = 2048
_TOOL_HEAD_BYTES = 6144
_TOOL_TAIL_BYTES = 2048
_MIN_TOOL_PREVIEW_BYTES = 2048


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _canonical_size(value: Any) -> int:
    return len(_canonical_bytes(value))


def _default_context_bytes() -> int:
    raw = os.environ.get("MMM_SMALL_AGENT_CONTEXT_BYTES", "").strip()
    if not raw:
        return _DEFAULT_CONTEXT_BYTES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_CONTEXT_BYTES
    return max(_MIN_CONTEXT_BYTES, min(_MAX_CONTEXT_BYTES, value))


def effective_context_tokens(config: Any) -> int:
    """Return the context of the runtime slot that will actually serve the request."""

    adapter = str(getattr(config, "adapter", "") or "").strip().casefold()
    if adapter == "llama_cpp":
        from .llama_server_runtime_tuning import _per_request_context

        return max(0, int(_per_request_context(config)))
    try:
        return max(0, int(getattr(config, "max_context", 0) or 0))
    except (TypeError, ValueError):
        return 0


def tool_action_token_budget(config: Any) -> int:
    """Return one bounded function-call page budget, not a whole-job limit."""

    raw = os.environ.get("MMM_LLAMA_TOOL_MAX_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("MMM_LLAMA_TOOL_MAX_TOKENS must be a positive integer") from exc
        if value <= 0:
            raise ValueError("MMM_LLAMA_TOOL_MAX_TOKENS must be a positive integer")
        return value
    context = effective_context_tokens(config)
    if context <= 0:
        return 4096
    return max(2048, min(8192, context // 4))


def request_message_budget(config: Any, tools: Sequence[Any] = ()) -> int:
    """Return a conservative input budget from the slot that actually serves the turn.

    A tool page reserves the same finite action allowance used by the transport, so
    prompt fitting and decode can never independently spend the same context window.
    Non-tool llama pages keep their existing semantic/unlimited output policy.
    """

    default = _default_context_bytes()
    try:
        max_context = effective_context_tokens(config)
        max_new_tokens = max(0, int(getattr(config, "max_new_tokens", 0) or 0))
    except (TypeError, ValueError):
        return default
    if max_context <= 0:
        return default

    adapter = str(getattr(config, "adapter", "") or "").strip().casefold()
    if adapter == "llama_cpp":
        reserved_output_tokens = tool_action_token_budget(config) if tools else 0
    else:
        reserved_output_tokens = max_new_tokens
    available_input_tokens = max(
        2048,
        max_context - reserved_output_tokens - _CONTEXT_TOKEN_GUARD,
    )
    context_bytes = available_input_tokens * _BYTES_PER_TOKEN_BUDGET
    tool_bytes = _canonical_size(tuple(tools)) if tools else 0
    derived = context_bytes - tool_bytes
    return max(_MIN_CONTEXT_BYTES, min(default, derived))


def _tool_preview(
    raw: str,
    *,
    allowance: int,
    encoded: bytes | None = None,
) -> str:
    encoded = raw.encode("utf-8") if encoded is None else encoded
    if len(encoded) <= allowance:
        return raw
    allowance = max(_MIN_TOOL_PREVIEW_BYTES, allowance)
    head_budget = min(_TOOL_HEAD_BYTES, max(1024, allowance * 3 // 4))
    tail_budget = min(_TOOL_TAIL_BYTES, max(512, allowance - head_budget))
    head = encoded[:head_budget].decode("utf-8", errors="ignore")
    tail = encoded[-tail_budget:].decode("utf-8", errors="ignore")
    return head + "\n...[exact observation archived by host]...\n" + tail


def _summary_payload(
    message: Mapping[str, Any],
    *,
    archive: Mapping[str, Any],
    preview_bytes: int,
) -> str:
    raw = str(message.get("content", ""))
    raw_bytes = raw.encode("utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    summary: dict[str, Any] = {
        "_mmm_context_compaction": {
            "schema_version": "mmm/tool-observation-context-v1",
            "raw_observation": dict(archive),
            "original_bytes": len(raw_bytes),
            "sha256": "sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
            "protocol_role_preserved": True,
        },
        "tool": str(message.get("name", "")),
        "preview": _tool_preview(raw, allowance=preview_bytes, encoded=raw_bytes),
    }
    if isinstance(parsed, Mapping):
        for key in ("ok", "tool", "error", "_mmm_source_mutation"):
            if key in parsed:
                value = parsed.get(key)
                if key == "error" and isinstance(value, str):
                    value = value[:1200]
                summary[key] = value
        result = parsed.get("result")
        if isinstance(result, Mapping):
            compact_result: dict[str, Any] = {}
            for key in (
                "receipt",
                "preserved_evidence",
                "truncated",
                "original_bytes",
                "hint",
                "_mmm_observation",
            ):
                if key in result:
                    compact_result[key] = result.get(key)
            if compact_result:
                summary["result"] = compact_result
    return json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def bounded_tool_message(
    message: Mapping[str, Any],
    *,
    config: Any,
    tools: Sequence[Any] = (),
) -> Mapping[str, Any]:
    """Archive one oversized tool observation before it enters live history."""

    original = dict(message)
    if str(original.get("role", "")) != "tool" or not isinstance(
        original.get("content"), str
    ):
        return original
    raw = str(original["content"])
    raw_bytes = raw.encode("utf-8")
    request_budget = request_message_budget(config, tools)
    allowance = max(
        _MIN_TOOL_PREVIEW_BYTES,
        min(16 * 1024, max(_MIN_TOOL_PREVIEW_BYTES, request_budget // 4)),
    )
    if len(raw_bytes) <= allowance:
        return original

    from .small_model_context_compaction import _archive_transcript

    archive = _archive_transcript((original,))
    if not bool(archive.get("available")):
        return original
    replacement = dict(original)
    replacement["content"] = _summary_payload(
        original,
        archive=archive,
        preview_bytes=min(8 * 1024, allowance),
    )
    return replacement


def _compact_tool_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    preview_bytes: int,
) -> tuple[Mapping[str, Any], ...]:
    values: list[Mapping[str, Any]] = [dict(message) for message in messages]
    current_size = _canonical_size(values)
    if current_size <= budget:
        return tuple(values)

    from .small_model_context_compaction import _archive_transcript

    candidates = sorted(
        (
            (len(str(message.get("content", "")).encode("utf-8")), index)
            for index, message in enumerate(values)
            if str(message.get("role", "")) == "tool"
            and isinstance(message.get("content"), str)
        ),
        reverse=True,
    )
    for raw_size, index in candidates:
        if raw_size <= preview_bytes:
            continue
        original = dict(values[index])
        archive = _archive_transcript((original,))
        if not bool(archive.get("available")):
            continue
        replacement = dict(original)
        replacement["content"] = _summary_payload(
            original,
            archive=archive,
            preview_bytes=preview_bytes,
        )
        current_size += _canonical_size(replacement) - _canonical_size(original)
        values[index] = replacement
        if current_size <= budget:
            break
    return tuple(values)


def _last_mutation_exchange_start(
    messages: Sequence[Mapping[str, Any]],
) -> int | None:
    """Return the assistant boundary that owns the latest proven source mutation."""

    from .source_mutation_contract import mutation_observation_applied

    latest_tool_index: int | None = None
    for index, message in enumerate(messages):
        if mutation_observation_applied(message):
            latest_tool_index = index
    if latest_tool_index is None:
        return None
    for index in range(latest_tool_index - 1, -1, -1):
        message = messages[index]
        if str(message.get("role", "")) != "assistant":
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes, bytearray)):
            return index
    return None


def _compact_old_exchanges(
    messages: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> tuple[Mapping[str, Any], ...]:
    original = tuple(messages)
    if _canonical_size(original) <= budget:
        return original
    assistants = [
        index
        for index, item in enumerate(original)
        if str(item.get("role", "")) == "assistant"
    ]
    if len(assistants) < 2:
        return original

    from .small_model_context_compaction import (
        _archive_preview,
        _archive_transcript,
        _ledger,
    )

    first = assistants[0]
    mutation_start = _last_mutation_exchange_start(original)
    for keep in (2, 1):
        if len(assistants) <= keep:
            continue
        start = assistants[-keep]
        if mutation_start is not None and first <= mutation_start < start:
            start = mutation_start
        dropped = original[first:start]
        if not dropped:
            continue
        archive = _archive_preview(dropped)
        context = {
            "role": "system",
            "content": (
                "HOST COMPACTED VERIFIED CONTEXT. Exact prior tool observations are "
                "recoverable from raw_history; retain the current task and recent "
                "tool protocol exactly.\n"
                + json.dumps(
                    _ledger(dropped, archive=archive),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        }
        compacted: tuple[Mapping[str, Any], ...] = (
            *original[:first],
            context,
            *original[start:],
        )
        if _canonical_size(compacted) > budget:
            continue
        persisted = _archive_transcript(dropped)
        if not bool(persisted.get("available")):
            return original
        if persisted != archive:
            context = {
                "role": "system",
                "content": (
                    "HOST COMPACTED VERIFIED CONTEXT. Exact prior tool observations "
                    "are recoverable from raw_history; retain the current task and "
                    "recent tool protocol exactly.\n"
                    + json.dumps(
                        _ledger(dropped, archive=persisted),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
            }
            compacted = (*original[:first], context, *original[start:])
        return compacted
    return original


def fit_messages_to_context(
    messages: Sequence[Mapping[str, Any]],
    *,
    config: Any,
    tools: Sequence[Any] = (),
) -> tuple[Mapping[str, Any], ...]:
    """Fit a model turn before decode, including the first assistant/tool exchange."""

    budget = request_message_budget(config, tools)
    values = tuple(messages)
    if _canonical_size(values) <= budget:
        return values

    values = _compact_tool_messages(values, budget=budget, preview_bytes=8 * 1024)
    if _canonical_size(values) <= budget:
        return values

    values = _compact_old_exchanges(values, budget=budget)
    if _canonical_size(values) <= budget:
        return values

    return _compact_tool_messages(values, budget=budget, preview_bytes=4 * 1024)


def emergency_fit_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    budget_bytes: int = 40 * 1024,
) -> tuple[Mapping[str, Any], ...]:
    """Bound a retry payload when a backend reports context pressure."""

    budget = max(_MIN_CONTEXT_BYTES, min(_MAX_CONTEXT_BYTES, int(budget_bytes)))
    values = _compact_tool_messages(messages, budget=budget, preview_bytes=4 * 1024)
    if _canonical_size(values) <= budget:
        return values
    return _compact_old_exchanges(values, budget=budget)


__all__ = [
    "bounded_tool_message",
    "effective_context_tokens",
    "emergency_fit_messages",
    "fit_messages_to_context",
    "request_message_budget",
    "tool_action_token_budget",
]
