from __future__ import annotations

"""Context-budget fitting for tool-capable small/local model turns.

The ordinary history compactor is exchange-oriented: it can replace older assistant
rounds with a recoverable ledger.  The first tool round is different because there is
no *older* assistant exchange to drop yet.  Large RAG/LSP observations can therefore
fill the server context before the second assistant turn.  This module handles that
first-round boundary by archiving exact tool observations and keeping a bounded,
recoverable evidence preview in the protocol-preserving tool message.
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


def _default_context_bytes() -> int:
    raw = os.environ.get("MMM_SMALL_AGENT_CONTEXT_BYTES", "").strip()
    if not raw:
        return _DEFAULT_CONTEXT_BYTES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_CONTEXT_BYTES
    return max(_MIN_CONTEXT_BYTES, min(_MAX_CONTEXT_BYTES, value))


def request_message_budget(config: Any, tools: Sequence[Any] = ()) -> int:
    """Return a conservative message-byte budget with tool/output space reserved."""

    default = _default_context_bytes()
    try:
        max_context = max(0, int(getattr(config, "max_context", 0) or 0))
        max_new_tokens = max(0, int(getattr(config, "max_new_tokens", 0) or 0))
    except (TypeError, ValueError):
        return default
    if max_context <= 0:
        return default

    adapter = str(getattr(config, "adapter", "") or "").strip().casefold()
    # Native llama-server production requests use max_tokens=-1. Reserving the
    # registry max_new_tokens here would silently reintroduce the old 8K-style cap on
    # the input side even though generation itself is unbounded. Keep only a fixed
    # safety guard for EOS/tool completion and the actual tool-schema footprint.
    reserved_output_tokens = 0 if adapter == "llama_cpp" else max_new_tokens

    # Byte accounting is intentionally conservative for code/JSON-heavy turns.
    # Tool schemas consume the same server context but are not part of messages.
    available_input_tokens = max(
        2048,
        max_context - reserved_output_tokens - _CONTEXT_TOKEN_GUARD,
    )
    context_bytes = available_input_tokens * _BYTES_PER_TOKEN_BUDGET
    tool_bytes = len(_canonical_bytes(tuple(tools))) if tools else 0
    derived = context_bytes - tool_bytes
    return max(_MIN_CONTEXT_BYTES, min(default, derived))


def _tool_preview(raw: str, *, allowance: int) -> str:
    encoded = raw.encode("utf-8")
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
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    summary: dict[str, Any] = {
        "_mmm_context_compaction": {
            "schema_version": "mmm/tool-observation-context-v1",
            "raw_observation": dict(archive),
            "original_bytes": len(raw.encode("utf-8")),
            "sha256": "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "protocol_role_preserved": True,
        },
        "tool": str(message.get("name", "")),
        "preview": _tool_preview(raw, allowance=preview_bytes),
    }
    if isinstance(parsed, Mapping):
        for key in ("ok", "tool", "error"):
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


def _compact_tool_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    preview_bytes: int,
) -> tuple[Mapping[str, Any], ...]:
    values: list[Mapping[str, Any]] = [dict(message) for message in messages]
    if len(_canonical_bytes(values)) <= budget:
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
            # Keep the research contract lossless.  A failed archive must never be
            # disguised as successful context compaction.
            continue
        replacement = dict(original)
        replacement["content"] = _summary_payload(
            original,
            archive=archive,
            preview_bytes=preview_bytes,
        )
        values[index] = replacement
        if len(_canonical_bytes(values)) <= budget:
            break
    return tuple(values)


def _compact_old_exchanges(
    messages: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> tuple[Mapping[str, Any], ...]:
    original = tuple(messages)
    if len(_canonical_bytes(original)) <= budget:
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
    for keep in (2, 1):
        if len(assistants) <= keep:
            continue
        start = assistants[-keep]
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
        if len(_canonical_bytes(compacted)) > budget:
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
    if len(_canonical_bytes(values)) <= budget:
        return values

    # First pass keeps enough evidence text for code decisions while removing the
    # pathological 48 KiB-per-tool accumulation seen after parallel RAG/LSP reads.
    values = _compact_tool_messages(values, budget=budget, preview_bytes=8 * 1024)
    if len(_canonical_bytes(values)) <= budget:
        return values

    values = _compact_old_exchanges(values, budget=budget)
    if len(_canonical_bytes(values)) <= budget:
        return values

    # Emergency second pass still keeps an exact archive pointer and a useful head/
    # tail evidence sample, but makes the protocol fit before spending decode time.
    return _compact_tool_messages(values, budget=budget, preview_bytes=4 * 1024)


def emergency_fit_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    budget_bytes: int = 40 * 1024,
) -> tuple[Mapping[str, Any], ...]:
    """Bound a retry payload when a backend reports finish_reason='length'."""

    budget = max(_MIN_CONTEXT_BYTES, min(_MAX_CONTEXT_BYTES, int(budget_bytes)))
    values = _compact_tool_messages(messages, budget=budget, preview_bytes=4 * 1024)
    if len(_canonical_bytes(values)) <= budget:
        return values
    return _compact_old_exchanges(values, budget=budget)


__all__ = [
    "emergency_fit_messages",
    "fit_messages_to_context",
    "request_message_budget",
]
