from __future__ import annotations

"""Context-aware packing for lossless pre-design research synthesis."""

import json
import os
from contextvars import ContextVar
from functools import wraps
from typing import Any

_MARKER = "_mmm_context_aware_research_synthesis_v1"
_MAX_INPUT_BYTES = 65_536
_MAX_GROUP_ITEMS = 64
_CONTEXT_RESERVE_TOKENS = 4_096
_ACTIVE_LIMITS: ContextVar[tuple[int, int] | None] = ContextVar(
    "mmm_research_synthesis_limits",
    default=None,
)


def _planner_limits(router: Any, agentic_module: Any) -> tuple[int, int]:
    fallback_bytes = int(getattr(agentic_module, "_SYNTHESIS_INPUT_BYTES", 3_600))
    fallback_items = int(getattr(agentic_module, "_SYNTHESIS_GROUP_ITEMS", 4))
    page_bytes = max(1, int(getattr(agentic_module, "_EVIDENCE_PAGE_CHARS", 1_800)))
    try:
        config = router.registry.role(router.profile, "planner")
        max_context = int(getattr(config, "max_context", 0) or 0)
        max_new_tokens = int(getattr(config, "max_new_tokens", 0) or 0)
        max_input_tokens = int(getattr(config, "max_input_tokens", 0) or 0)
    except Exception:
        return fallback_bytes, fallback_items
    if max_context <= 0:
        return fallback_bytes, fallback_items

    # Native runtime tuning may lower per-slot context to fit VRAM when it enables
    # multiple llama slots. The live receipt is authoritative for the current managed
    # server, so never pack against a larger registry-only context window.
    raw_receipt = os.environ.get("MMM_LLAMA_RUNTIME_RECEIPT", "").strip()
    if raw_receipt:
        try:
            receipt = json.loads(raw_receipt)
            runtime_context = int(receipt.get("context_per_slot", 0) or 0)
        except (TypeError, ValueError, json.JSONDecodeError):
            runtime_context = 0
        if runtime_context > 0:
            max_context = min(max_context, runtime_context)

    prompt_tokens = max(1, max_context - max(0, max_new_tokens))
    if max_input_tokens > 0:
        prompt_tokens = min(prompt_tokens, max_input_tokens)
    # UTF-8 byte count is a conservative upper bound for byte-fallback tokenizers.
    # Keep an additional host/schema envelope beyond the configured output reserve.
    child_tokens = max(1_024, prompt_tokens - _CONTEXT_RESERVE_TOKENS)
    max_bytes = max(1_024, min(_MAX_INPUT_BYTES, child_tokens))
    max_items = max(
        fallback_items,
        min(_MAX_GROUP_ITEMS, max_bytes // page_bytes),
    )
    return max_bytes, max_items


def _encoded_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )


def _pack(
    notes: list[dict[str, Any]],
    *,
    max_bytes: int,
    max_items: int,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 2
    for note in notes:
        size = _encoded_size(note)
        separator = 1 if current else 0
        if current and (
            len(current) >= max_items
            or current_bytes + separator + size > max_bytes
        ):
            groups.append(current)
            current = []
            current_bytes = 2
            separator = 0
        current.append(note)
        current_bytes += separator + size
    if current:
        groups.append(current)
    return groups


def _atomic_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    atomic: list[dict[str, Any]] = []
    for note in notes:
        domain_id = str(note.get("domain_id", "unknown"))
        sufficient = bool(note.get("sufficient"))
        for claim in note.get("claims", []):
            atomic.append(
                {
                    "domain_id": domain_id,
                    "claims": [claim],
                    "gaps": [],
                    "next_queries": [],
                    "sufficient": sufficient,
                }
            )
        for gap in note.get("gaps", []):
            atomic.append(
                {
                    "domain_id": domain_id,
                    "claims": [],
                    "gaps": [gap],
                    "next_queries": [],
                    "sufficient": sufficient,
                }
            )
        for query in note.get("next_queries", []):
            atomic.append(
                {
                    "domain_id": domain_id,
                    "claims": [],
                    "gaps": [],
                    "next_queries": [query],
                    "sufficient": sufficient,
                }
            )
    return atomic


def harden(agentic_module: Any) -> None:
    """Use planner context capacity without changing evidence/checkpoint semantics."""

    current_group = agentic_module._group_synthesis_notes
    current_hierarchy = agentic_module._hierarchical_synthesis
    if getattr(current_hierarchy, _MARKER, False):
        return

    @wraps(current_group)
    def group_synthesis_notes(notes: list[dict[str, Any]]):
        limits = _ACTIVE_LIMITS.get()
        if limits is None:
            return current_group(notes)
        max_bytes, max_items = limits
        groups = _pack(notes, max_bytes=max_bytes, max_items=max_items)
        # Preserve the original no-progress escape: atomize semantic summaries only
        # when every item would otherwise be forced into its own synthesis request.
        if len(notes) > 1 and len(groups) == len(notes):
            atomic = _atomic_notes(notes)
            if atomic:
                return _pack(atomic, max_bytes=max_bytes, max_items=max_items)
        return groups

    @wraps(current_hierarchy)
    def hierarchical_synthesis(
        agentic_runtime: Any,
        router: Any,
        *args: Any,
        **kwargs: Any,
    ):
        token = _ACTIVE_LIMITS.set(_planner_limits(router, agentic_module))
        try:
            return current_hierarchy(agentic_runtime, router, *args, **kwargs)
        finally:
            _ACTIVE_LIMITS.reset(token)

    group_synthesis_notes._mmm_context_aware_research_synthesis_group_v1 = True  # type: ignore[attr-defined]
    setattr(hierarchical_synthesis, _MARKER, True)
    hierarchical_synthesis.__wrapped__ = current_hierarchy  # type: ignore[attr-defined]
    agentic_module._group_synthesis_notes = group_synthesis_notes
    agentic_module._hierarchical_synthesis = hierarchical_synthesis


__all__ = ["harden"]
