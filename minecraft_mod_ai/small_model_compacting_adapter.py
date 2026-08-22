from __future__ import annotations

import copy
import json
from dataclasses import replace
from functools import wraps
from typing import Any

from .model_context_budget import fit_messages_to_context
from .small_model_context_compaction import compact_messages


_MARKER = "_mmm_lossless_context_compaction"
_IMPLEMENTATION_SOURCE_SEED_BYTES = 12 * 1024
_REDUNDANT_IMPLEMENTATION_FIELDS = frozenset(
    {
        "project_manifest",
        "source_observation_receipt",
        "research_context",
    }
)


def _json_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _trim_record_tail(
    value: dict[str, Any],
    key: str,
    *,
    current_bytes: int,
    byte_budget: int,
) -> int:
    """Trim one record list without repeatedly serializing the full payload."""

    records = value.get(key)
    if not isinstance(records, list):
        return current_bytes
    while records and current_bytes > byte_budget:
        removed = records.pop()
        current_bytes -= _json_bytes(removed)
        if records:
            # Canonical JSON has one comma between adjacent list elements.
            current_bytes -= 1
    return current_bytes


def _bounded_exact_source_seed(value: Any, *, byte_budget: int) -> Any:
    """Bound the first exact-source seed while preserving host receipt metadata.

    Supplemental workspace/RAG tools remain available after the first decode.  The
    seed therefore needs representative exact source, not an entire project page on
    every subsequent tool turn.  Records are removed whole so byte ranges and hashes
    are never rewritten or turned into approximate evidence.
    """

    if not isinstance(value, dict):
        return value
    current_bytes = _json_bytes(value)
    if current_bytes <= byte_budget:
        return value

    # The seed came from json.loads in the live path, so deepcopy preserves the same
    # JSON-compatible structure without paying for another encode/decode round trip.
    bounded = copy.deepcopy(value)
    original_records = sum(
        len(bounded.get(key, ()))
        for key in ("global_anchors", "page_observations")
        if isinstance(bounded.get(key), list)
    )
    for key in ("page_observations", "global_anchors"):
        current_bytes = _trim_record_tail(
            bounded,
            key,
            current_bytes=current_bytes,
            byte_budget=byte_budget,
        )
        if current_bytes <= byte_budget:
            break

    bounded["global_anchor_count"] = len(bounded.get("global_anchors", ()))
    retained_records = sum(
        len(bounded.get(key, ()))
        for key in ("global_anchors", "page_observations")
        if isinstance(bounded.get(key), list)
    )
    bounded["model_seed_compaction"] = {
        "bounded_bytes": int(byte_budget),
        "omitted_record_count": max(0, original_records - retained_records),
        "supplemental_retrieval_available": True,
    }
    return bounded


def _compact_implementation_seed(messages: Any) -> tuple[dict[str, Any], ...]:
    """Canonicalize duplicated host evidence in one ``implement_module`` request.

    ``host_grounding`` already carries the authoritative manifest/source/research
    receipts, and the research router injects its own bounded live research bundle.
    Re-sending the raw receipts and research payload beside those owners made every
    tool round re-prefill tens of kilobytes of duplicate context.
    """

    compacted: list[dict[str, Any]] = []
    for raw_message in messages:
        message = dict(raw_message)
        content = message.get("content")
        if (
            str(message.get("role", "")).casefold() != "user"
            or not isinstance(content, str)
            or not content.lstrip().startswith("{")
        ):
            compacted.append(message)
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            compacted.append(message)
            continue
        if not isinstance(payload, dict) or payload.get("phase") != "implement_module":
            compacted.append(message)
            continue

        for key in _REDUNDANT_IMPLEMENTATION_FIELDS:
            payload.pop(key, None)
        if "initial_exact_source_context" in payload:
            payload["initial_exact_source_context"] = _bounded_exact_source_seed(
                payload["initial_exact_source_context"],
                byte_budget=_IMPLEMENTATION_SOURCE_SEED_BYTES,
            )
        message["content"] = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        compacted.append(message)
    return tuple(compacted)


class CompactingAdapter:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def generate_turn(self, request: Any) -> Any:
        # Canonicalize the immutable implementation seed before historical exchange
        # compaction.  This shrinks the prefix reused by every tool round instead of
        # waiting until the context window is nearly exhausted.
        messages = _compact_implementation_seed(request.messages)
        messages = compact_messages(messages)
        messages = fit_messages_to_context(
            messages,
            config=getattr(self.inner, "config", None),
            tools=getattr(request, "tools", ()) or (),
        )
        if messages == tuple(request.messages):
            return self.inner.generate_turn(request)

        # Clone the frozen GenerationRequest instead of reconstructing it field by
        # field. This preserves task/prompt/metadata and any future request fields
        # added by another runtime contract while changing only the compacted history.
        return self.inner.generate_turn(replace(request, messages=messages))


def _is_live_compaction_wrapper(value: Any) -> bool:
    """Identify the actual wrapper implementation, not inherited marker metadata.

    ``functools.wraps`` copies the immediate wrapper metadata, but late composition can
    place another wrapper above a callable whose ``_mmm_*`` marker is used as an
    executable contract by independent validators. The code object cannot be copied by
    ``wraps`` and is the authoritative owner check for this runtime boundary.
    """

    code = getattr(value, "__code__", None)
    if code is None:
        return False
    filename = str(getattr(code, "co_filename", "")).replace("\\", "/")
    return (
        filename.endswith("/small_model_compacting_adapter.py")
        and str(getattr(code, "co_name", "")) == "generate_with_compaction"
    )


def install(model_router_module: Any) -> None:
    """Bind lossless tool-context compaction at the live model-router boundary."""
    current = model_router_module.ModelRouter._generate_with_tools
    if _is_live_compaction_wrapper(current):
        return

    @wraps(current)
    def generate_with_compaction(
        self,
        *,
        config,
        adapter,
        request,
        runtime,
        stage,
        role,
    ):
        return current(
            self,
            config=config,
            adapter=CompactingAdapter(adapter),
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

    setattr(generate_with_compaction, _MARKER, True)
    model_router_module.ModelRouter._generate_with_tools = generate_with_compaction


__all__ = [
    "CompactingAdapter",
    "_bounded_exact_source_seed",
    "_compact_implementation_seed",
    "_is_live_compaction_wrapper",
    "install",
]
