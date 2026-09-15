from __future__ import annotations

"""Canonicalize and compact host observations before they enter agent transcripts."""

import json
from functools import wraps
from typing import Any

_DEFAULT_MODEL_OBSERVATION_BYTES = 16 * 1024


def _sort_key(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _dedupe_normalized_result(result: Any) -> Any:
    """Keep one semantic copy when MCP mirrors JSON across result channels."""

    if not isinstance(result, dict):
        return result
    compact = dict(result)
    structured = compact.get("structured_content")
    parsed = compact.get("parsed_text")
    texts = compact.get("text")

    if parsed is not None and isinstance(texts, list) and len(texts) == 1:
        try:
            text_value = json.loads(texts[0])
        except (json.JSONDecodeError, TypeError):
            text_value = object()
        if text_value == parsed:
            compact["text"] = []

    if parsed is not None and structured == parsed:
        compact["parsed_text"] = None

    return compact


def install(*, agent_tool_runtime_module: Any) -> None:
    # A 48 KiB single observation can dominate a 32K-token local-model slot before
    # message compaction even runs. Keep the default model observation page small;
    # callers that genuinely need a larger page can still opt in through the existing
    # MMM_AGENT_OBSERVATION_BYTES environment override.
    agent_tool_runtime_module._DEFAULT_MAX_TOOL_RESULT_BYTES = (
        _DEFAULT_MODEL_OBSERVATION_BYTES
    )

    current_normalize = agent_tool_runtime_module._normalize_tool_result
    if not getattr(current_normalize, "_mmm_duplicate_channels_compacted", False):

        @wraps(current_normalize)
        def normalize(raw: Any) -> Any:
            return _dedupe_normalized_result(current_normalize(raw))

        normalize._mmm_duplicate_channels_compacted = True  # type: ignore[attr-defined]
        normalize.__wrapped__ = current_normalize  # type: ignore[attr-defined]
        agent_tool_runtime_module._normalize_tool_result = normalize

    current_jsonable = agent_tool_runtime_module._jsonable
    if not getattr(current_jsonable, "_mmm_unordered_canonical", False):

        @wraps(current_jsonable)
        def jsonable(value: Any) -> Any:
            if isinstance(value, (set, frozenset)):
                items = [jsonable(item) for item in value]
                return sorted(items, key=_sort_key)
            return current_jsonable(value)

        jsonable._mmm_unordered_canonical = True  # type: ignore[attr-defined]
        jsonable.__wrapped__ = current_jsonable  # type: ignore[attr-defined]
        agent_tool_runtime_module._jsonable = jsonable

    current_sanitize = agent_tool_runtime_module._sanitize_observation
    if not getattr(current_sanitize, "_mmm_unordered_canonical", False):

        @wraps(current_sanitize)
        def sanitize(value: Any) -> Any:
            if isinstance(value, (set, frozenset)):
                items = [sanitize(item) for item in value]
                return sorted(items, key=_sort_key)
            return current_sanitize(value)

        sanitize._mmm_unordered_canonical = True  # type: ignore[attr-defined]
        sanitize.__wrapped__ = current_sanitize  # type: ignore[attr-defined]
        agent_tool_runtime_module._sanitize_observation = sanitize

    current_metadata = agent_tool_runtime_module._small_metadata
    if not getattr(current_metadata, "_mmm_unordered_canonical", False):

        @wraps(current_metadata)
        def small_metadata(value: Any, *, depth: int = 0) -> Any:
            if isinstance(value, (set, frozenset)):
                items = [small_metadata(item, depth=depth + 1) for item in value]
                return sorted(items, key=_sort_key)[:16]
            return current_metadata(value, depth=depth)

        small_metadata._mmm_unordered_canonical = True  # type: ignore[attr-defined]
        small_metadata.__wrapped__ = current_metadata  # type: ignore[attr-defined]
        agent_tool_runtime_module._small_metadata = small_metadata


__all__ = ["install"]
