from __future__ import annotations

"""Canonicalize unordered host values before they enter agent transcripts."""

import json
from functools import wraps
from typing import Any


def _sort_key(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def install(*, agent_tool_runtime_module: Any) -> None:
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
