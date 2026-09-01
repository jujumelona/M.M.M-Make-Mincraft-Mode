from __future__ import annotations

"""Reject internal model reasoning from user-facing accepted design fields."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

_META_MARKERS = (
    re.compile(r"<\s*/?\s*think\b", re.IGNORECASE),
    re.compile(r"\bi\s+need\s+to\b", re.IGNORECASE),
    re.compile(r"\bi\s+should\b", re.IGNORECASE),
    re.compile(r"\bthe\s+user\s+wants\b", re.IGNORECASE),
    re.compile(r"\bthe\s+user\s+asked\b", re.IGNORECASE),
    re.compile(r"\bbranch[- ]policy\b", re.IGNORECASE),
    re.compile(r"\bmain[- ]only\s+(?:branch|policy)\b", re.IGNORECASE),
    re.compile(r"\brepository\s+(?:branch|policy)\s+(?:check|review)\b", re.IGNORECASE),
)
_GUARDED_FIELDS = frozenset({"title", "pitch", "core_loop"})


def _text_values(value: Any):
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _text_values(child)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _text_values(child)


def contains_internal_model_meta(value: Any) -> bool:
    return any(pattern.search(text) for text in _text_values(value) for pattern in _META_MARKERS)


def assert_design_field_clean(field: str, value: Any) -> None:
    if field not in _GUARDED_FIELDS:
        return
    if contains_internal_model_meta(value):
        raise ValueError(
            f"game_design.{field} contains internal model reasoning/meta output and cannot be accepted"
        )


__all__ = ["assert_design_field_clean", "contains_internal_model_meta"]
