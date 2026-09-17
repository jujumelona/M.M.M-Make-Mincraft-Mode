from __future__ import annotations

"""Dependency-free normalization helpers shared by runtime contract modules."""

import json
from collections.abc import Mapping, Sequence
from typing import Any


def structured_payload(
    content: Any,
) -> Mapping[str, Any] | list[Any] | tuple[Any, ...] | None:
    if isinstance(content, (Mapping, list, tuple)):
        return content
    if not isinstance(content, str):
        return None
    raw = content.strip()
    if not raw.startswith(("{", "[")):
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, (Mapping, list, tuple)) else None


def as_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


__all__ = ["as_sequence", "structured_payload"]
