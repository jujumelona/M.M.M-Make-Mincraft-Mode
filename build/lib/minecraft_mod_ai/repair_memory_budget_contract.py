from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any

_MEMORY_CONTEXT_BYTES = 12 * 1024
_PATTERN_LIMIT = 8
_EXCERPT_BYTES = 1024


def _truncate_utf8(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")


def install(agentic_module: Any) -> None:
    """Keep verified episodic repair context bounded and high-signal."""

    current_pattern = agentic_module._repair_pattern
    if not getattr(current_pattern, "_mmm_bounded_repair_pattern", False):

        def compact_pattern(
            operations: Iterable[Mapping[str, Any]],
        ) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            for item in operations:
                if len(result) >= _PATTERN_LIMIT:
                    break
                content = item.get("content")
                replacements = item.get("replacements")
                excerpt = ""
                if isinstance(content, str):
                    excerpt = _truncate_utf8(content, _EXCERPT_BYTES)
                elif replacements is not None:
                    excerpt = _truncate_utf8(
                        json.dumps(replacements, ensure_ascii=False, sort_keys=True),
                        _EXCERPT_BYTES,
                    )
                result.append(
                    {
                        "operation": str(item.get("operation", "")),
                        "path": str(item.get("path", "")),
                        "repair_excerpt": excerpt,
                    }
                )
            return result

        compact_pattern._mmm_bounded_repair_pattern = True  # type: ignore[attr-defined]
        compact_pattern.__wrapped__ = current_pattern  # type: ignore[attr-defined]
        agentic_module._repair_pattern = compact_pattern

    current_read = agentic_module._read_memory
    if getattr(current_read, "_mmm_bounded_repair_memory", False):
        return

    @wraps(current_read)
    def read_bounded(root: Path, signature: str, *, limit: int = 4):
        matches = current_read(root, signature, limit=min(3, limit))
        selected: list[dict[str, Any]] = []
        for match in matches:
            candidate = {
                "similarity": match.get("similarity", 0.0),
                "signature_sha256": match.get("signature_sha256", ""),
                "evidence": match.get("evidence", {}),
                "repair_pattern": list(match.get("repair_pattern", []))[:_PATTERN_LIMIT],
            }
            trial = [*selected, candidate]
            size = len(
                json.dumps(
                    trial,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            if size > _MEMORY_CONTEXT_BYTES:
                continue
            selected.append(candidate)
        return selected

    read_bounded._mmm_bounded_repair_memory = True  # type: ignore[attr-defined]
    agentic_module._read_memory = read_bounded


__all__ = ["install"]
