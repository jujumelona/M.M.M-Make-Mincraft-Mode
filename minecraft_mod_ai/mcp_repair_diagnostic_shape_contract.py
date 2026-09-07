from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

from .validation_diagnostic_contract import diagnostic_items


def install(repair_batch_module: Any) -> None:
    """Teach MCP repair enrichment to consume the canonical diagnostic receipt shape."""

    current = repair_batch_module._diagnostic_text
    if getattr(current, "_mmm_jdt_v2_diagnostic_text", False):
        return

    @wraps(current)
    def diagnostic_text(evidence: Mapping[str, Any]) -> str:
        parts: list[str] = []
        for item in diagnostic_items(evidence.get("diagnostics", {})):
            parts.append(str(item.get("message", "")))
            parts.append(str(item.get("code", "")))
        build = evidence.get("build", {})
        if isinstance(build, Mapping):
            parts.append(str(build.get("error", "")))
        return "\n".join(part for part in parts if part)

    diagnostic_text._mmm_jdt_v2_diagnostic_text = True  # type: ignore[attr-defined]
    repair_batch_module._diagnostic_text = diagnostic_text


__all__ = ["install"]
