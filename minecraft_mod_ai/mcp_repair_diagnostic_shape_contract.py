from __future__ import annotations

from functools import wraps
from typing import Any, Mapping


def install(repair_batch_module: Any) -> None:
    """Teach MCP repair enrichment to read JDT-LS v2 URI-grouped diagnostics."""

    current = repair_batch_module._diagnostic_text
    if getattr(current, "_mmm_jdt_v2_diagnostic_text", False):
        return

    @wraps(current)
    def diagnostic_text(evidence: Mapping[str, Any]) -> str:
        parts: list[str] = []
        receipt = evidence.get("diagnostics", {})
        raw = receipt.get("diagnostics", {}) if isinstance(receipt, Mapping) else {}
        if isinstance(raw, Mapping):
            diagnostics = [
                item
                for group in raw.values()
                if isinstance(group, list)
                for item in group
                if isinstance(item, Mapping)
            ]
        elif isinstance(raw, list):
            diagnostics = [item for item in raw if isinstance(item, Mapping)]
        else:
            diagnostics = []

        for item in diagnostics:
            parts.append(str(item.get("message", "")))
            parts.append(str(item.get("code", "")))
        build = evidence.get("build", {})
        if isinstance(build, Mapping):
            parts.append(str(build.get("error", "")))
        return "\n".join(part for part in parts if part)

    diagnostic_text._mmm_jdt_v2_diagnostic_text = True  # type: ignore[attr-defined]
    repair_batch_module._diagnostic_text = diagnostic_text


__all__ = ["install"]
