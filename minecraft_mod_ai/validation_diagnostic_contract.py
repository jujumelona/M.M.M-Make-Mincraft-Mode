from __future__ import annotations

from typing import Any


def diagnostic_errors(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    raw = receipt.get("diagnostics", {})
    if isinstance(raw, dict):
        values = [
            item
            for group in raw.values()
            if isinstance(group, list)
            for item in group
            if isinstance(item, dict)
        ]
    elif isinstance(raw, list):
        values = [item for item in raw if isinstance(item, dict)]
    else:
        values = []
    return [item for item in values if int(item.get("severity", 1)) <= 2]


def install(validation_module: Any) -> None:
    """Make progressive repair consume the actual URI->diagnostics JDT-LS shape."""

    diagnostic_errors._mmm_flattened_jdt_mapping = True
    validation_module._diagnostic_errors = diagnostic_errors
