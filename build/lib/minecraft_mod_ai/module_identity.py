from __future__ import annotations

"""Canonical logical module identity normalization.

A logical module id is deliberately distinct from a Gradle project path. This helper is
pure and host-owned so planning/artifact code does not depend on retired target-grounding
runtime patches.
"""

import re
from collections.abc import Mapping
from typing import Any


def logical_module_id(raw_path: str, item: Mapping[str, Any] | None = None) -> str:
    metadata = item or {}
    explicit = str(
        metadata.get("logical_module_id")
        or metadata.get("artifact_id")
        or metadata.get("name")
        or ""
    ).strip()
    if explicit:
        source = explicit
    elif raw_path == ":":
        source = "root"
    elif raw_path.startswith(":"):
        source = raw_path.strip(":").replace(":", "_")
    else:
        source = raw_path

    value = re.sub(r"[^a-z0-9_]+", "_", source.casefold()).strip("_")
    value = re.sub(r"_+", "_", value)
    if not value:
        value = "root"
    if not value[0].isalpha():
        value = "module_" + value
    return value[:64]


__all__ = ["logical_module_id"]
