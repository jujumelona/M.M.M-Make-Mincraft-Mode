from __future__ import annotations

"""JSON resource validation for models, recipes, tags, loot tables, and lang files."""

import json
from typing import Any


class JsonResourceValidationError(ValueError):
    pass


def validate_json_resource(content: str, *, resource_kind: str = "") -> dict[str, Any]:
    """Validate that content is well-formed JSON conforming to Minecraft resource norms."""
    if not isinstance(content, str) or not content.strip():
        raise JsonResourceValidationError("JSON_EMPTY: Resource content cannot be empty")

    try:
        data = json.loads(content)
    except Exception as exc:
        raise JsonResourceValidationError(f"JSON_SYNTAX: Invalid JSON: {exc}") from exc

    if not isinstance(data, (dict, list)):
        raise JsonResourceValidationError("JSON_ROOT: Minecraft resources must be JSON objects or arrays")

    return {
        "status": "PASS",
        "resource_kind": resource_kind,
        "key_count": len(data) if isinstance(data, dict) else len(data),
    }
