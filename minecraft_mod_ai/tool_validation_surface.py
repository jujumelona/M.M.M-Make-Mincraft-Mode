from __future__ import annotations

"""Pure tool-schema validation-surface composition.

No transport, adapter, router, or runtime owner is imported here. Both the native
adapter and its installation-time contract depend downward on this module.
"""

from collections.abc import Mapping, Sequence
from typing import Any


def tool_name(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return ""
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def assert_unique_schema_names(schemas: Sequence[Any], *, surface: str) -> None:
    seen: set[str] = set()
    for schema in schemas:
        name = tool_name(schema)
        if not name:
            continue
        if name in seen:
            raise RuntimeError(f"duplicate tool schema name {name!r} in {surface} surface")
        seen.add(name)


def validation_surface(visible: Sequence[Any], authorized: Sequence[Any]) -> tuple[Any, ...]:
    assert_unique_schema_names(visible, surface="model-visible")
    assert_unique_schema_names(authorized, surface="authorized-validation")
    result = list(visible)
    visible_names = {name for schema in visible if (name := tool_name(schema))}
    result.extend(
        schema
        for schema in authorized
        if not (name := tool_name(schema)) or name not in visible_names
    )
    return tuple(result)


__all__ = ["assert_unique_schema_names", "tool_name", "validation_surface"]
