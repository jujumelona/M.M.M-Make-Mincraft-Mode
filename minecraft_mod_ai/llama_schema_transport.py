from __future__ import annotations

"""Project host JSON Schema into a conservative llama.cpp transport schema.

The full response schema is an application contract and stays host-owned.  This module
keeps only structural information useful to the sampler and deliberately drops
conditional/validation keywords that can make native grammar compilation brittle.
"""

import copy
from collections.abc import Mapping, Sequence
from typing import Any

_JSON_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)
_BRANCH_KEYS = ("oneOf", "anyOf", "allOf")


def _project_type(value: Any) -> str | list[str] | None:
    if isinstance(value, str):
        return value if value in _JSON_TYPES else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        types = [item for item in value if isinstance(item, str) and item in _JSON_TYPES]
        return list(dict.fromkeys(types)) or None
    return None


def _project_enum(value: Any) -> list[Any] | None:
    if not isinstance(value, list) or not value:
        return None
    return copy.deepcopy(value)


def _fallback_branch(schema: Mapping[str, Any]) -> dict[str, Any]:
    for keyword in _BRANCH_KEYS:
        variants = schema.get(keyword)
        if not isinstance(variants, Sequence) or isinstance(
            variants, (str, bytes, bytearray)
        ):
            continue
        for branch in variants:
            projected = project_llama_transport_schema(branch)
            if projected:
                return projected
    if "const" in schema:
        return {"enum": [copy.deepcopy(schema["const"])]}
    return {}


def project_llama_transport_schema(schema: Any) -> dict[str, Any]:
    """Return a structural schema suitable for llama.cpp sampler initialization.

    Explicit base structure always wins over ``allOf``/``if`` branches.  Detailed
    requirements such as conditionals, numeric/string bounds, refs and cardinality are
    intentionally omitted because the host validates the original schema after decoding.
    """

    if not isinstance(schema, Mapping):
        return {}

    projected_type = _project_type(schema.get("type"))
    has_properties = isinstance(schema.get("properties"), Mapping)
    has_items = isinstance(schema.get("items"), Mapping)

    # Preserve an explicit/base object before considering combinators.  This is important
    # for schemas that append conditional allOf clauses to an otherwise normal object.
    if projected_type == "object" or has_properties:
        result: dict[str, Any] = {"type": "object"}
        raw_properties = schema.get("properties")
        if isinstance(raw_properties, Mapping):
            result["properties"] = {
                str(name): project_llama_transport_schema(child)
                for name, child in raw_properties.items()
                if isinstance(name, str)
            }
            raw_required = schema.get("required")
            if isinstance(raw_required, Sequence) and not isinstance(
                raw_required, (str, bytes, bytearray)
            ):
                required = [
                    name
                    for name in raw_required
                    if isinstance(name, str) and name in result["properties"]
                ]
                if required:
                    result["required"] = required

        additional = schema.get("additionalProperties")
        if additional is False:
            result["additionalProperties"] = False
        elif isinstance(additional, Mapping):
            result["additionalProperties"] = project_llama_transport_schema(additional)
        return result

    if projected_type == "array" or has_items:
        result = {"type": "array"}
        items = schema.get("items")
        result["items"] = (
            project_llama_transport_schema(items) if isinstance(items, Mapping) else {}
        )
        return result

    result = {}
    if projected_type is not None:
        result["type"] = projected_type
    enum = _project_enum(schema.get("enum"))
    if enum is not None:
        result["enum"] = enum
    if result:
        return result

    return _fallback_branch(schema)


__all__ = ["project_llama_transport_schema"]
