"""Generation-time schema binding for host-owned design vocabularies."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .content_design_contract import CONTENT_KIND_TO_FACT_TYPE, CONTENT_KINDS


_CONTEXT_ENUM_BINDINGS = {
    "slot_id": "allowed_slots",
    "property": "allowed_properties",
    "relation_type": "allowed_relation_types",
}


class DesignGenerationSchemaError(ValueError):
    pass


def _bind_context_enums(
    properties: dict[str, Any],
    context: Mapping[str, Any],
) -> None:
    for field, context_key in _CONTEXT_ENUM_BINDINGS.items():
        field_schema = properties.get(field)
        values = context.get(context_key)
        if isinstance(field_schema, dict) and isinstance(values, (list, tuple)):
            normalized = list(
                dict.fromkeys(value for value in values if isinstance(value, str))
            )
            if normalized:
                field_schema["enum"] = normalized


def _bind_content_entity_kind(properties: dict[str, Any]) -> None:
    kind_schema = properties.get("kind")
    if not isinstance(kind_schema, dict):
        raise DesignGenerationSchemaError("CONTENT_KIND_SCHEMA_INVALID")
    kind_schema["enum"] = list(CONTENT_KINDS)


def _bind_content_capability(
    properties: dict[str, Any],
    context: Mapping[str, Any],
) -> None:
    fact_schema = properties.get("fact_type")
    entity = context.get("entity")
    kind = entity.get("kind") if isinstance(entity, Mapping) else None
    if not isinstance(fact_schema, dict) or kind not in CONTENT_KIND_TO_FACT_TYPE:
        raise DesignGenerationSchemaError(f"CONTENT_CAPABILITY_KIND_INVALID: {kind!r}")
    fact_schema["enum"] = [CONTENT_KIND_TO_FACT_TYPE[kind].value]


def _bind_requested_property(
    properties: dict[str, Any],
    context: Mapping[str, Any],
) -> None:
    requested_property = context.get("requested_property")
    property_schema = properties.get("property")
    if isinstance(requested_property, str) and isinstance(property_schema, dict):
        property_schema["enum"] = [requested_property]


def context_bound_record_schema(
    identifier: str,
    schema: dict[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact JSON schema that both generation and host validation use."""
    bound = deepcopy(schema)
    properties = bound.get("properties")
    if not isinstance(properties, dict):
        return bound

    _bind_context_enums(properties, context)
    if identifier == "design/content_entity":
        _bind_content_entity_kind(properties)
    elif identifier == "design/content_capability":
        _bind_content_capability(properties, context)
    elif identifier == "design/content_property":
        _bind_requested_property(properties, context)

    Draft202012Validator.check_schema(bound)
    return bound


__all__ = ["DesignGenerationSchemaError", "context_bound_record_schema"]
