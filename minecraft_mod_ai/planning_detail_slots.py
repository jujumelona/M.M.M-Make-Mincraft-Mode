"""Fixed record layouts for every engineering concern (no model-selected keys)."""

# Each concern is a required array with host-fixed field names. State-model field
# types come from structured_state_runtime; this module does not redefine them.
from copy import deepcopy

from .task_template_catalog import detail_records
from .structured_state_runtime import (
    state_concern_schema,
)

DETAIL_RECORDS = detail_records()


def record_field_schema(section: str, concern: str, field: str) -> dict:
    """Canonical authored-field type contract shared by storage and model paging."""

    if section == "state_model":
        canonical = state_concern_schema(concern)
        properties = canonical.get("properties")
        if not isinstance(properties, dict) or field not in properties:
            raise ValueError(
                f"Unknown canonical state field: {concern}.{field}"
            )
        return deepcopy(properties[field])

    if (
        section == "integration"
        and concern == "initialization_order"
        and field == "prerequisite"
    ):
        return {"type": ["string", "null"], "minLength": 1, "maxLength": 512}
    if (
        section == "authority_and_network"
        and concern == "synchronization"
        and field == "recipients"
    ):
        return {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {"type": "string", "minLength": 1, "maxLength": 256},
        }
    if (
        section == "persistence"
        and concern == "missing_defaults"
        and field == "default"
    ):
        return {
            "type": ["string", "array", "null"],
            "minLength": 1,
            "maxLength": 256,
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 256},
        }
    if section == "persistence" and concern == "migration":
        if field == "operation":
            return {
                "type": "string",
                "enum": [
                    "preserve",
                    "rename_key",
                    "delete_key",
                    "set_default",
                ],
            }
        if field in {"source_key", "destination_key"}:
            return {
                "type": ["string", "null"],
                "minLength": 1,
                "maxLength": 128,
            }
        if field == "value":
            return {
                "type": [
                    "string",
                    "number",
                    "integer",
                    "boolean",
                    "null",
                ],
                "maxLength": 512,
            }
    return {"type": "string", "minLength": 1, "maxLength": 512}


def specification_schema(
    section,
    *,
    model_transport: bool = False,
    state_symbols=None,
):
    records = DETAIL_RECORDS[section]
    properties = {}
    for concern, columns in records.items():
        fields = columns.split()
        if section == "state_model":
            canonical = state_concern_schema(
                concern,
                allowed_state_symbols=state_symbols,
            )
            canonical_properties = canonical.get("properties")
            if not isinstance(canonical_properties, dict):
                raise ValueError(
                    f"Invalid canonical state concern schema: {concern}"
                )
            missing = [
                field for field in fields
                if field not in canonical_properties
            ]
            if missing:
                raise ValueError(
                    f"Canonical state schema missing fields for {concern}: {missing}"
                )
            item_schema = {
                "type": "object",
                "properties": {
                    field: deepcopy(canonical_properties[field])
                    for field in fields
                },
                "required": fields,
                "additionalProperties": False,
            }
        else:
            item_schema = {
                "type": "object",
                "properties": {
                    field: record_field_schema(section, concern, field)
                    for field in fields
                },
                "required": fields,
                "additionalProperties": False,
            }
        properties[concern] = {
            "type": "array",
            "maxItems": 4,
            "items": item_schema,
        }
    properties["inapplicable_concerns"] = {
        "type": "array",
        "maxItems": max(1, len(records)),
        "items": {
            "type": "object",
            "properties": {
                "concern": {"type": "string", "enum": list(records)},
                "reason": {"type": "string", "minLength": 1, "maxLength": 512},
            },
            "required": ["concern", "reason"],
            "additionalProperties": False,
        },
    }
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    if model_transport:
        from .model_output_atomicity_contract import effective_model_transport_schema

        return effective_model_transport_schema(schema)
    return schema


def model_specification_schema(section):
    return specification_schema(section, model_transport=True)


__all__ = [
    "DETAIL_RECORDS",
    "model_specification_schema",
    "record_field_schema",
    "specification_schema",
]
