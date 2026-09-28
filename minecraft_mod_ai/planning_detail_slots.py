"""Fixed record layouts for every engineering concern (no model-selected keys)."""

from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from .task_template_catalog import detail_records, load_record_template

DETAIL_RECORDS = detail_records()


def concern_leaf_schemas(section: str, concern: str) -> dict[str, dict[str, Any]]:
    """Return the canonical schema for each flattened required leaf field."""
    if section not in DETAIL_RECORDS or concern not in DETAIL_RECORDS[section]:
        raise ValueError(
            f"DETAILED_PLAN_SCHEMA: unknown concern {section}.{concern}"
        )

    identifier = f"feature/{section}/{concern}"
    schema = load_record_template(identifier)["record_schema"]
    result: dict[str, dict[str, Any]] = {}

    def visit(node: Mapping[str, Any], path: tuple[str, ...]) -> None:
        properties = node.get("properties")
        required = node.get("required")
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise ValueError(
                f"DETAILED_PLAN_SCHEMA: {identifier} has invalid object schema "
                f"at {'.'.join(path) or '<root>'}"
            )
        for raw_field in required:
            field = str(raw_field)
            child = properties.get(field)
            if not isinstance(child, Mapping):
                raise ValueError(
                    f"DETAILED_PLAN_SCHEMA: {identifier} missing required field {field!r}"
                )
            if child.get("type") == "object":
                visit(child, (*path, field))
                continue
            if field in result:
                raise ValueError(
                    f"DETAILED_PLAN_SCHEMA: {identifier} has duplicate leaf field {field!r}"
                )
            result[field] = deepcopy(dict(child))

    visit(schema, ())
    expected = tuple(DETAIL_RECORDS[section][concern].split())
    if tuple(result) != expected:
        raise ValueError(
            f"DETAILED_PLAN_SCHEMA: {identifier} leaf schema order {tuple(result)!r} "
            f"does not match record layout {expected!r}"
        )
    return result


def specification_schema(section):
    records = DETAIL_RECORDS[section]
    properties = {}
    for concern, columns in records.items():
        fields = columns.split()
        leaf_schemas = concern_leaf_schemas(section, concern)
        properties[concern] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    field: deepcopy(leaf_schemas[field])
                    for field in fields
                },
                "required": fields,
                "additionalProperties": False,
            },
        }
    properties["inapplicable_concerns"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "concern": {"type": "string", "enum": list(records)},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["concern", "reason"],
            "additionalProperties": False,
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


__all__ = [
    "DETAIL_RECORDS",
    "concern_leaf_schemas",
    "specification_schema",
]
