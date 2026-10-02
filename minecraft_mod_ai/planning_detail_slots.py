"""Fixed record layouts for every engineering concern (no model-selected keys)."""

# Each concern is a required array of records with exactly these string fields.
# Empty arrays are allowed only with a concrete reason in inapplicable_concerns.
from .task_template_catalog import detail_records

DETAIL_RECORDS = detail_records()


def record_field_schema(section: str, concern: str, field: str) -> dict:
    """Canonical authored-field type contract shared by storage and model paging."""

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
    return {"type": "string", "minLength": 1, "maxLength": 512}


def specification_schema(section):
    records = DETAIL_RECORDS[section]
    properties = {}
    for concern, columns in records.items():
        fields = columns.split()
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
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


__all__ = ["DETAIL_RECORDS", "record_field_schema", "specification_schema"]
