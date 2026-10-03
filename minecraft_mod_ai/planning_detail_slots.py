"""Fixed record layouts for every engineering concern (no model-selected keys)."""

# Each concern is a required array of records with exactly these string fields.
# Empty arrays are allowed only with a concrete reason in inapplicable_concerns.
from .task_template_catalog import detail_records
from .structured_state_runtime import constrain_state_record_schema

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


def _model_transport_schema(schema, *, is_properties_map: bool = False):
    if isinstance(schema, dict):
        return {
            key: _model_transport_schema(value, is_properties_map=(key == "properties"))
            for key, value in schema.items()
            if not (key == "pattern" and not is_properties_map)
        }
    if isinstance(schema, list):
        return [_model_transport_schema(value) for value in schema]
    return schema


def specification_schema(section, *, model_transport: bool = False):
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
        if section == "state_model":
            item_schema = constrain_state_record_schema(concern, item_schema)
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
        return _model_transport_schema(schema)
    return schema


def model_specification_schema(section):
    return specification_schema(section, model_transport=True)


__all__ = [
    "DETAIL_RECORDS",
    "model_specification_schema",
    "record_field_schema",
    "specification_schema",
]
