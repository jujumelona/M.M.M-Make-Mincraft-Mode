"""Fixed record layouts for every engineering concern (no model-selected keys)."""

# Each concern is a required array of records with exactly these string fields.
# Empty arrays are allowed only with a concrete reason in inapplicable_concerns.
from .task_template_catalog import detail_records

DETAIL_RECORDS = detail_records()


def specification_schema(section):
    records = DETAIL_RECORDS[section]
    properties = {}
    for concern, columns in records.items():
        fields = columns.split()
        properties[concern] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {field: {"type": "string", "minLength": 1} for field in fields},
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
