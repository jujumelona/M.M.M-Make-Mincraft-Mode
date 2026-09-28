"""Fixed record layouts for every engineering concern (no model-selected keys)."""

from copy import deepcopy

from .task_template_catalog import detail_records, load_record_template

DETAIL_RECORDS = detail_records()


def concern_record_schema(section: str, concern: str) -> dict:
    """Return the canonical record schema without weakening field constraints."""
    if section not in DETAIL_RECORDS or concern not in DETAIL_RECORDS[section]:
        raise ValueError(
            f"DETAILED_PLAN_SCHEMA: unknown concern {section}.{concern}"
        )
    identifier = f"feature/{section}/{concern}"
    return deepcopy(load_record_template(identifier)["record_schema"])


def specification_schema(section):
    records = DETAIL_RECORDS[section]
    properties = {
        concern: {
            "type": "array",
            "items": concern_record_schema(section, concern),
        }
        for concern in records
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
    "concern_record_schema",
    "specification_schema",
]
