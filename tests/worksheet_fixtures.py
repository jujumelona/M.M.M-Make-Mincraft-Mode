"""Structured design fixtures shared by planning integration tests."""
from typing import Any

from minecraft_mod_ai.planning_detail_slots import (
    DETAIL_RECORDS,
    specification_schema,
)


def _fixture_value(schema: dict[str, Any], text: str) -> Any:
    description = str(schema.get("description") or "")
    if "Stable ASCII internal state identifier" in description:
        return "stateValue"
    if "Host state-" in description and "DSL" in description:
        if "mutation" in description.casefold():
            if schema.get("type") == "array":
                return [{
                    "target": "stateValue",
                    "operator": "=",
                    "value": {"kind": "literal", "value": 1},
                }]
            return "stateValue = 1"
        if schema.get("type") == "object":
            return {"kind": "literal", "value": True}
        return "true"

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), "null")

    if schema_type == "array":
        item_schema = schema.get("items")
        count = max(1, int(schema.get("minItems", 1) or 1))
        if isinstance(item_schema, dict):
            return [
                _fixture_value(item_schema, f"{text} item {index + 1}")
                for index in range(count)
            ]
        return [text]
    if schema_type == "boolean":
        return True
    if schema_type == "integer":
        return max(1, int(schema.get("minimum", 1) or 1))
    if schema_type == "number":
        return float(schema.get("minimum", 1) or 1)
    if schema_type == "null":
        return None
    return text


def specification(section):
    result = {}
    for concern, columns in DETAIL_RECORDS[section].items():
        schemas = specification_schema(section)["properties"][concern]["items"]["properties"]
        record = {}
        for field in columns.split():
            text = f"{section} {concern} {field}: server owns the observable outcome."
            record[field] = _fixture_value(schemas[field], text)
        result[concern] = [record]
    result["inapplicable_concerns"] = []
    return result


def row(section, refs=()):
    return {"specification": specification(section), "constraint_evidence_refs": list(refs)}
