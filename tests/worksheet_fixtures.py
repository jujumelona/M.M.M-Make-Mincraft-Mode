"""Structured design fixtures shared by planning integration tests."""
from typing import Any

from minecraft_mod_ai.planning_detail_slots import (
    DETAIL_RECORDS,
    specification_schema,
)


def _fixture_value(schema: dict[str, Any], text: str) -> Any:
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    for union_key in ("oneOf", "anyOf"):
        branches = schema.get(union_key)
        if isinstance(branches, list) and branches:
            branch = next(
                (item for item in branches if isinstance(item, dict)),
                None,
            )
            if branch is not None:
                return _fixture_value(branch, text)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), "null")

    description = str(schema.get("description") or "")
    pattern = str(schema.get("pattern") or "")
    if (
        "Stable ASCII internal state identifier" in description
        or "Stable ASCII identifier" in description
        or pattern == r"^[A-Za-z_][A-Za-z0-9_]*$"
    ):
        return "stateValue"
    normalized_description = description.casefold()
    if (
        ("host state-" in normalized_description and "dsl" in normalized_description)
        or "canonical typed state-mutation ir" in normalized_description
    ):
        if "mutation" in normalized_description or schema_type == "array":
            # The canonical fixture's first state-variable type is boolean. Keep the
            # generated mutation semantically compatible with that declared symbol.
            return [{
                "target": "stateValue",
                "operator": "=",
                "value": {"kind": "literal", "value": True},
            }]
        return {"kind": "literal", "value": True}

    if schema_type == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if isinstance(properties, dict) and isinstance(required, list):
            return {
                str(field): _fixture_value(
                    properties[str(field)],
                    f"{text} {field}",
                )
                for field in required
                if isinstance(properties.get(str(field)), dict)
            }
        return {}
    if schema_type == "array":
        item_schema = schema.get("items")
        count = max(1, int(schema.get("minItems", 1) or 1))
        if int(schema.get("maxItems", count) or count) == 0:
            return []
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

    if pattern in {
        r"^-?[0-9]+(\.[0-9]+)?$",
        r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?$",
    }:
        return "1"
    value = text
    max_length = schema.get("maxLength")
    if type(max_length) is int and max_length >= 0:
        value = value[:max_length]
    min_length = schema.get("minLength")
    if type(min_length) is int and len(value) < min_length:
        value += "x" * (min_length - len(value))
    return value


def specification(section):
    result = {}
    for concern, columns in DETAIL_RECORDS[section].items():
        schemas = specification_schema(section)["properties"][concern]["items"]["properties"]
        record = {}
        for field in columns.split():
            text = f"{section} {concern} {field}: server owns the observable outcome."
            record[field] = _fixture_value(schemas[field], text)
        if section == "state_model" and concern == "variables":
            defaults = {
                "boolean": "false",
                "int": "0",
                "long": "0",
                "double": "0",
                "string": "",
            }
            record["default"] = defaults[str(record["type"])]
        result[concern] = [record]
    result["inapplicable_concerns"] = []
    return result


def row(section, refs=()):
    return {"specification": specification(section), "constraint_evidence_refs": list(refs)}
