"""Constrain one JSON string to an immutable prefix and suffix at sampling time."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _fragment(value: str) -> str:
    # The grammar matches the JSON-encoded value, including escaped Java quotes
    # and newlines. A second encoding quotes that text as a GBNF terminal.
    return json.dumps(json.dumps(value, ensure_ascii=False)[1:-1], ensure_ascii=False)


def constrained_string_grammar(schema: Mapping[str, Any], frame: Mapping[str, Any]) -> str:
    field = frame["field"]
    properties = schema.get("properties", {})
    if (set(properties) != {field} or schema.get("required") != [field]
            or schema.get("additionalProperties") is not False):
        raise ValueError("STRING_FRAME_REQUIRES_ONE_CLOSED_STRING_FIELD")
    value_schema = properties[field]
    prefix, suffix = frame["prefix"], frame["suffix"]
    minimum, maximum = frame["min_length"], frame["max_length"]
    empty = frame.get("empty_value")
    if (value_schema.get("type") != "string" or not isinstance(prefix, str)
            or not isinstance(suffix, str) or not isinstance(minimum, int)
            or not isinstance(maximum, int) or minimum < 0 or maximum < minimum
            or len(prefix) + maximum + len(suffix) > value_schema["maxLength"]
            or (empty is not None and not isinstance(empty, str))):
        raise ValueError("STRING_FRAME_CONTRACT_INVALID")
    content = f"{_fragment(prefix)} char{{{minimum},{maximum}}} {_fragment(suffix)}"
    if empty is not None:
        if not value_schema.get("minLength", 0) <= len(empty) <= value_schema["maxLength"]:
            raise ValueError("STRING_FRAME_EMPTY_VALUE_OUTSIDE_SCHEMA")
        content += " | " + _fragment(empty)
    key = json.dumps(json.dumps(field, ensure_ascii=False), ensure_ascii=False)
    return (
        f'root ::= "{{" ws {key} ws ":" ws "\\\"" content "\\\"" ws "}}" ws\n'
        f"content ::= {content}\n"
        + r'char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})' + "\n"
        + r'ws ::= [ \t\n\r]*' + "\n"
    )
