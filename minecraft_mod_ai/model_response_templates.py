"""Fixed response contracts loaded from the runtime template authority."""
from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache

from jsonschema import Draft202012Validator, ValidationError

from .task_template_catalog import RUNTIME_TEMPLATE_ROOT

_CONTRACTS_PATH = RUNTIME_TEMPLATE_ROOT / "response" / "contracts.json"


@lru_cache(maxsize=1)
def _contracts():
    value = json.loads(_CONTRACTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("RESPONSE_TEMPLATE: contracts.json must contain named schemas")
    for name, schema in value.items():
        if not isinstance(name, str) or not name or not isinstance(schema, dict):
            raise ValueError("RESPONSE_TEMPLATE: invalid named response contract")
        Draft202012Validator.check_schema(schema)
    return value


def response_schema(name):
    try:
        return deepcopy(_contracts()[name])
    except KeyError as exc:
        raise ValueError(f"RESPONSE_TEMPLATE: unknown response contract {name!r}") from exc


def response_template_prompt(name):
    return (
        "Return exactly one JSON value that conforms to this fixed response schema. "
        "Do not return the schema itself. Do not invent, rename, or omit fields. "
        "Populate only the allowed value slots: "
        + json.dumps(response_schema(name), ensure_ascii=False, separators=(",", ":"))
    )


def validate_response_value(name, value):
    """Validate one concrete response value against the named fixed contract."""

    validator = Draft202012Validator(response_schema(name))
    try:
        validator.validate(value)
    except ValidationError as exc:
        raise ValueError(f"RESPONSE_TEMPLATE: {name} output violates fixed schema: {exc.message}") from exc
    return value


def serialize_response(name, value):
    """Serialize a host-owned fixed response only after schema validation."""

    validate_response_value(name, value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_response_text(name, text):
    """Parse exactly one JSON value and validate it against the named fixed contract."""

    if not isinstance(text, str):
        raise ValueError(f"RESPONSE_TEMPLATE: {name} output must be text")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"RESPONSE_TEMPLATE: {name} output is not exactly one JSON value") from exc
    return validate_response_value(name, value)


__all__ = [
    "parse_response_text",
    "response_schema",
    "response_template_prompt",
    "serialize_response",
    "validate_response_value",
]
