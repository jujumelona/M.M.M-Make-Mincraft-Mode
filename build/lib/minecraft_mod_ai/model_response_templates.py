"""Fixed response contracts loaded from the runtime template authority."""
from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache

from jsonschema import Draft202012Validator

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


__all__ = ["response_schema", "response_template_prompt"]
