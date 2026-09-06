"""Lossless transport repair and strict validation of one research decision."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator

from .research_requirement_template import FACET_AUGMENTATION_RESPONSE_SCHEMA

_VALIDATOR = Draft202012Validator(FACET_AUGMENTATION_RESPONSE_SCHEMA)


def decode_facet_response(raw: Any) -> dict[str, Any]:
    value = dict(raw) if isinstance(raw, Mapping) else json.loads(str(raw))
    if not isinstance(value, dict):
        raise ValueError("facet response must be an object")
    for field in ("evidence_refs", "implementation_obligations", "acceptance"):
        encoded = value.get(field)
        if isinstance(encoded, str) and encoded.strip().startswith("["):
            decoded = json.loads(encoded)
            if isinstance(decoded, list) and all(
                isinstance(item, str) for item in decoded
            ):
                value[field] = decoded
    errors = sorted(
        _VALIDATOR.iter_errors(value), key=lambda error: str(list(error.path))
    )
    if errors:
        raise ValueError(
            "; ".join(f"{list(error.path)}: {error.message}" for error in errors)
        )
    return value
