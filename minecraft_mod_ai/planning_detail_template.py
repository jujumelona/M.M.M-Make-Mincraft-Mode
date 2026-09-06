from __future__ import annotations

"""One evidence-bound engineering worksheet shared by planning and coding."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

DETAIL_FIELDS = {
    "behavior_contract": "Exact actors, trigger, preconditions, inputs, outputs, units, bounds and observable postconditions.",
    "state_model": "Owned state, types, defaults, legal transitions, invariants, reset and lifecycle rules.",
    "algorithm": "Ordered operations, branches, termination, boundary cases and determinism; no vague 'implement behavior'.",
    "integration": "Required platform hooks and call relationships supported by source evidence; distinguish verified symbols from requirements still needing target binding.",
    "authority_and_network": "Client/server ownership, packet direction, validation, synchronization and reconnect behavior; explain inapplicability.",
    "persistence": "Save/load ownership, format, lifetime, migration and malformed-data behavior; explain inapplicability.",
    "resources_and_ui": "Required data/assets, identifiers, resource dependencies and UI interactions; explain inapplicability.",
    "failure_and_limits": "Failure handling, cleanup, concurrency, resource limits and algorithmic cost supported by the intended behavior.",
    "reuse_assessment": "Source locator, relevant implementation, what transfers, what changes, license/dependency/target compatibility and missing evidence. Unverified reuse stays reference-only.",
    "verification": "Given/when/then checks, boundary/failure cases, observation method and expected outcome for each behavioral invariant.",
}

WORKSHEET_SCHEMA = {
    "type": "object",
    "properties": {
        key: {
            "type": "object",
            "description": description,
            "properties": {
                "specification": {"type": "string", "minLength": 1},
                "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            },
            "required": ["specification", "evidence_refs"],
            "additionalProperties": False,
        }
        for key, description in DETAIL_FIELDS.items()
    },
    "required": list(DETAIL_FIELDS),
    "additionalProperties": False,
}


def validate_worksheet(value: Any, allowed_refs: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(DETAIL_FIELDS):
        raise ValueError("DETAILED_PLAN_WORKSHEET: every engineering section must be filled")
    for key, row in value.items():
        if not isinstance(row, Mapping) or not str(row.get("specification") or "").strip():
            raise ValueError(f"DETAILED_PLAN_WORKSHEET: {key} has no specification")
        refs = row.get("evidence_refs")
        if not isinstance(refs, list) or not refs or any(ref not in allowed_refs for ref in refs):
            raise ValueError(f"DETAILED_PLAN_WORKSHEET: {key} lacks grounded evidence")
    return deepcopy(dict(value))
