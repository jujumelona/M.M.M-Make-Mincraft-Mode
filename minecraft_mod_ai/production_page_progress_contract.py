"""Keep production-page generation grammar aligned with host validation.

The native structured-output schema previously allowed a production page whose
modules/assets/audio/acceptance_tests were all empty, while the host semantic
validator rejected exactly that page. Small models could therefore complete a
perfectly schema-valid JSON decode and still fail every repair attempt.
"""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import Any

_OUTPUT_ARRAYS = ("modules", "assets", "audio", "acceptance_tests")
_INSTALLED = False


def _require_concrete_production_output(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a schema that requires at least one concrete output category."""

    variants: list[dict[str, Any]] = []
    for field in _OUTPUT_ARRAYS:
        variant = deepcopy(schema)
        properties = variant.get("properties")
        if not isinstance(properties, dict):
            return schema
        output_schema = properties.get(field)
        if not isinstance(output_schema, dict) or output_schema.get("type") != "array":
            return schema
        output_schema["minItems"] = 1
        variants.append(variant)
    return {"anyOf": variants}


def install() -> None:
    """Install the production-only schema invariant and restore retry parity."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import planner_json_runtime_contract as runtime
    from . import production_stream_efficiency_contract as stream

    original_schema_for_contract = runtime._schema_for_contract
    if not getattr(original_schema_for_contract, "_mmm_production_progress_schema", False):

        @wraps(original_schema_for_contract)
        def schema_for_contract(view: dict[str, Any]) -> dict[str, Any]:
            schema = original_schema_for_contract(view)
            if frozenset(view) != runtime._PRODUCTION_FIELDS:
                return schema
            return _require_concrete_production_output(schema)

        schema_for_contract._mmm_production_progress_schema = True  # type: ignore[attr-defined]
        runtime._schema_for_contract = schema_for_contract

    # The canonical planner runtime owns this budget. The stream optimization
    # layer must not silently reduce production retries (it previously used 2
    # while the canonical production budget defaults to 5).
    production_budget = runtime._attempt_budget(True)
    stream._FULL_PAGE_DECODE_LIMIT = max(
        int(getattr(stream, "_FULL_PAGE_DECODE_LIMIT", 0) or 0),
        production_budget,
    )

    _INSTALLED = True
