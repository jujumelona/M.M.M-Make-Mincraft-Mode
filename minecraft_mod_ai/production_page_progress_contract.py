"""Keep production-page generation grammar aligned with host validation.

The native structured-output schema must not admit pages that the host or the
durable item parser will deterministically reject. Keep those semantic
invariants in the grammar so small models do not spend repair rounds on values
that were invalid before parsing began.
"""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import Any

_OUTPUT_ARRAYS = ("modules", "assets", "audio", "acceptance_tests")
_NON_EMPTY_MODULE_FIELDS = ("module_id", "kind")
_NON_EMPTY_MODULE_ARRAY_FIELDS = (
    "depends_on",
    "required_gates",
    "implements_deliverables",
)
_NON_EMPTY_PAGE_ARRAY_FIELDS = ("acceptance_tests", "completed_deliverables")
_PRODUCTION_CHECKPOINT_VERSION = 2
_INSTALLED = False


def _require_non_empty_string(schema: Any) -> None:
    if isinstance(schema, dict) and schema.get("type") == "string":
        schema["minLength"] = max(1, int(schema.get("minLength", 0) or 0))


def _require_non_empty_string_items(schema: Any) -> None:
    if not isinstance(schema, dict) or schema.get("type") != "array":
        return
    _require_non_empty_string(schema.get("items"))


def _align_durable_item_semantics(schema: dict[str, Any]) -> dict[str, Any]:
    """Tighten generation-only strings that durable parsing rejects when blank."""

    aligned = deepcopy(schema)
    properties = aligned.get("properties")
    if not isinstance(properties, dict):
        return aligned

    modules = properties.get("modules")
    module_item = modules.get("items") if isinstance(modules, dict) else None
    module_properties = (
        module_item.get("properties") if isinstance(module_item, dict) else None
    )
    if isinstance(module_properties, dict):
        for field in _NON_EMPTY_MODULE_FIELDS:
            _require_non_empty_string(module_properties.get(field))
        for field in _NON_EMPTY_MODULE_ARRAY_FIELDS:
            _require_non_empty_string_items(module_properties.get(field))

    for field in _NON_EMPTY_PAGE_ARRAY_FIELDS:
        _require_non_empty_string_items(properties.get(field))

    return aligned


def _require_concrete_production_output(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a schema that requires at least one concrete output category."""

    aligned = _align_durable_item_semantics(schema)
    variants: list[dict[str, Any]] = []
    for field in _OUTPUT_ARRAYS:
        variant = deepcopy(aligned)
        properties = variant.get("properties")
        if not isinstance(properties, dict):
            return aligned
        output_schema = properties.get(field)
        if not isinstance(output_schema, dict) or output_schema.get("type") != "array":
            return aligned
        output_schema["minItems"] = 1
        variants.append(variant)
    return {"anyOf": variants}


def install() -> None:
    """Install production schema invariants, retry parity, and checkpoint epoch."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import planner_json_runtime_contract as runtime
    from . import production_page_durable_contract as durable
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

    # A page saved under the older, looser grammar can bypass generation on resume.
    # Move the checkpoint epoch forward so schema-invalid saved pages are never
    # replayed into the durable parser after this contract change.
    durable._VERSION = max(
        int(getattr(durable, "_VERSION", 0) or 0),
        _PRODUCTION_CHECKPOINT_VERSION,
    )

    # The canonical planner runtime owns this budget. The stream optimization
    # layer must not silently reduce production retries.
    production_budget = runtime._attempt_budget(True)
    stream._FULL_PAGE_DECODE_LIMIT = max(
        int(getattr(stream, "_FULL_PAGE_DECODE_LIMIT", 0) or 0),
        production_budget,
    )

    _INSTALLED = True
