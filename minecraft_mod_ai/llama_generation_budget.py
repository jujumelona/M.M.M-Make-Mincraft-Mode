from __future__ import annotations

"""Finite native llama.cpp generation budgeting.

The backend-specific wrapper delegates general budget ownership to the transport-neutral
``generation_output_budget`` policy. The staged game-design planner's tool-free
``{"section": ...}`` JSON responses additionally receive a schema-derived ceiling: a
small section must not inherit the entire remaining model context as its decode allowance.
Host-selected semantic/retrieval decisions retain the existing bounded function-call page
budget even when Qwen native-tool parsing falls back to schema-constrained JSON.
Generic structured requests and expansive native tool actions retain the existing
output-budget contract.
"""

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from .generation_output_budget import (
    apply_payload_generation_budget,
    generation_output_token_budget,
)
from .model_context_budget import tool_action_token_budget

_MARKER = "_mmm_finite_generation_budget"
_SEMANTIC_DECISION_FIELDS = frozenset(
    {
        "source_clause_index",
        "capability_id",
        "source_anchor",
        "semantic_statement",
        "given",
        "when",
        "then",
    }
)
_SEMANTIC_DECISION_OPTIONAL_FIELDS = frozenset(
    {
        "semantic_type",
        "required_prerequisite_capabilities",
        "optional_prerequisite_capabilities",
    }
)
_RETRIEVAL_DECISION_FIELDS = frozenset(
    {"requirement_id", "depends_on", "search_queries"}
)


def plain_action_token_budget(config: Any) -> int:
    """Return the finite plain-text budget for the active runtime slot."""

    return generation_output_token_budget(config, input_tokens=0, tools=())


def action_token_budget(config: Any, *, constrained_action: bool) -> int:
    """Compatibility helper for callers that do not expose the concrete tool schemas."""

    if constrained_action:
        return tool_action_token_budget(config)
    return plain_action_token_budget(config)


def apply_generation_budget(
    payload: Mapping[str, Any],
    *,
    config: Any,
) -> dict[str, Any]:
    """Apply the common finite output policy; ``-1`` is never a transport value."""

    return apply_payload_generation_budget(payload, config=config)


def _schema_shape(value: Any, *, depth: int = 0) -> tuple[int, int, int, int, int]:
    """Return scalar/array/object/required counts and maximum structural depth."""

    if not isinstance(value, Mapping):
        return 0, 0, 0, 0, depth
    scalar = 0
    arrays = 0
    objects = 0
    required = 0
    max_depth = depth
    schema_type = str(value.get("type", "") or "").strip().casefold()
    if schema_type == "array":
        arrays += 1
        child = value.get("items")
        child_shape = _schema_shape(child, depth=depth + 1)
        scalar += child_shape[0]
        arrays += child_shape[1]
        objects += child_shape[2]
        required += child_shape[3]
        max_depth = max(max_depth, child_shape[4])
        if not isinstance(child, Mapping):
            scalar += 1
    elif schema_type == "object" or isinstance(value.get("properties"), Mapping):
        objects += 1
        raw_required = value.get("required")
        if isinstance(raw_required, Sequence) and not isinstance(
            raw_required, (str, bytes, bytearray)
        ):
            required += len(raw_required)
        properties = value.get("properties")
        if isinstance(properties, Mapping):
            for child in properties.values():
                child_shape = _schema_shape(child, depth=depth + 1)
                scalar += child_shape[0]
                arrays += child_shape[1]
                objects += child_shape[2]
                required += child_shape[3]
                max_depth = max(max_depth, child_shape[4])
        additional = value.get("additionalProperties")
        if isinstance(additional, Mapping):
            child_shape = _schema_shape(additional, depth=depth + 1)
            scalar += child_shape[0]
            arrays += child_shape[1]
            objects += child_shape[2]
            required += child_shape[3]
            max_depth = max(max_depth, child_shape[4])
    elif schema_type:
        scalar += 1
    return scalar, arrays, objects, required, max_depth


def _is_staged_planner_section_schema(schema: Mapping[str, Any]) -> bool:
    """Recognize the game-design section envelope without coupling to a stage name."""

    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != {"section"}:
        return False
    section = properties.get("section")
    if not isinstance(section, Mapping):
        return False
    return str(section.get("type", "") or "").strip().casefold() == "object"


def _planning_decision_schema_kind(schema: Any) -> str:
    """Recognize only MMM's two host-owned planning decision argument contracts."""

    if not isinstance(schema, Mapping):
        return ""
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != {"requirements"}:
        return ""
    requirements = properties.get("requirements")
    if not isinstance(requirements, Mapping) or requirements.get("type") != "array":
        return ""
    item = requirements.get("items")
    if not isinstance(item, Mapping):
        return ""
    item_properties = item.get("properties")
    if not isinstance(item_properties, Mapping):
        return ""
    fields = frozenset(str(key) for key in item_properties)
    if (
        _SEMANTIC_DECISION_FIELDS <= fields <= _SEMANTIC_DECISION_FIELDS | _SEMANTIC_DECISION_OPTIONAL_FIELDS
    ):
        return "semantic"
    if fields == _RETRIEVAL_DECISION_FIELDS:
        return "retrieval"
    return ""


def _planning_decision_json_fallback(request: Any) -> str:
    """Identify argument-only fallback pages after native Qwen tool parsing fails."""

    if getattr(request, "tools", ()) or ():
        return ""
    if str(getattr(request, "response_format", "") or "").strip().casefold() != "json":
        return ""
    return _planning_decision_schema_kind(getattr(request, "response_schema", None))


def structured_response_token_ceiling(request: Any) -> tuple[int, dict[str, int]] | None:
    """Derive a decode ceiling only for staged game-design section JSON.

    Generic JSON generation remains governed by the common runtime budget. Native tool
    actions are also excluded because source edits can require large argument payloads.
    Host planning-decision fallback JSON is handled separately by the same finite tool
    page policy as its native function-call form.

    The coefficients represent serialization capacity, not a fixed stage clamp:
    scalars reserve 320 tokens, arrays 1280, objects 896, required fields 96, and
    each nesting level 256, plus a 768-token envelope for JSON syntax and natural
    language values. Thus broader/deeper section schemas automatically receive more room.
    """

    if getattr(request, "tools", ()) or ():
        return None
    if str(getattr(request, "response_format", "") or "").strip().casefold() != "json":
        return None
    schema = getattr(request, "response_schema", None)
    if not isinstance(schema, Mapping) or not schema:
        return None
    if not _is_staged_planner_section_schema(schema):
        return None
    scalar, arrays, objects, required, depth = _schema_shape(schema)
    ceiling = (
        768
        + scalar * 320
        + arrays * 1280
        + objects * 896
        + required * 96
        + depth * 256
    )
    metrics = {
        "schema_scalars": scalar,
        "schema_arrays": arrays,
        "schema_objects": objects,
        "schema_required": required,
        "schema_depth": depth,
    }
    return max(1024, int(ceiling)), metrics


def install(hardware_module: Any) -> None:
    """Install the common output budget at the llama-server payload boundary."""

    current = hardware_module._server_payload
    if bool(getattr(current, _MARKER, False)):
        return

    @wraps(current)
    def bounded_server_payload(adapter: Any, request: Any) -> dict[str, Any]:
        raw_payload = current(adapter, request)
        bounded = apply_generation_budget(raw_payload, config=adapter.config)

        decision_kind = _planning_decision_json_fallback(request)
        if decision_kind:
            common_budget = max(1, int(bounded.get("max_tokens", 1) or 1))
            page_budget = max(1, int(tool_action_token_budget(adapter.config)))
            effective = min(common_budget, page_budget)
            bounded["max_tokens"] = effective
            print(
                "llama server: planner decision fallback output budget",
                f" kind={decision_kind}",
                f" common={common_budget}",
                f" tool_page={page_budget}",
                f" effective={effective}",
                sep="",
                flush=True,
            )
            return bounded

        derived = structured_response_token_ceiling(request)
        if derived is None:
            return bounded
        structured_ceiling, metrics = derived
        try:
            requested = int(raw_payload.get("max_tokens", 0) or 0)
        except (TypeError, ValueError):
            requested = 0
        common_budget = max(1, int(bounded.get("max_tokens", 1) or 1))
        effective = min(common_budget, structured_ceiling)
        bounded["max_tokens"] = max(1, effective)
        print(
            "llama server: planner section output budget",
            f" requested={requested if requested > 0 else 'dynamic'}",
            f" common={common_budget}",
            f" schema_ceiling={structured_ceiling}",
            f" effective={effective}",
            *(f" {key}={value}" for key, value in metrics.items()),
            sep="",
            flush=True,
        )
        return bounded

    setattr(bounded_server_payload, _MARKER, True)
    hardware_module._server_payload = bounded_server_payload


__all__ = [
    "action_token_budget",
    "apply_generation_budget",
    "install",
    "plain_action_token_budget",
    "structured_response_token_ceiling",
]
