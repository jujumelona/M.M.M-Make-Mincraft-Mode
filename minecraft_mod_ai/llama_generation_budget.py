from __future__ import annotations

"""Finite native llama.cpp generation budgeting.

The backend-specific wrapper delegates general budget ownership to the transport-neutral
``generation_output_budget`` policy. Structured JSON turns additionally receive a
schema-derived ceiling at this boundary: a three-field planner section must not inherit
the entire remaining model context as its decode allowance. The ceiling is computed
from schema shape and is always intersected with the common context-aware budget.
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
        if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes, bytearray)):
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


def structured_response_token_ceiling(request: Any) -> tuple[int, dict[str, int]] | None:
    """Derive a decode ceiling from the host-owned response schema.

    The coefficients represent serialization capacity, not a fixed stage clamp:
    scalars reserve 320 tokens, arrays 1280, objects 896, required fields 96, and
    each nesting level 256, plus a 768-token envelope for JSON syntax and natural
    language values. Thus broader/deeper schemas automatically receive more room.
    """

    if str(getattr(request, "response_format", "") or "").strip().casefold() != "json":
        return None
    schema = getattr(request, "response_schema", None)
    if not isinstance(schema, Mapping) or not schema:
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
            "llama server: structured output budget",
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
