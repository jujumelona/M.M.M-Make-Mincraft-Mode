from __future__ import annotations

"""Global small-model output boundary.

Machine-owned JSON remains valid for storage and transport. Model-authored structure is
kept atomic: large schemas are rejected before generation, and forced-tool recovery stays
on the same native tool wire instead of falling back to free-form JSON.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

_INSTALLED = False
_MAX_SCHEMA_CHARS = 12_000
_MAX_SCHEMA_NODES = 120
_MAX_SCHEMA_DEPTH = 8
_MAX_SCHEMA_PROPERTIES = 32
_MAX_REPAIR_ERROR_CHARS = 900


def _schema_metrics(value: Any, *, depth: int = 0) -> tuple[int, int, int]:
    nodes = 1
    max_depth = depth
    properties = 0
    if isinstance(value, Mapping):
        raw_properties = value.get("properties")
        if isinstance(raw_properties, Mapping):
            properties += len(raw_properties)
        for child in value.values():
            child_nodes, child_depth, child_properties = _schema_metrics(child, depth=depth + 1)
            nodes += child_nodes
            max_depth = max(max_depth, child_depth)
            properties += child_properties
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            child_nodes, child_depth, child_properties = _schema_metrics(child, depth=depth + 1)
            nodes += child_nodes
            max_depth = max(max_depth, child_depth)
            properties += child_properties
    return nodes, max_depth, properties


def assert_atomic_model_schema(schema: Mapping[str, Any], *, surface: str) -> None:
    """Reject a model-authored payload whose structure should be host-decomposed."""

    encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    nodes, depth, properties = _schema_metrics(schema)
    if (
        len(encoded) > _MAX_SCHEMA_CHARS
        or nodes > _MAX_SCHEMA_NODES
        or depth > _MAX_SCHEMA_DEPTH
        or properties > _MAX_SCHEMA_PROPERTIES
    ):
        from .model_adapters import ModelConfigurationError

        raise ModelConfigurationError(
            "MODEL_STRUCTURE_ATOMICITY: "
            f"{surface} is too large for one model-authored structured payload "
            f"(chars={len(encoded)}, nodes={nodes}, depth={depth}, properties={properties}). "
            "The host must own the container and decompose generation into bounded semantic units."
        )


def _repair_message(name: str, error: str = "") -> dict[str, str]:
    suffix = f" Validation: {error[:_MAX_REPAIR_ERROR_CHARS]}" if error else ""
    return {
        "role": "user",
        "content": (
            f"The host already selected the only permitted function: {name}. "
            f"Call {name} exactly once using the supplied native function schema. "
            "Do not answer in prose and do not serialize a replacement JSON document outside the function call. "
            "Correct only the malformed function arguments."
            + suffix
        ),
    }


def _same_tool_repair_request(forced: Any, request: Any, name: str, error: str = "") -> Any:
    narrowed = forced._single_tool_request(request, name)
    messages = tuple(
        dict(message)
        for message in tuple(getattr(narrowed, "messages", ()) or ())
        if isinstance(message, Mapping)
    ) + (_repair_message(name, error),)
    return replace(
        narrowed,
        messages=messages,
        response_format="text",
        response_schema=None,
    )


def _restore_native_support(forced: Any, adapter: Any, request: Any) -> None:
    key = forced._native_probe_cache_key(adapter, request)
    if key is None:
        return
    with forced._NATIVE_PROBE_LOCK:
        forced._NATIVE_PROBE_CACHE[key] = True
        forced._NATIVE_PROBE_NEGATIVE_AT.pop(key, None)
        forced._NATIVE_PROBE_TRANSIENT_AT.pop(key, None)


def _install_forced_tool_boundary(forced: Any) -> None:
    def host_selected_argument_turn(
        current: Any,
        adapter: Any,
        request: Any,
        name: str,
        *,
        prefix: str = "host_action",
    ) -> Any:
        del prefix
        from .model_adapters import ModelConfigurationError

        parameters = forced._parameters(forced._selected_schema(request, name))
        assert_atomic_model_schema(parameters, surface=f"forced tool {name!r}")
        repair_request = _same_tool_repair_request(forced, request, name)
        try:
            turn = current(adapter, repair_request)
        except BaseException as exc:
            cause = getattr(exc, "cause", exc)
            raise ModelConfigurationError(
                f"Host-selected action {name!r} failed its bounded native-tool repair; "
                "free-form JSON fallback is disabled."
            ) from cause
        if forced._contains_exact_call(turn, name):
            _restore_native_support(forced, adapter, request)
            return turn
        raise ModelConfigurationError(
            f"Host-selected action {name!r} repair returned {forced._call_names(turn)}; "
            "free-form JSON fallback is disabled."
        )

    def host_selected_mutation_turn(current: Any, adapter: Any, request: Any, name: str) -> Any:
        return host_selected_argument_turn(current, adapter, request, name, prefix="host_mutation")

    forced.host_selected_argument_turn = host_selected_argument_turn
    forced.host_selected_mutation_turn = host_selected_mutation_turn


def _install_router_boundary(model_router_module: Any) -> None:
    cls = model_router_module.ModelRouter
    marker = "_mmm_atomic_model_output_boundary"
    if getattr(cls.generate_tool_decision, marker, False):
        return

    current_tool = cls.generate_tool_decision

    @wraps(current_tool)
    def generate_tool_decision(
        self: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        tool_name: str,
        parameters: Mapping[str, Any],
        description: str = "",
    ) -> dict[str, Any]:
        assert_atomic_model_schema(parameters, surface=f"tool decision {tool_name!r}")
        return current_tool(
            self,
            role,
            messages,
            tool_name=tool_name,
            parameters=parameters,
            description=description,
        )

    setattr(generate_tool_decision, marker, True)
    cls.generate_tool_decision = generate_tool_decision

    current_text = cls.generate_text

    @wraps(current_text)
    def generate_text(self: Any, role: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        response_format = str(kwargs.get("response_format", "text") or "text").strip().casefold()
        response_schema = kwargs.get("response_schema")
        if response_format == "json" and isinstance(response_schema, Mapping):
            assert_atomic_model_schema(response_schema, surface=f"JSON response for role {role!r}")
        return current_text(self, role, messages, **kwargs)

    setattr(generate_text, marker, True)
    cls.generate_text = generate_text


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from . import forced_tool_execution_contract, model_router

    _install_forced_tool_boundary(forced_tool_execution_contract)
    _install_router_boundary(model_router)
    _INSTALLED = True


__all__ = ["assert_atomic_model_schema", "install"]
