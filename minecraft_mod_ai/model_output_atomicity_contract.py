from __future__ import annotations

"""Global small-model output atomicity boundary.

Machine-owned JSON remains valid for storage and transport. Model-authored structured
payloads stay bounded. Host-selected actions are decomposed by the host and recovered
through bounded native function calls; large original containers are never handed back
to the model as raw JSON documents.
"""

import json
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_INSTALLED = False
_MAX_SCHEMA_CHARS = 12_000
_MAX_SCHEMA_NODES = 120
_MAX_SCHEMA_DEPTH = 8
_MAX_SCHEMA_PROPERTIES = 32
_MARKER = "_mmm_atomic_model_output_boundary"


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


def _install_forced_tool_boundary(forced: Any) -> None:
    # The original forced-tool module still owns selection, deterministic read recovery,
    # native capability probing, and adapter wrapping. Only its argument recovery entry
    # points are replaced. Crucially, do not reject the original large schema here: the
    # host must be allowed to see it so it can decompose it before any model call.
    from .native_atomic_argument_recovery import install_into

    if forced.host_selected_argument_turn.__module__ == "minecraft_mod_ai.native_atomic_argument_recovery":
        return
    install_into(forced)


def _install_router_boundary(model_router_module: Any) -> None:
    cls = model_router_module.ModelRouter
    if getattr(cls.generate_tool_decision, _MARKER, False):
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

    setattr(generate_tool_decision, _MARKER, True)
    cls.generate_tool_decision = generate_tool_decision

    current_text = cls.generate_text

    @wraps(current_text)
    def generate_text(self: Any, role: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        response_format = str(kwargs.get("response_format", "text") or "text").strip().casefold()
        response_schema = kwargs.get("response_schema")
        if response_format == "json" and isinstance(response_schema, Mapping):
            assert_atomic_model_schema(response_schema, surface=f"JSON response for role {role!r}")
        return current_text(self, role, messages, **kwargs)

    setattr(generate_text, _MARKER, True)
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
