from __future__ import annotations

"""Global small-model structured-output boundary.

Every model-authored JSON response must have an explicit, closed schema. Machine-owned
JSON remains valid for storage and transport, while host-owned containers may still be
large because they are decomposed before model generation. This boundary prevents raw
or open-ended JSON contracts from silently re-entering the model path.
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


def _configuration_error(message: str) -> Exception:
    from .model_adapters import ModelConfigurationError

    return ModelConfigurationError(message)


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


def _assert_closed_object_schemas(value: Any, *, path: str = "$") -> None:
    """Reject object schemas that allow the model to invent undeclared keys."""

    if isinstance(value, Mapping):
        schema_type = value.get("type")
        is_object = schema_type == "object" or "properties" in value
        if is_object:
            properties = value.get("properties")
            if not isinstance(properties, Mapping):
                raise _configuration_error(
                    "MODEL_JSON_TEMPLATE_REQUIRED: "
                    f"object schema at {path} must declare a properties mapping"
                )
            if value.get("additionalProperties") is not False:
                raise _configuration_error(
                    "MODEL_JSON_TEMPLATE_REQUIRED: "
                    f"object schema at {path} must set additionalProperties=false; "
                    "free-form model-authored object keys are forbidden"
                )
        for key, child in value.items():
            _assert_closed_object_schemas(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_closed_object_schemas(child, path=f"{path}[{index}]")


def assert_atomic_model_schema(schema: Mapping[str, Any], *, surface: str) -> None:
    """Require one bounded, closed template for a model-authored JSON payload."""

    _assert_closed_object_schemas(schema)
    encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    nodes, depth, properties = _schema_metrics(schema)
    if (
        len(encoded) > _MAX_SCHEMA_CHARS
        or nodes > _MAX_SCHEMA_NODES
        or depth > _MAX_SCHEMA_DEPTH
        or properties > _MAX_SCHEMA_PROPERTIES
    ):
        raise _configuration_error(
            "MODEL_STRUCTURE_ATOMICITY: "
            f"{surface} is too large for one model-authored structured payload "
            f"(chars={len(encoded)}, nodes={nodes}, depth={depth}, properties={properties}). "
            "The host must own the container and decompose generation into bounded semantic units."
        )


def is_atomic_model_schema(schema: Mapping[str, Any]) -> bool:
    """Return whether one schema is bounded and closed enough for model generation."""

    try:
        _assert_closed_object_schemas(schema)
    except Exception:
        return False
    encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    nodes, depth, properties = _schema_metrics(schema)
    return (
        len(encoded) <= _MAX_SCHEMA_CHARS
        and nodes <= _MAX_SCHEMA_NODES
        and depth <= _MAX_SCHEMA_DEPTH
        and properties <= _MAX_SCHEMA_PROPERTIES
    )


def _install_router_boundary(model_router_module: Any) -> None:
    cls = model_router_module.ModelRouter
    if getattr(cls.generate_text, _MARKER, False):
        return

    current_text = cls.generate_text

    @wraps(current_text)
    def generate_text(
        self: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> str:
        response_format = str(kwargs.get("response_format", "text") or "text").strip().casefold()
        if response_format == "json":
            response_schema = kwargs.get("response_schema")
            if not isinstance(response_schema, Mapping):
                raise _configuration_error(
                    "MODEL_JSON_SCHEMA_REQUIRED: "
                    f"JSON response for role {role!r} has no explicit response_schema. "
                    "All model-authored JSON must use a fixed schema/template."
                )
            assert_atomic_model_schema(
                response_schema,
                surface=f"JSON response for role {role!r}",
            )
        return current_text(self, role, messages, **kwargs)

    setattr(generate_text, _MARKER, True)
    cls.generate_text = generate_text


def install(*, model_router_module: Any | None = None) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if model_router_module is None:
        from . import model_router as model_router_module

    _install_router_boundary(model_router_module)
    _INSTALLED = True


def assert_installed(*, model_router_module: Any | None = None) -> None:
    if model_router_module is None:
        from . import model_router as model_router_module

    if not getattr(model_router_module.ModelRouter.generate_text, _MARKER, False):
        raise RuntimeError("model structured-output template boundary is not installed")


__all__ = [
    "assert_atomic_model_schema",
    "assert_installed",
    "is_atomic_model_schema",
    "install",
]
