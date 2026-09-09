from __future__ import annotations

"""Global model structured-output template boundary.

Every model-authored JSON response must have an explicit, closed schema. Machine-owned
JSON remains valid for storage and transport, and generation paths may group semantic
work units as needed. This boundary prevents raw or open-ended JSON contracts from
silently re-entering the model path.
"""

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_INSTALLED = False
_TEXT_MARKER = "_mmm_atomic_model_output_boundary"
_TOOL_MARKER = "_mmm_atomic_model_tool_boundary"
_SAME_INSTANCE_CONSTRAINT_KEYWORDS = frozenset(
    {"allOf", "anyOf", "oneOf", "not", "if", "then", "else"}
)


def _configuration_error(message: str) -> Exception:
    from .model_adapters import ModelConfigurationError

    return ModelConfigurationError(message)


def _assert_closed_object_schemas(
    value: Any,
    *,
    path: str = "$",
    scoped_object_constraint: bool = False,
) -> None:
    """Reject object schemas that allow the model to invent undeclared keys.

    JSON Schema applicators such as ``anyOf`` may contain ``properties`` fragments that
    constrain the *same already-closed object instance*. Those fragments are not new
    object schemas and therefore must not be forced to repeat ``additionalProperties``.
    A fragment is only accepted when it is reached from a closed object scope; explicit
    ``type: object`` schemas remain independently required to be closed everywhere.
    """

    if isinstance(value, Mapping):
        schema_type = value.get("type")
        has_properties = "properties" in value
        is_object = schema_type == "object" or (
            has_properties and not scoped_object_constraint
        )

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
            closed_object_scope = True
        elif has_properties:
            properties = value.get("properties")
            if not isinstance(properties, Mapping):
                raise _configuration_error(
                    "MODEL_JSON_TEMPLATE_REQUIRED: "
                    f"constraint fragment at {path} must declare a properties mapping"
                )
            closed_object_scope = scoped_object_constraint
        else:
            closed_object_scope = scoped_object_constraint

        for key, child in value.items():
            child_scope = (
                closed_object_scope
                if key in _SAME_INSTANCE_CONSTRAINT_KEYWORDS
                else False
            )
            _assert_closed_object_schemas(
                child,
                path=f"{path}.{key}",
                scoped_object_constraint=child_scope,
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_closed_object_schemas(
                child,
                path=f"{path}[{index}]",
                scoped_object_constraint=scoped_object_constraint,
            )


def assert_atomic_model_schema(schema: Mapping[str, Any], *, surface: str) -> None:
    """Require a closed template without arbitrary schema-size rejection.

    Schema syntax depth, metadata length, and property counts do not establish whether
    the configured model can execute a request. Keep the template contract here; actual
    model context/output capacity is handled by the generation runtime.
    """

    _assert_closed_object_schemas(schema)


def is_atomic_model_schema(schema: Mapping[str, Any]) -> bool:
    """Compatibility predicate for closed model templates, independent of size."""

    try:
        assert_atomic_model_schema(schema, surface="model template")
    except Exception:
        return False
    return True


def _install_router_boundary(model_router_module: Any) -> None:
    """Install the same structured-output boundary on text JSON and native tool JSON."""

    cls = model_router_module.ModelRouter

    if not getattr(cls.generate_text, _TEXT_MARKER, False):
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

        setattr(generate_text, _TEXT_MARKER, True)
        cls.generate_text = generate_text

    if not getattr(cls.generate_tool_decision, _TOOL_MARKER, False):
        current_tool_decision = cls.generate_tool_decision

        @wraps(current_tool_decision)
        def generate_tool_decision(
            self: Any,
            role: str,
            messages: Sequence[Mapping[str, Any]],
            *,
            tool_name: str,
            parameters: Mapping[str, Any],
            description: str = "",
        ) -> dict[str, Any]:
            if not isinstance(parameters, Mapping):
                raise _configuration_error(
                    "MODEL_JSON_SCHEMA_REQUIRED: "
                    f"native tool decision for role {role!r} has no explicit parameters schema."
                )
            assert_atomic_model_schema(
                parameters,
                surface=(
                    f"native tool decision {str(tool_name or '').strip()!r} "
                    f"for role {role!r}"
                ),
            )
            return current_tool_decision(
                self,
                role,
                messages,
                tool_name=tool_name,
                parameters=parameters,
                description=description,
            )

        setattr(generate_tool_decision, _TOOL_MARKER, True)
        cls.generate_tool_decision = generate_tool_decision


def install(*, model_router_module: Any | None = None) -> None:
    """Idempotently install all model structured-output boundaries.

    Per-method markers, rather than the module flag alone, make upgrades safe when a
    process already has one older boundary installed: a newly added surface is still
    wrapped instead of being skipped as globally 'installed'.
    """

    global _INSTALLED
    if model_router_module is None:
        from . import model_router as model_router_module

    _install_router_boundary(model_router_module)
    _INSTALLED = True


def assert_installed(*, model_router_module: Any | None = None) -> None:
    if model_router_module is None:
        from . import model_router as model_router_module

    cls = model_router_module.ModelRouter
    if not getattr(cls.generate_text, _TEXT_MARKER, False):
        raise RuntimeError("model JSON response template boundary is not installed")
    if not getattr(cls.generate_tool_decision, _TOOL_MARKER, False):
        raise RuntimeError("model native-tool template boundary is not installed")


__all__ = [
    "assert_atomic_model_schema",
    "assert_installed",
    "is_atomic_model_schema",
    "install",
]