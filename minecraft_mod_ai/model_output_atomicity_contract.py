from __future__ import annotations

"""Global model structured-output template boundary.

Every model-authored structured response must have an explicit, closed schema. Real
models never author JSON syntax directly: the host turns the schema into one forced
function-argument template, receives typed arguments, serializes them, and validates the
result. Machine-owned JSON remains valid for storage and transport.
"""

import json
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from .execution_contract_policy import (
    DEFAULT_ATOMIC_SCHEMA_LIMITS,
    DEFAULT_SCHEMA_PROFILE,
    MODEL_MAX_COMPLETION_TOKENS,
    SCHEMA_CONTRACT_PROFILE_KEY,
    SCHEMA_STRING_CLASS_KEY,
    STRING_CLASS_GENERIC,
    atomic_schema_limits,
    string_limit_for_schema_class,
)

_INSTALLED = False
_TEXT_MARKER = "_mmm_atomic_model_output_boundary"
_TOOL_MARKER = "_mmm_atomic_model_tool_boundary"
_TEMPLATE_TOOL_NAME = "submit_fixed_template"
_SAME_INSTANCE_CONSTRAINT_KEYWORDS = frozenset(
    {"allOf", "anyOf", "oneOf", "not", "if", "then", "else"}
)

# Compatibility aliases.  Hard limits are owned by execution_contract_policy.
MAX_MODEL_FIELDS = DEFAULT_ATOMIC_SCHEMA_LIMITS.max_fields
MAX_MODEL_STRING_CHARS = DEFAULT_ATOMIC_SCHEMA_LIMITS.max_string_chars
MAX_MODEL_ARRAY_ITEMS = DEFAULT_ATOMIC_SCHEMA_LIMITS.max_array_items
MAX_SCHEMA_DEPTH = DEFAULT_ATOMIC_SCHEMA_LIMITS.max_schema_depth
MAX_COMPLETION_TOKENS = MODEL_MAX_COMPLETION_TOKENS


def _schema_has_type(schema: Mapping[str, Any], expected: str) -> bool:
    raw = schema.get("type")
    if raw == expected:
        return True
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return expected in raw
    return False


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
        has_properties = "properties" in value
        is_object = _schema_has_type(value, "object") or (
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


def assert_strict_atomicity_bounds(
    value: Any,
    *,
    surface: str = "",
    path: str = "$",
    depth: int = 0,
    profile: str = DEFAULT_SCHEMA_PROFILE,
) -> None:
    """Enforce the central atomicity profile selected by the root schema."""
    try:
        limits = atomic_schema_limits(profile)
    except ValueError as exc:
        raise _configuration_error(
            f"MODEL_ATOMICITY_PROFILE_INVALID: {exc} for {surface}"
        ) from exc

    if depth > limits.max_schema_depth:
        raise _configuration_error(
            f"MODEL_ATOMICITY_DEPTH_EXCEEDED: Schema depth {depth} exceeds "
            f"max_schema_depth={limits.max_schema_depth} for profile {profile!r} "
            f"at {path} for {surface}"
        )
    if isinstance(value, Mapping):
        if depth > 0 and SCHEMA_CONTRACT_PROFILE_KEY in value:
            raise _configuration_error(
                "MODEL_ATOMICITY_PROFILE_SCOPE_INVALID: contract profile may be "
                f"declared only at the root schema, not at {path} for {surface}"
            )

        props = value.get("properties")
        if isinstance(props, Mapping):
            if len(props) > limits.max_fields:
                raise _configuration_error(
                    f"MODEL_ATOMICITY_FIELDS_EXCEEDED: Declared {len(props)} properties at {path}, "
                    f"exceeding max_fields={limits.max_fields} for profile {profile!r} "
                    f"for {surface}"
                )
            for k, child in props.items():
                assert_strict_atomicity_bounds(
                    child,
                    surface=surface,
                    path=f"{path}.{k}",
                    depth=depth + 1,
                    profile=profile,
                )
        if _schema_has_type(value, "array"):
            max_items = value.get("maxItems")
            if max_items is None:
                raise _configuration_error(
                    f"MODEL_ATOMICITY_ARRAY_UNBOUNDED: Array schema at {path} must declare "
                    f"'maxItems' <= {limits.max_array_items} for profile {profile!r} "
                    f"for {surface}"
                )
            if max_items > limits.max_array_items:
                raise _configuration_error(
                    f"MODEL_ATOMICITY_ARRAY_EXCEEDED: maxItems={max_items} exceeds "
                    f"max_array_items={limits.max_array_items} for profile {profile!r} "
                    f"at {path} for {surface}"
                )
            if "items" in value and isinstance(value["items"], Mapping):
                item_schema = value["items"]
                item_type = item_schema.get("type")
                item_is_structural = (
                    item_type == "object"
                    or item_type == "array"
                    or "properties" in item_schema
                )
                assert_strict_atomicity_bounds(
                    item_schema,
                    surface=surface,
                    path=f"{path}[]",
                    depth=depth + (1 if item_is_structural else 0),
                    profile=profile,
                )
        for keyword in ("allOf", "anyOf", "oneOf"):
            branches = value.get(keyword)
            if isinstance(branches, Sequence) and not isinstance(
                branches, (str, bytes, bytearray)
            ):
                for index, branch in enumerate(branches):
                    if isinstance(branch, Mapping):
                        assert_strict_atomicity_bounds(
                            branch,
                            surface=surface,
                            path=f"{path}.{keyword}[{index}]",
                            depth=depth,
                            profile=profile,
                        )
        if _schema_has_type(value, "string"):
            string_class = str(
                value.get(SCHEMA_STRING_CLASS_KEY, STRING_CLASS_GENERIC)
                or STRING_CLASS_GENERIC
            )
            try:
                max_string_chars = string_limit_for_schema_class(
                    profile,
                    string_class,
                )
            except ValueError as exc:
                raise _configuration_error(
                    f"MODEL_ATOMICITY_STRING_CLASS_INVALID: {exc} at {path} for {surface}"
                ) from exc
            if "enum" not in value:
                max_len = value.get("maxLength")
                if max_len is None:
                    raise _configuration_error(
                        f"MODEL_ATOMICITY_STRING_UNBOUNDED: String schema at {path} must "
                        f"declare 'maxLength' <= {max_string_chars} for string class "
                        f"{string_class!r} under profile {profile!r} for {surface}"
                    )
                if max_len > max_string_chars:
                    raise _configuration_error(
                        f"MODEL_ATOMICITY_STRING_EXCEEDED: maxLength={max_len} exceeds "
                        f"allowed={max_string_chars} for string class {string_class!r} "
                        f"under profile {profile!r} at {path} for {surface}"
                    )
            else:
                for opt in value.get("enum", ()):
                    if len(str(opt)) > max_string_chars:
                        raise _configuration_error(
                            f"MODEL_ATOMICITY_STRING_EXCEEDED: enum option {opt!r} length "
                            f"exceeds allowed={max_string_chars} for string class "
                            f"{string_class!r} under profile {profile!r} at {path} for {surface}"
                        )


def assert_atomic_model_schema(schema: Mapping[str, Any], *, surface: str) -> None:
    """Require one closed fixed template with strict physical and structural bounds."""

    _assert_closed_object_schemas(schema)
    raw_profile = schema.get(SCHEMA_CONTRACT_PROFILE_KEY, DEFAULT_SCHEMA_PROFILE)
    profile = str(raw_profile or DEFAULT_SCHEMA_PROFILE).strip()
    try:
        atomic_schema_limits(profile)
    except ValueError as exc:
        raise _configuration_error(
            f"MODEL_ATOMICITY_PROFILE_INVALID: {exc} for {surface}"
        ) from exc
    assert_strict_atomicity_bounds(schema, surface=surface, profile=profile)


def is_atomic_model_schema(schema: Mapping[str, Any]) -> bool:
    """Return whether the schema is a closed model-fillable template."""

    try:
        assert_atomic_model_schema(schema, surface="model template")
    except Exception:
        return False
    return True


def _tool_template_schema(
    response_schema: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return function parameters plus whether the host must unwrap ``value``.

    Function parameters are object-shaped. Existing structured callers may legitimately
    request an array/scalar root, so the host wraps only the transport and validates the
    unwrapped value against the caller's original schema.
    """

    schema_type = response_schema.get("type")
    if schema_type == "object" or "properties" in response_schema:
        return dict(response_schema), False
    return (
        {
            "type": "object",
            "properties": {"value": dict(response_schema)},
            "required": ["value"],
            "additionalProperties": False,
        },
        True,
    )


def _semantic_prelude_required(
    self: Any,
    role: str,
    kwargs: Mapping[str, Any],
    model_router_module: Any,
) -> bool:
    """Keep retrieval/tool/media semantics before the final fixed-template fill."""

    if tuple(kwargs.get("media_paths") or ()):
        return True
    enable_tools = bool(kwargs.get("enable_tools", True))
    if not enable_tools:
        return False
    try:
        config = self.registry.role(self.profile, role)
    except Exception:
        return False
    stage = str(
        kwargs.get("tool_stage")
        or model_router_module._ROLE_TOOL_STAGE.get(role, "")
        or ""
    ).strip().lower()
    enabled = getattr(self, "_tools_enabled", None)
    if not callable(enabled):
        return False
    return bool(
        enabled(
            enable_tools=True,
            stage=stage,
            adapter_name=str(getattr(config, "adapter", "") or ""),
        )
    )


def _template_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    semantic_output: str = "",
) -> tuple[dict[str, Any], ...]:
    result = tuple(dict(message) for message in messages)
    if not semantic_output.strip():
        return result
    return (
        *result,
        {
            "role": "system",
            "content": (
                "A semantic/tool/media pass has already completed. Use its result only as "
                "content evidence for the required fixed template. Do not reproduce JSON "
                "syntax or protocol prose yourself.\n\n"
                "Completed semantic result:\n"
                + semantic_output
            ),
        },
    )



def _generate_structured_text_atomically(
    router: Any,
    role: str,
    messages: Sequence[Mapping[str, Any]],
    kwargs: Mapping[str, Any],
    current_text: Any,
    model_router_module: Any,
) -> str:
    response_format = str(kwargs.get("response_format", "text") or "text").strip().casefold()
    if response_format != "json":
        return current_text(router, role, messages, **dict(kwargs))
    response_schema = kwargs.get("response_schema")
    if not isinstance(response_schema, Mapping):
        raise _configuration_error(
            "MODEL_JSON_SCHEMA_REQUIRED: "
            f"JSON response for role {role!r} has no explicit response_schema. "
            "All model-authored structured output must use a fixed template."
        )
    assert_atomic_model_schema(
        response_schema,
        surface=f"JSON response for role {role!r}",
    )
    if role == "planner":
        return current_text(router, role, messages, **dict(kwargs))
    try:
        config = router.registry.role(router.profile, role)
        adapter_name = str(getattr(config, "adapter", "") or "")
    except Exception:
        adapter_name = ""
    if adapter_name == "mock":
        return current_text(router, role, messages, **dict(kwargs))
    semantic_output = ""
    if _semantic_prelude_required(router, role, kwargs, model_router_module):
        semantic_kwargs = dict(kwargs)
        semantic_kwargs["response_format"] = "text"
        semantic_kwargs["response_schema"] = None
        semantic_output = current_text(router, role, messages, **semantic_kwargs)
    parameters, unwrap_value = _tool_template_schema(response_schema)
    arguments = router.generate_tool_decision(
        role,
        _template_messages(messages, semantic_output=semantic_output),
        tool_name=_TEMPLATE_TOOL_NAME,
        parameters=parameters,
        description=(
            "Fill the host-supplied fixed response template exactly once. "
            "Populate only declared fields; do not answer in prose."
        ),
    )
    if unwrap_value:
        if "value" not in arguments:
            raise _configuration_error(
                "MODEL_TEMPLATE_RESULT_INVALID: fixed template call omitted value"
            )
        value: Any = arguments["value"]
    else:
        value = arguments
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    from .structured_output import validate_structured_output
    return validate_structured_output(
        encoded,
        response_format="json",
        response_schema=response_schema,
    )


def _install_router_boundary(model_router_module: Any) -> None:
    """Force every real structured response through fixed function arguments."""

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
            return _generate_structured_text_atomically(
                self, role, messages, kwargs, current_text, model_router_module
            )


        setattr(generate_text, _TEXT_MARKER, True)
        generate_text._mmm_role_specific_structured_transport = True  # type: ignore[attr-defined]
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
            output_token_ceiling: int | None = None,
            force_non_thinking: bool = False,
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
                **(
                    {"output_token_ceiling": output_token_ceiling}
                    if output_token_ceiling is not None else {}
                ),
                **({"force_non_thinking": True} if force_non_thinking else {}),
            )

        setattr(generate_tool_decision, _TOOL_MARKER, True)
        cls.generate_tool_decision = generate_tool_decision


def install(*, model_router_module: Any | None = None) -> None:
    """Idempotently install all model structured-output boundaries."""

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
    if not getattr(
        cls.generate_text, "_mmm_role_specific_structured_transport", False
    ):
        raise RuntimeError("model structured response transport policy is not installed")
    if not getattr(cls.generate_tool_decision, _TOOL_MARKER, False):
        raise RuntimeError("model native-tool template boundary is not installed")


__all__ = [
    "MAX_COMPLETION_TOKENS",
    "MAX_MODEL_ARRAY_ITEMS",
    "MAX_MODEL_FIELDS",
    "MAX_MODEL_STRING_CHARS",
    "MAX_SCHEMA_DEPTH",
    "assert_atomic_model_schema",
    "assert_installed",
    "assert_strict_atomicity_bounds",
    "is_atomic_model_schema",
    "install",
]
