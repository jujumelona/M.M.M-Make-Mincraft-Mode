from __future__ import annotations

"""Global model structured-output template boundary.

Every model-authored structured response must have an explicit, closed schema. Real
models never author JSON syntax directly: the host turns the schema into one forced
function-argument template, receives typed arguments, serializes them, and validates the
result. Machine-owned JSON remains valid for storage and transport.
"""

import json
import math
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from .execution_contract_policy import (
    DEFAULT_SCHEMA_PROFILE,
    PLANNER_RECORD_PAGE_OUTPUT_TOKEN_CEILING,
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


_HOST_ONLY_MODEL_CONSTRAINT_KEYWORDS = frozenset({
    # llama.cpp's projected JSON schema does not enforce these semantic
    # cross-value constraints. Model-facing contracts must encode them
    # structurally in host-controlled calls instead of relying on late validation.
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "dependentRequired",
    "dependentSchemas",
    "patternProperties",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
    "multipleOf",
})


def _assert_no_host_only_model_constraints(
    value: Any,
    *,
    surface: str,
    path: str = "$",
) -> None:
    if not isinstance(value, Mapping):
        return

    for keyword in _HOST_ONLY_MODEL_CONSTRAINT_KEYWORDS:
        if keyword in value:
            raise _configuration_error(
                "MODEL_SCHEMA_HOST_ONLY_CONSTRAINT: "
                f"{keyword!r} at {path} for {surface}; "
                "encode this invariant in host-owned structure before inference"
            )

    # Recurse only through JSON-Schema-bearing positions. Keys inside
    # properties/$defs are user-defined names, not schema keywords.
    for container_key in ("properties", "$defs", "definitions"):
        container = value.get(container_key)
        if isinstance(container, Mapping):
            for name, child in container.items():
                if isinstance(child, Mapping):
                    _assert_no_host_only_model_constraints(
                        child,
                        surface=surface,
                        path=f"{path}.{container_key}[{name!r}]",
                    )

    for child_key in (
        "items",
        "additionalProperties",
        "if",
        "then",
        "else",
        "not",
    ):
        child = value.get(child_key)
        if isinstance(child, Mapping):
            _assert_no_host_only_model_constraints(
                child,
                surface=surface,
                path=f"{path}.{child_key}",
            )

    for list_key in ("oneOf", "anyOf", "allOf", "prefixItems"):
        children = value.get(list_key)
        if isinstance(children, Sequence) and not isinstance(
            children, (str, bytes, bytearray)
        ):
            for index, child in enumerate(children):
                if isinstance(child, Mapping):
                    _assert_no_host_only_model_constraints(
                        child,
                        surface=surface,
                        path=f"{path}.{list_key}[{index}]",
                    )


def assert_strict_atomicity_bounds(
    value: Any,
    *,
    surface: str = "",
    path: str = "$",
    depth: int = 0,
    profile: str = DEFAULT_SCHEMA_PROFILE,
) -> None:
    """Validate schema annotations without imposing arbitrary semantic size limits.

    Model/context/token budgets are transport-resource concerns, not correctness
    properties of a response schema. This validator therefore keeps only profile
    ownership checks here; closed-object safety remains enforced separately.
    """
    try:
        atomic_schema_limits(profile)
    except ValueError as exc:
        raise _configuration_error(
            f"MODEL_ATOMICITY_PROFILE_INVALID: {exc} for {surface}"
        ) from exc

    if isinstance(value, Mapping):
        if depth > 0 and SCHEMA_CONTRACT_PROFILE_KEY in value:
            raise _configuration_error(
                "MODEL_ATOMICITY_PROFILE_SCOPE_INVALID: contract profile may be "
                f"declared only at the root schema, not at {path} for {surface}"
            )
        if SCHEMA_STRING_CLASS_KEY in value:
            string_class = str(value.get(SCHEMA_STRING_CLASS_KEY) or STRING_CLASS_GENERIC)
            try:
                string_limit_for_schema_class(profile, string_class)
            except ValueError as exc:
                raise _configuration_error(
                    f"MODEL_ATOMICITY_STRING_CLASS_INVALID: {exc} at {path} for {surface}"
                ) from exc
        for key, child in value.items():
            if isinstance(child, Mapping):
                assert_strict_atomicity_bounds(
                    child,
                    surface=surface,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                    profile=profile,
                )
            elif isinstance(child, Sequence) and not isinstance(
                child, (str, bytes, bytearray)
            ):
                for index, item in enumerate(child):
                    if isinstance(item, Mapping):
                        assert_strict_atomicity_bounds(
                            item,
                            surface=surface,
                            path=f"{path}.{key}[{index}]",
                            depth=depth + 1,
                            profile=profile,
                        )


def assert_atomic_model_schema(schema: Mapping[str, Any], *, surface: str) -> None:
    """Require a closed, transport-enforceable host-owned model template."""

    _assert_closed_object_schemas(schema)
    _assert_no_host_only_model_constraints(schema, surface=surface)
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


def _integer_transport_bounds(schema: Mapping[str, Any]) -> tuple[int, int]:
    """Intersect the logical range with the existing 19-digit transport envelope."""
    minimum, maximum = -(10**19 - 1), 10**19 - 1
    if "minimum" in schema:
        minimum = max(minimum, math.ceil(schema["minimum"]))
    if "maximum" in schema:
        maximum = min(maximum, math.floor(schema["maximum"]))
    if "exclusiveMinimum" in schema:
        minimum = max(minimum, math.floor(schema["exclusiveMinimum"]) + 1)
    if "exclusiveMaximum" in schema:
        maximum = min(maximum, math.ceil(schema["exclusiveMaximum"]) - 1)
    if minimum > maximum:
        raise ValueError("MODEL_INTEGER_TRANSPORT_EMPTY: integer range has no encodable value")
    return minimum, maximum


def _integer_range_pattern(minimum: int, maximum: int) -> str:
    """Compile an inclusive integer interval to finite decimal alternatives."""

    def digits(low: str, high: str) -> str:
        if low == high:
            return low
        width = len(low)
        if low == "0" * width and high == "9" * width:
            return "[0-9]" if width == 1 else f"[0-9]{{{width}}}"
        if low[0] == high[0]:
            return low[0] + digits(low[1:], high[1:])
        parts = [low[0] + digits(low[1:], "9" * (width - 1))]
        first, last = int(low[0]) + 1, int(high[0]) - 1
        if first <= last:
            head = str(first) if first == last else f"[{first}-{last}]"
            parts.append(head + digits("0" * (width - 1), "9" * (width - 1)))
        parts.append(high[0] + digits("0" * (width - 1), high[1:]))
        return "(?:" + "|".join(parts) + ")"

    def unsigned(low: int, high: int) -> str:
        parts = []
        for width in range(len(str(low)), len(str(high)) + 1):
            start = max(low, 0 if width == 1 else 10 ** (width - 1))
            stop = min(high, 10**width - 1)
            parts.append(digits(str(start), str(stop)))
        return "(?:" + "|".join(parts) + ")"

    parts = []
    if minimum < 0:
        parts.append("-" + unsigned(abs(min(maximum, -1)), abs(minimum)))
    if maximum >= 0:
        parts.append(unsigned(max(0, minimum), maximum))
    return "^(?:" + "|".join(parts) + ")$"


def _model_transport_schema(
    value: Any,
    *,
    profile: str = DEFAULT_SCHEMA_PROFILE,
    string_class: str = STRING_CLASS_GENERIC,
) -> Any:
    """Project a logical schema into a finite model-visible transport contract.

    Logical/storage schemas may intentionally omit generic resource bounds. A model
    decode may not: every model-visible string/array receives a finite transport bound
    unless the logical schema already declares one. Explicit domain bounds are preserved.
    Host-only profile annotations select the appropriate bound but are never sent over
    the wire.
    """

    if isinstance(value, Mapping):
        local_profile = str(
            value.get(SCHEMA_CONTRACT_PROFILE_KEY, profile) or profile
        ).strip()
        limits = atomic_schema_limits(local_profile)
        local_string_class = str(
            value.get(SCHEMA_STRING_CLASS_KEY, string_class) or string_class
        ).strip()
        result = {
            key: _model_transport_schema(
                child,
                profile=local_profile,
                string_class=local_string_class,
            )
            for key, child in value.items()
            if key not in {SCHEMA_CONTRACT_PROFILE_KEY, SCHEMA_STRING_CLASS_KEY}
        }
        if _schema_has_type(value, "string") and "maxLength" not in result:
            result["maxLength"] = string_limit_for_schema_class(
                local_profile,
                local_string_class,
            )
        if _schema_has_type(value, "array") and "maxItems" not in result:
            result["maxItems"] = limits.max_array_items
        if (
            _schema_has_type(value, "integer")
            and "enum" not in result
            and "const" not in result
        ):
            minimum, maximum = _integer_transport_bounds(value)
            raw_type = value.get("type")
            nullable = isinstance(raw_type, (list, tuple)) and "null" in raw_type
            if maximum - minimum <= 256:
                result["enum"] = list(range(minimum, maximum + 1))
                if nullable:
                    result["enum"].append(None)
            else:
                result["type"] = ["string", "null"] if nullable else "string"
                result["pattern"] = _integer_range_pattern(minimum, maximum)
                result["maxLength"] = max(len(str(minimum)), len(str(maximum)))
                for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
                    result.pop(k, None)
        if (
            _schema_has_type(value, "number")
            and "enum" not in result
            and "const" not in result
        ):
            raw_type = value.get("type")
            if isinstance(raw_type, (list, tuple)) and "null" in raw_type:
                result["type"] = ["string", "null"]
            else:
                result["type"] = "string"
            result["pattern"] = r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"
            result["maxLength"] = 32
            for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
                result.pop(k, None)
        if (
            (_schema_has_type(value, "object") or "properties" in value)
            and value.get("additionalProperties") is not False
            and "maxProperties" not in result
        ):
            result["maxProperties"] = limits.max_fields
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _model_transport_schema(
                child,
                profile=profile,
                string_class=string_class,
            )
            for child in value
        ]
    return value


def effective_model_transport_schema(
    value: Any,
    *,
    profile: str = DEFAULT_SCHEMA_PROFILE,
    string_class: str = STRING_CLASS_GENERIC,
) -> Any:
    """Return the *effective* decoder schema: the single schema that both the
    token ceiling verifier and the llama.cpp sampler must use.

    This composes two transformations:

    1. ``_model_transport_schema`` — adds finite bounds (``maxLength``,
       ``maxItems``, ``pattern``, etc.) to unbounded logical schemas.
    2. ``project_llama_transport_schema`` — projects the schema to the
       structural subset that llama.cpp actually enforces.

    The resulting schema is identical for ceiling verification and sampler
    constraint generation, eliminating the dualization bug.
    """
    from .llama_schema_transport import project_llama_transport_schema

    bounded = _model_transport_schema(value, profile=profile, string_class=string_class)
    if isinstance(bounded, Mapping):
        return project_llama_transport_schema(bounded)
    return bounded


_SCHEMA_ANNOTATION_KEYS = frozenset(
    {"title", "description", "$comment", "default", "examples", "$defs", "definitions"}
)


def _resolve_local_schema_ref(
    root: Mapping[str, Any],
    ref: str,
) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"structured schema ref must be local: {ref!r}")
    current: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"structured schema ref is unresolved: {ref!r}")
        current = current[part]
    if not isinstance(current, Mapping):
        raise ValueError(f"structured schema ref is not an object schema: {ref!r}")
    return current


def _schema_max_json_chars(
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    ref_stack: tuple[str, ...] = (),
) -> int:
    """Conservative serialized-JSON upper bound for one structured model page."""

    if "const" in schema:
        return len(
            json.dumps(
                schema["const"],
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return max(
            len(json.dumps(value, ensure_ascii=True, separators=(",", ":")))
            for value in enum
        )

    candidates: list[int] = []

    raw_ref = schema.get("$ref")
    if isinstance(raw_ref, str):
        if raw_ref in ref_stack:
            raise ValueError(f"recursive structured schema ref is not finitely bounded: {raw_ref!r}")
        target = _resolve_local_schema_ref(root, raw_ref)
        ref_bound = _schema_max_json_chars(
            target,
            root=root,
            ref_stack=(*ref_stack, raw_ref),
        )
        sibling_shape = {
            key: value
            for key, value in schema.items()
            if key != "$ref" and key not in _SCHEMA_ANNOTATION_KEYS
        }
        if sibling_shape:
            candidates.append(
                ref_bound
                + _schema_max_json_chars(
                    sibling_shape,
                    root=root,
                    ref_stack=ref_stack,
                )
            )
        else:
            candidates.append(ref_bound)

    for keyword in ("oneOf", "anyOf"):
        branches = schema.get(keyword)
        if isinstance(branches, Sequence) and not isinstance(
            branches, (str, bytes, bytearray)
        ) and branches:
            bounds = [
                _schema_max_json_chars(branch, root=root, ref_stack=ref_stack)
                for branch in branches
                if isinstance(branch, Mapping)
            ]
            if len(bounds) != len(branches):
                raise ValueError(f"{keyword} contains a non-object schema branch")
            candidates.append(max(bounds))

    branches = schema.get("allOf")
    if isinstance(branches, Sequence) and not isinstance(
        branches, (str, bytes, bytearray)
    ) and branches:
        bounds = [
            _schema_max_json_chars(branch, root=root, ref_stack=ref_stack)
            for branch in branches
            if isinstance(branch, Mapping)
        ]
        if len(bounds) != len(branches):
            raise ValueError("allOf contains a non-object schema branch")
        # The same JSON instance satisfies every branch. Summation deliberately
        # overestimates combined fragments so it remains a safe upper bound.
        candidates.append(sum(bounds))

    types = {
        str(item)
        for item in (
            [schema.get("type")]
            if isinstance(schema.get("type"), str)
            else schema.get("type", ())
            if isinstance(schema.get("type"), Sequence)
            and not isinstance(schema.get("type"), (str, bytes, bytearray))
            else ()
        )
        if item
    }

    properties = schema.get("properties")
    if "object" in types or isinstance(properties, Mapping):
        if not isinstance(properties, Mapping):
            raise ValueError("structured object schema must declare properties")
        parts = 2
        for index, (name, child) in enumerate(properties.items()):
            if not isinstance(child, Mapping):
                raise ValueError("structured object property schema must be an object")
            if index:
                parts += 1
            parts += len(json.dumps(str(name), ensure_ascii=True)) + 1
            parts += _schema_max_json_chars(
                child,
                root=root,
                ref_stack=ref_stack,
            )
        candidates.append(parts)

    if "array" in types:
        try:
            max_items = int(schema["maxItems"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("structured array schema must declare finite maxItems") from exc
        items = schema.get("items")
        if not isinstance(items, Mapping):
            raise ValueError("structured array schema must declare item schema")
        item_chars = _schema_max_json_chars(
            items,
            root=root,
            ref_stack=ref_stack,
        )
        candidates.append(2 + max_items * item_chars + max(0, max_items - 1))

    if "string" in types:
        try:
            max_length = int(schema["maxLength"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("structured string schema must declare finite maxLength") from exc
        candidates.append(2 + 6 * max(0, max_length))

    if types & {"integer", "number"}:
        raise ValueError("structured numeric schema has unbounded lexical output")

    if "boolean" in types:
        candidates.append(5)
    if "null" in types:
        candidates.append(4)

    if not candidates:
        raise ValueError(f"structured schema has no bounded serializable type: {schema!r}")
    return max(candidates)


def structured_output_token_ceiling(
    schema: Mapping[str, Any],
    *,
    absolute_ceiling: int = PLANNER_RECORD_PAGE_OUTPUT_TOKEN_CEILING,
) -> int:
    """Prove and return a finite decode ceiling for one structured model page."""

    if not isinstance(schema, Mapping):
        raise TypeError("structured output budget requires a schema mapping")
    max_json_chars = _schema_max_json_chars(schema, root=schema)
    derived = max(64, max_json_chars + 32)
    limit = max(1, int(absolute_ceiling))
    if derived > limit:
        raise ValueError(
            "structured schema exceeds the global atomic output bound; "
            "split the page further before inference: "
            f"derived={derived} limit={limit}"
        )
    return derived


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
            transport_parameters = _model_transport_schema(parameters)
            return current_tool_decision(
                self,
                role,
                messages,
                tool_name=tool_name,
                parameters=transport_parameters,
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
    "assert_atomic_model_schema",
    "assert_installed",
    "assert_strict_atomicity_bounds",
    "effective_model_transport_schema",
    "is_atomic_model_schema",
    "structured_output_token_ceiling",
    "install",
]
