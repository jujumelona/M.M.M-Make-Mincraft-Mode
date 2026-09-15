from __future__ import annotations

"""Host-owned fixed-template generation for every structured model response.

Planning writes structured design content directly. Action-producing roles fill forced
function arguments. The deterministic ``mock`` profile keeps its fixture transport.
"""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .model_output_atomicity_contract import assert_atomic_model_schema
from .structured_output import validate_structured_output

_JSON_FIXTURE_FORMAT = "json"
_ROLE_TOOL_STAGE = {
    "planner": "planning",
    "researcher": "research",
    "coder": "generation",
    "coder_safe": "quality",
    "visual_critic": "quality",
}
_DEFAULT_TOOL_NAME = "submit_fixed_template"


def _adapter_name(router: Any, role: str) -> str:
    try:
        config = router.registry.role(router.profile, role)
    except Exception:
        return ""
    return str(getattr(config, "adapter", "") or "")


def _structured_text_transport_required(router: Any, role: str) -> bool:
    return (
        (role == "planner" and callable(getattr(router, "generate_text", None)))
        or _adapter_name(router, role) == "mock"
        or not callable(getattr(router, "generate_tool_decision", None))
    )


def _structured_text_generator(router: Any):
    generate_text = getattr(router, "generate_text", None)
    if not callable(generate_text):
        raise RuntimeError(
            "FIXED_TEMPLATE_TRANSPORT_UNAVAILABLE: router exposes neither "
            "generate_tool_decision nor generate_text"
        )
    return generate_text


def _semantic_prelude_required(
    router: Any,
    role: str,
    *,
    media_paths: Sequence[str | Path],
    tool_stage: str | None,
    enable_tools: bool,
) -> bool:
    if media_paths:
        return True
    if not enable_tools:
        return False
    enabled = getattr(router, "_tools_enabled", None)
    if not callable(enabled):
        return False
    stage = str(tool_stage or _ROLE_TOOL_STAGE.get(role, "") or "").strip().lower()
    return bool(
        enabled(
            enable_tools=True,
            stage=stage,
            adapter_name=_adapter_name(router, role),
        )
    )


def _tool_parameters(response_schema: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    if response_schema.get("type") == "object" or "properties" in response_schema:
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


def _template_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    semantic_output: str,
) -> tuple[dict[str, Any], ...]:
    copied = tuple(dict(message) for message in messages)
    if not semantic_output.strip():
        return copied
    return (
        *copied,
        {
            "role": "system",
            "content": (
                "The semantic/tool/media pass is complete. Use the result only as evidence "
                "when filling the supplied fixed template. Do not reproduce serialization "
                "syntax or protocol prose.\n\nCompleted semantic result:\n"
                + semantic_output
            ),
        },
    )


def _schema_repair_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    failure: BaseException,
    directive: str,
) -> tuple[dict[str, Any], ...]:
    """Feed a rejected schema call back to the model without inventing semantic content."""

    copied = tuple(dict(message) for message in messages)
    return (
        *copied,
        {
            "role": "system",
            "content": (
                "The previous fixed-template function call was rejected by the host schema. "
                "Regenerate the function arguments from the declared schema instead of "
                "repeating the rejected argument shape. Populate every required field, use "
                "only declared fields, and preserve the requested semantics. "
                f"Repair obligation: {directive}. "
                f"Host validation error: {type(failure).__name__}: {failure}"
            ),
        },
    )


def _schema_required_paths(
    schema: Mapping[str, Any],
    *,
    path: str = "$",
) -> tuple[str, ...]:
    """Enumerate every required obligation reachable through required containers."""

    paths: list[str] = []
    schema_type = schema.get("type")
    properties = schema.get("properties")
    required = schema.get("required")

    if (schema_type == "object" or isinstance(properties, Mapping)) and isinstance(
        properties, Mapping
    ):
        required_fields = (
            tuple(str(item) for item in required if isinstance(item, str) and item.strip())
            if isinstance(required, list)
            else ()
        )
        for field in required_fields:
            field_path = f"{path}.{field}"
            paths.append(field_path)
            child = properties.get(field)
            if isinstance(child, Mapping):
                paths.extend(_schema_required_paths(child, path=field_path))
        return tuple(dict.fromkeys(paths))

    if schema_type == "array" or "items" in schema:
        items = schema.get("items")
        if isinstance(items, Mapping):
            paths.extend(_schema_required_paths(items, path=f"{path}[]"))
    return tuple(dict.fromkeys(paths))


def _schema_repair_directives(parameters: Mapping[str, Any]) -> tuple[str, ...]:
    """Build a finite repair frontier directly from all schema-required obligations."""

    required_paths = sorted(
        _schema_required_paths(parameters),
        key=lambda item: (item.count(".") + item.count("[]"), len(item)),
        reverse=True,
    )
    directives = [
        "reconstruct the complete object while explicitly satisfying required schema path "
        f"{required_path!r}"
        for required_path in required_paths
    ]
    directives.append("reconstruct the complete argument object from the schema from scratch")
    return tuple(directives)


def _validate_native_arguments(
    arguments: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run the exact host validator before a native call can leave the repair frontier."""

    encoded = json.dumps(dict(arguments), ensure_ascii=False, separators=(",", ":"))
    validated = validate_structured_output(
        encoded,
        response_format=_JSON_FIXTURE_FORMAT,
        response_schema=parameters,
    )
    value = json.loads(validated)
    if not isinstance(value, Mapping):
        raise ValueError("fixed-template function call did not validate to an argument mapping")
    return value


def _object_field_schemas(
    parameters: Mapping[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Project a failed multi-field object into deterministic one-field repair schemas."""

    properties = parameters.get("properties")
    if parameters.get("type") != "object" or not isinstance(properties, Mapping):
        return ()
    if len(properties) <= 1:
        return ()

    required = parameters.get("required")
    required_fields = {
        str(item)
        for item in required
        if isinstance(required, list) and isinstance(item, str)
    }
    projected: list[tuple[str, dict[str, Any]]] = []
    for raw_name, raw_schema in properties.items():
        if not isinstance(raw_name, str) or not isinstance(raw_schema, Mapping):
            return ()
        field_schema: dict[str, Any] = {
            "type": "object",
            "properties": {raw_name: deepcopy(dict(raw_schema))},
            "required": [raw_name] if raw_name in required_fields else [],
            "additionalProperties": False,
        }
        for keyword in ("$defs", "definitions"):
            definitions = parameters.get(keyword)
            if isinstance(definitions, Mapping):
                field_schema[keyword] = deepcopy(dict(definitions))
        assert_atomic_model_schema(
            field_schema,
            surface=f"fixed-template isolated field {raw_name!r}",
        )
        projected.append((raw_name, field_schema))
    return tuple(projected)


def _safe_tool_component(value: str) -> str:
    safe = "".join(character if character.isalnum() or character == "_" else "_" for character in value)
    return safe.strip("_") or "field"


def _generate_native_arguments_by_field(
    router: Any,
    role: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    tool_name: str,
    parameters: Mapping[str, Any],
    description: str,
    failure: BaseException,
) -> Mapping[str, Any]:
    """Recover a failed multi-field call without regenerating the same invalid object shape."""

    field_schemas = _object_field_schemas(parameters)
    if not field_schemas:
        raise ValueError("fixed-template schema cannot be isolated by top-level field")

    merged: dict[str, Any] = {}
    for field_index, (field_name, field_schema) in enumerate(field_schemas, start=1):
        field_messages = _schema_repair_messages(
            messages,
            failure=failure,
            directive=(
                f"recover only top-level field {field_name!r}; do not emit sibling fields"
            ),
        )
        field_tool_name = (
            f"{tool_name}_field_{field_index}_{_safe_tool_component(field_name)}"
        )
        arguments = router.generate_tool_decision(
            role,
            field_messages,
            tool_name=field_tool_name,
            parameters=field_schema,
            description=(
                description
                + f" Isolated fixed-template recovery for field {field_name!r}. "
                "Populate only the declared field."
            ),
        )
        if not isinstance(arguments, Mapping):
            raise ValueError(
                f"fixed-template isolated field {field_name!r} did not return an argument mapping"
            )
        validated = _validate_native_arguments(arguments, field_schema)
        merged.update(validated)

    return _validate_native_arguments(merged, parameters)


def _generate_native_template_arguments(
    router: Any,
    role: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    tool_name: str,
    parameters: Mapping[str, Any],
    description: str,
) -> Mapping[str, Any]:
    """Generate host-valid tool arguments using bounded, schema-derived recovery."""

    initial_messages = tuple(dict(message) for message in messages)
    try:
        arguments = router.generate_tool_decision(
            role,
            initial_messages,
            tool_name=tool_name,
            parameters=parameters,
            description=description,
        )
        if not isinstance(arguments, Mapping):
            raise ValueError("fixed-template function call did not return an argument mapping")
        return _validate_native_arguments(arguments, parameters)
    except Exception as initial_error:
        if role == "planner":
            raise
        # A multi-field schema can enter a deterministic invalid attractor when one field is
        # repeatedly malformed. Do not regenerate the same object again. Project the failed
        # object into one-field forced calls, merge the host-validated fields, then validate
        # the complete object exactly once.
        if _object_field_schemas(parameters):
            try:
                return _generate_native_arguments_by_field(
                    router,
                    role,
                    initial_messages,
                    tool_name=tool_name,
                    parameters=parameters,
                    description=description,
                    failure=initial_error,
                )
            except Exception as field_error:
                raise RuntimeError(
                    "FIXED_TEMPLATE_SCHEMA_REPAIR_FRONTIER_EXHAUSTED: model could not satisfy "
                    "the host schema after deterministic field-isolated recovery"
                ) from field_error

        last_error: BaseException = initial_error
        current_messages = initial_messages
        directives = _schema_repair_directives(parameters)
        for repair_index, directive in enumerate(directives, start=1):
            current_messages = _schema_repair_messages(
                messages,
                failure=last_error,
                directive=directive,
            )
            try:
                arguments = router.generate_tool_decision(
                    role,
                    current_messages,
                    tool_name=f"{tool_name}_repair_{repair_index}",
                    parameters=parameters,
                    description=(
                        description
                        + " The prior function arguments failed host schema validation. "
                        + directive
                        + "."
                    ),
                )
                if not isinstance(arguments, Mapping):
                    raise ValueError(
                        "fixed-template function call did not return an argument mapping"
                    )
                return _validate_native_arguments(arguments, parameters)
            except Exception as exc:
                last_error = exc

    raise RuntimeError(
        "FIXED_TEMPLATE_SCHEMA_REPAIR_FRONTIER_EXHAUSTED: model could not satisfy every "
        "host-schema obligation across the schema-derived repair frontier"
    ) from last_error


def generate_fixed_template_value(
    router: Any,
    role: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    response_schema: Mapping[str, Any],
    media_paths: Sequence[str | Path] = (),
    tool_stage: str | None = None,
    enable_tools: bool = True,
    tool_name: str = _DEFAULT_TOOL_NAME,
    description: str = "",
) -> Any:
    """Return structured data through the role's content or action transport."""

    if not isinstance(response_schema, Mapping):
        raise TypeError("fixed-template generation requires a response_schema mapping")
    assert_atomic_model_schema(response_schema, surface=f"fixed template for role {role!r}")

    # Planning authors structured content without function-call repair. Other roles
    # retain the native action boundary. Mock and text-only routers use text as well.
    if _structured_text_transport_required(router, role):
        generate_text = _structured_text_generator(router)
        fixture_kwargs: dict[str, Any] = {
            "media_paths": media_paths,
            "response_format": _JSON_FIXTURE_FORMAT,
            "response_schema": response_schema,
            "enable_tools": enable_tools,
        }
        # A missing stage means there is no tool-capability route to describe. Omitting the
        # key keeps read-only fixed-template transports inert instead of publishing a
        # misleading ``tool_stage=None`` pseudo-capability to adapters and test routers.
        if tool_stage is not None:
            fixture_kwargs["tool_stage"] = tool_stage
        raw = generate_text(role, messages, **fixture_kwargs)
        extra_evidence_refs = None
        try:
            val = json.loads(raw)
            if (
                isinstance(val, Mapping)
                and "evidence_refs" in val
                and "evidence_refs" not in response_schema.get("properties", {})
            ):
                extra_evidence_refs = val["evidence_refs"]
                val = {k: v for k, v in val.items() if k != "evidence_refs"}
                raw = json.dumps(val, ensure_ascii=False)
        except Exception:
            pass
        validated = validate_structured_output(
            raw,
            response_format=_JSON_FIXTURE_FORMAT,
            response_schema=response_schema,
        )
        out = json.loads(validated)
        if extra_evidence_refs is not None and isinstance(out, dict):
            out["evidence_refs"] = extra_evidence_refs
        return out

    semantic_output = ""
    if _semantic_prelude_required(
        router,
        role,
        media_paths=media_paths,
        tool_stage=tool_stage,
        enable_tools=enable_tools,
    ):
        semantic_output = router.generate_text(
            role,
            messages,
            media_paths=media_paths,
            response_format="text",
            response_schema=None,
            tool_stage=tool_stage,
            enable_tools=enable_tools,
        )

    parameters, unwrap_value = _tool_parameters(response_schema)
    base_messages = _template_messages(messages, semantic_output=semantic_output)
    resolved_description = (
        description.strip()
        or "Fill the host-supplied fixed response template exactly once. Populate only declared fields."
    )
    arguments = _generate_native_template_arguments(
        router,
        role,
        base_messages,
        tool_name=str(tool_name or _DEFAULT_TOOL_NAME),
        parameters=parameters,
        description=resolved_description,
    )
    value: Any
    if unwrap_value:
        if "value" not in arguments:
            raise ValueError("fixed-template function call omitted wrapped value")
        value = arguments["value"]
    else:
        value = arguments

    extra_evidence_refs = None
    if (
        isinstance(value, Mapping)
        and "evidence_refs" in value
        and "evidence_refs" not in response_schema.get("properties", {})
    ):
        extra_evidence_refs = value["evidence_refs"]
        value = {k: v for k, v in value.items() if k != "evidence_refs"}

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    validated = validate_structured_output(
        encoded,
        response_format=_JSON_FIXTURE_FORMAT,
        response_schema=response_schema,
    )
    out = json.loads(validated)
    if extra_evidence_refs is not None and isinstance(out, dict):
        out["evidence_refs"] = extra_evidence_refs
    return out


def generate_fixed_template_text(
    router: Any,
    role: str,
    messages: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> str:
    """Compatibility surface returning host-serialized JSON after fixed-template fill."""

    value = generate_fixed_template_value(router, role, messages, **kwargs)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = ["generate_fixed_template_text", "generate_fixed_template_value"]
