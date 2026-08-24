from __future__ import annotations

"""Deterministic parser repair for Qwen tagged tool output.

Qwen's tagged fallback format places tool names and scalar values directly inside tagged
blocks. Small local models occasionally serialize string enums with harmless formatting
drift or emit the canonical permission name instead of the narrower model-facing alias.

This module performs only deterministic, request-scoped canonicalization. It never
invents semantic enum aliases, never exposes a broader host schema, and never asks the
model to generate again. One parser hook owns both recoveries so the runtime mutation
surface does not grow for closely related Qwen syntax handling.
"""

import hashlib
import json
import re
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_qwen_parser_recovery_v4"
_NO_MATCH = object()
_MAX_VALUE_PREVIEW = 160


class QwenEnumValueError(RuntimeError):
    """A parsed Qwen tool parameter did not match its declared enum."""

    def __init__(
        self,
        *,
        tool_name: str,
        parameter_name: str,
        raw_value: str,
        allowed_values: Sequence[Any],
    ) -> None:
        self.tool_name = tool_name
        self.parameter_name = parameter_name
        self.raw_value = raw_value
        self.allowed_values = tuple(allowed_values)
        preview = raw_value[:_MAX_VALUE_PREVIEW]
        if len(raw_value) > _MAX_VALUE_PREVIEW:
            preview += "..."
        super().__init__(
            f"Qwen tool {tool_name!r} emitted value outside enum for parameter "
            f"{parameter_name!r}: raw={preview!r}; allowed={list(self.allowed_values)!r}"
        )


def _enum_key(value: str) -> str:
    value = value.strip()
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[\s-]+", "_", value.casefold())
    return re.sub(r"_+", "_", value)


def _string_candidates(raw: str) -> tuple[str, ...]:
    candidates = [raw, raw.strip()]
    compact = raw.strip()
    if len(compact) >= 2 and compact[0] == compact[-1] == '"':
        try:
            decoded = json.loads(compact)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, str):
            candidates.append(decoded)
    return tuple(dict.fromkeys(candidates))


def canonical_string_enum(raw: str, allowed_values: Sequence[Any]) -> Any:
    """Return one formatting-equivalent enum member, or ``_NO_MATCH``."""

    if not allowed_values or not all(isinstance(value, str) for value in allowed_values):
        return _NO_MATCH

    allowed = tuple(str(value) for value in allowed_values)
    for candidate in _string_candidates(raw):
        if candidate in allowed:
            return candidate
        key = _enum_key(candidate)
        matches = tuple(value for value in allowed if _enum_key(value) == key)
        if len(matches) == 1:
            return matches[0]
    return _NO_MATCH


def _relax_string_enums(
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    relaxed: dict[str, Mapping[str, Any]] = dict(schemas)
    for tool_name, schema in schemas.items():
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            continue
        updated_properties = dict(properties)
        changed = False
        for key, property_schema in properties.items():
            if not isinstance(property_schema, Mapping):
                continue
            enum = property_schema.get("enum")
            if not isinstance(enum, list) or not enum or not all(
                isinstance(value, str) for value in enum
            ):
                continue
            updated = dict(property_schema)
            updated.pop("enum", None)
            updated_properties[key] = updated
            changed = True
        if changed:
            updated_schema = dict(schema)
            updated_schema["properties"] = updated_properties
            relaxed[tool_name] = updated_schema
    return relaxed


def _canonicalize_call_enums(
    call: Any,
    schema: Mapping[str, Any],
    *,
    call_index: int,
) -> Any:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return call
    arguments = dict(call.arguments)
    changed = False
    for key, value in tuple(arguments.items()):
        property_schema = properties.get(key)
        if not isinstance(property_schema, Mapping):
            continue
        enum = property_schema.get("enum")
        if not isinstance(enum, list) or not enum or not all(
            isinstance(item, str) for item in enum
        ):
            continue
        if not isinstance(value, str):
            raise QwenEnumValueError(
                tool_name=call.name,
                parameter_name=key,
                raw_value=str(value),
                allowed_values=enum,
            )
        canonical = canonical_string_enum(value, enum)
        if canonical is _NO_MATCH:
            raise QwenEnumValueError(
                tool_name=call.name,
                parameter_name=key,
                raw_value=value,
                allowed_values=enum,
            )
        if canonical != value:
            arguments[key] = canonical
            changed = True
    if not changed:
        return call
    raw_arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(
        f"{call_index}\0{call.name}\0{raw_arguments}".encode("utf-8")
    ).hexdigest()[:16]
    return replace(
        call,
        id=f"call_{digest}",
        arguments=arguments,
        raw_arguments=raw_arguments,
    )


def install(llama_cpp_module: Any) -> None:
    """Install one deterministic parser owner for tool names and string enums."""

    current_parse = getattr(llama_cpp_module, "_parse_qwen_function", None)
    if not callable(current_parse) or bool(getattr(current_parse, _MARKER, False)):
        return

    @wraps(current_parse)
    def parse_qwen_function(
        text: str,
        start: int,
        schemas: Mapping[str, Mapping[str, Any]],
        *,
        call_index: int,
    ):
        function_open = str(getattr(llama_cpp_module, "_FUNCTION_OPEN", "<function="))
        name_start = start + len(function_open)
        name_end = text.find(">", name_start)
        emitted_name = text[name_start:name_end].strip() if name_end >= 0 else ""
        exposed_name = emitted_name
        rewritten = text
        length_delta = 0

        if emitted_name and emitted_name not in schemas:
            from minecraft_mod_ai.model_tool_aliases import resolve_exposed_model_tool

            resolved = resolve_exposed_model_tool(emitted_name, schemas.keys())
            if resolved is not None:
                exposed_name = resolved
                raw_name = text[name_start:name_end]
                rewritten = text[:name_start] + exposed_name + text[name_end:]
                length_delta = len(exposed_name) - len(raw_name)

        relaxed = _relax_string_enums(schemas)
        call, rewritten_end = current_parse(
            rewritten,
            start,
            relaxed,
            call_index=call_index,
        )
        original_schema = schemas.get(call.name)
        if isinstance(original_schema, Mapping):
            call = _canonicalize_call_enums(
                call,
                original_schema,
                call_index=call_index,
            )
        return call, rewritten_end - length_delta

    setattr(parse_qwen_function, _MARKER, True)
    llama_cpp_module._parse_qwen_function = parse_qwen_function


__all__ = [
    "QwenEnumValueError",
    "canonical_string_enum",
    "install",
]
