from __future__ import annotations

"""Deterministic parser repair for Qwen tagged tool output.

Qwen's tagged fallback format places tool names and scalar values directly inside tagged
blocks. Small local models occasionally serialize string enums with harmless formatting
drift or emit the canonical permission name instead of the narrower model-facing alias.

This module performs only deterministic, request-scoped canonicalization. It never
invents semantic enum aliases, never exposes a broader host schema, and never asks the
model to generate again. Malformed calls that cannot be repaired without changing
meaning are raised as typed protocol errors; causal_stale_tool_recovery_contract remains
the sole owner of bounded re-synchronization.
"""

import json
import re
from functools import wraps
from typing import Any, Mapping, Sequence

_ENUM_MARKER = "_mmm_qwen_enum_recovery_v3"
_TOOL_NAME_MARKER = "_mmm_qwen_tool_name_recovery_v1"
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
    """Return one formatting-equivalent enum member, or ``_NO_MATCH``.

    Normalization is syntactic only: whitespace, hyphen/underscore, camel-case and an
    optional JSON string wrapper may be normalized. Semantic synonyms are never mapped
    because that would silently revive removed operations such as whole-file writes.
    Ambiguous normalized schemas fail closed.
    """

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


def _install_enum_recovery(llama_cpp_module: Any) -> None:
    current_decode = llama_cpp_module._decode_parameter_value
    if bool(getattr(current_decode, _ENUM_MARKER, False)):
        return

    @wraps(current_decode)
    def decode_parameter_value(
        tool_name: str,
        key: str,
        raw: str,
        schema: Mapping[str, Any],
    ) -> Any:
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            canonical = canonical_string_enum(raw, enum)
            if canonical is not _NO_MATCH:
                return canonical
        try:
            return current_decode(tool_name, key, raw, schema)
        except RuntimeError as exc:
            if isinstance(enum, list) and enum and "emitted value outside enum" in str(exc):
                raise QwenEnumValueError(
                    tool_name=tool_name,
                    parameter_name=key,
                    raw_value=raw,
                    allowed_values=enum,
                ) from exc
            raise

    setattr(decode_parameter_value, _ENUM_MARKER, True)
    llama_cpp_module._decode_parameter_value = decode_parameter_value


def _install_tool_name_recovery(llama_cpp_module: Any) -> None:
    current_parse = getattr(llama_cpp_module, "_parse_qwen_function", None)
    if not callable(current_parse) or bool(getattr(current_parse, _TOOL_NAME_MARKER, False)):
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
        if name_end < 0:
            return current_parse(text, start, schemas, call_index=call_index)

        raw_name = text[name_start:name_end]
        emitted_name = raw_name.strip()
        if not emitted_name or emitted_name in schemas:
            return current_parse(text, start, schemas, call_index=call_index)

        from minecraft_mod_ai.model_tool_aliases import resolve_exposed_model_tool

        exposed_name = resolve_exposed_model_tool(emitted_name, schemas.keys())
        if exposed_name is None:
            return current_parse(text, start, schemas, call_index=call_index)

        rewritten = text[:name_start] + exposed_name + text[name_end:]
        call, rewritten_end = current_parse(
            rewritten,
            start,
            schemas,
            call_index=call_index,
        )
        length_delta = len(exposed_name) - len(raw_name)
        return call, rewritten_end - length_delta

    setattr(parse_qwen_function, _TOOL_NAME_MARKER, True)
    llama_cpp_module._parse_qwen_function = parse_qwen_function


def install(llama_cpp_module: Any) -> None:
    """Install deterministic Qwen parser canonicalization without generation retries."""

    _install_enum_recovery(llama_cpp_module)
    _install_tool_name_recovery(llama_cpp_module)


__all__ = [
    "QwenEnumValueError",
    "canonical_string_enum",
    "install",
]
