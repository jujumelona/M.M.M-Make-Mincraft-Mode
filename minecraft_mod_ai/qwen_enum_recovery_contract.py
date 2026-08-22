from __future__ import annotations

"""Bounded recovery for malformed Qwen tagged tool parameters.

Qwen's native tagged tool format places scalar values directly inside tagged
parameter blocks. Small local models occasionally serialize string enums with harmless
formatting drift, or invent a parameter name that is not present in the authoritative
tool schema. Formatting-equivalent enum values may be canonicalized uniquely. A
schema-invalid parameter is never deleted, renamed, or executed: the malformed call is
discarded and the same currently visible tool gets exactly one schema-constrained
corrective generation attempt.
"""

import ast
import json
import re
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_qwen_enum_recovery_v1"
_NO_MATCH = object()
_MAX_VALUE_PREVIEW = 160
_UNKNOWN_PARAMETER_RE = re.compile(
    r"^Qwen tool (?P<tool>.+?) emitted unknown parameter (?P<parameter>.+)$"
)


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


class QwenUnknownParameterError(RuntimeError):
    """A visible Qwen tool call contained a key absent from its strict schema."""

    def __init__(
        self,
        *,
        tool_name: str,
        parameter_name: str,
        allowed_parameters: Sequence[str],
    ) -> None:
        self.tool_name = tool_name
        self.parameter_name = parameter_name
        self.allowed_parameters = tuple(str(value) for value in allowed_parameters)
        super().__init__(
            f"Qwen tool {tool_name!r} emitted unknown parameter {parameter_name!r}; "
            f"allowed parameters={list(self.allowed_parameters)!r}"
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

    Normalization is intentionally syntactic only. It never maps semantic aliases such
    as ``edit`` to ``replace_exact``. Ambiguous normalized schemas also fail closed.
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


def _schema_name(schema: Any) -> str | None:
    if not isinstance(schema, Mapping):
        return None
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    return str(name) if isinstance(name, str) and name else None


def _schema_for_tool(request: Any, tool_name: str, *, validation: bool) -> Mapping[str, Any] | None:
    attribute = "tool_validation_schemas" if validation else "tools"
    for schema in tuple(getattr(request, attribute, ()) or ()):
        if _schema_name(schema) == tool_name and isinstance(schema, Mapping):
            return schema
    return None


def _parameters_for_schema(schema: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(schema, Mapping):
        return None
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return None
    parameters = function.get("parameters")
    return parameters if isinstance(parameters, Mapping) else None


def _literal_string(raw: str) -> str | None:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, str) else None


def _classify_unknown_parameter(request: Any, exc: RuntimeError) -> QwenUnknownParameterError | None:
    """Promote only the parser's exact strict-schema unknown-key failure.

    The text exception alone is not trusted. Recovery is enabled only when the named
    tool is still model-visible and the authoritative validation schema independently
    proves that ``additionalProperties`` is false and that the emitted key is absent.
    """

    match = _UNKNOWN_PARAMETER_RE.fullmatch(str(exc))
    if match is None:
        return None
    tool_name = _literal_string(match.group("tool"))
    parameter_name = _literal_string(match.group("parameter"))
    if tool_name is None or parameter_name is None:
        return None

    visible_schema = _schema_for_tool(request, tool_name, validation=False)
    if visible_schema is None:
        return None
    authoritative_schema = _schema_for_tool(request, tool_name, validation=True) or visible_schema
    parameters = _parameters_for_schema(authoritative_schema)
    if parameters is None or parameters.get("additionalProperties", False) is not False:
        return None
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping) or parameter_name in properties:
        return None

    return QwenUnknownParameterError(
        tool_name=tool_name,
        parameter_name=parameter_name,
        allowed_parameters=tuple(str(name) for name in properties),
    )


def _recoverable_error(request: Any, exc: RuntimeError) -> RuntimeError | None:
    if isinstance(exc, (QwenEnumValueError, QwenUnknownParameterError)):
        return exc
    return _classify_unknown_parameter(request, exc)


def _correction_message(error: RuntimeError) -> Mapping[str, str]:
    if isinstance(error, QwenUnknownParameterError):
        allowed = ", ".join(repr(value) for value in error.allowed_parameters) or "<none>"
        return {
            "role": "system",
            "content": (
                "The previous tool call was discarded without execution because it used "
                f"unknown parameter {error.parameter_name!r} for tool {error.tool_name!r}. "
                f"The authoritative schema allows only these parameter names: {allowed}. "
                "Re-emit the complete same intended tool call exactly once using only "
                "parameters from that schema. Do not delete the intended edit, invent a "
                "replacement field, rename, translate, alias, or infer parameters."
            ),
        }
    if not isinstance(error, QwenEnumValueError):
        raise TypeError(f"Unsupported Qwen recovery error: {type(error).__name__}")
    allowed = ", ".join(repr(value) for value in error.allowed_values)
    preview = error.raw_value[:_MAX_VALUE_PREVIEW]
    return {
        "role": "system",
        "content": (
            "The previous tool call was discarded without execution because one enum "
            "parameter was invalid. Re-emit the same intended action exactly once using "
            f"tool {error.tool_name!r}. Parameter {error.parameter_name!r} must be exactly "
            f"one of: {allowed}. Invalid value: {preview!r}. Do not invent another enum "
            "value."
        ),
    }


def _retry_request(request: Any, error: RuntimeError) -> Any:
    tool_name = getattr(error, "tool_name", None)
    updates: dict[str, Any] = {
        "messages": (*tuple(request.messages), _correction_message(error)),
        "parallel_tool_calls": False,
    }
    if isinstance(tool_name, str):
        visible_schema = _schema_for_tool(request, tool_name, validation=False)
        if visible_schema is not None and hasattr(request, "tools"):
            updates["tools"] = (visible_schema,)
        if visible_schema is not None and hasattr(request, "tool_choice"):
            updates["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_name},
            }
    return replace(request, **updates)


def install(llama_cpp_module: Any) -> None:
    """Install formatting-only enum repair plus one bounded schema correction."""

    current_decode = llama_cpp_module._decode_parameter_value
    if bool(getattr(current_decode, _MARKER, False)):
        return
    current_tool_completion = llama_cpp_module._tool_semantic_completion

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

    @wraps(current_tool_completion)
    def tool_semantic_completion(adapter: Any, server_url: str, request: Any) -> Any:
        try:
            return current_tool_completion(adapter, server_url, request)
        except RuntimeError as exc:
            first_error = _recoverable_error(request, exc)
            if first_error is None:
                raise
            retry_request = _retry_request(request, first_error)
            try:
                return current_tool_completion(adapter, server_url, retry_request)
            except RuntimeError as retry_exc:
                retry_error = _recoverable_error(retry_request, retry_exc)
                if retry_error is None:
                    raise
                raise RuntimeError(
                    "Qwen emitted an invalid tool call after one bounded corrective "
                    f"retry: {retry_error}"
                ) from retry_error

    setattr(decode_parameter_value, _MARKER, True)
    setattr(tool_semantic_completion, _MARKER, True)
    llama_cpp_module._decode_parameter_value = decode_parameter_value
    llama_cpp_module._tool_semantic_completion = tool_semantic_completion


__all__ = [
    "QwenEnumValueError",
    "QwenUnknownParameterError",
    "canonical_string_enum",
    "install",
]
