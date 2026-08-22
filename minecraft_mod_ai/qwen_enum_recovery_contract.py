from __future__ import annotations

"""Bounded recovery for malformed Qwen tagged string-enum parameters.

Qwen's native tagged tool format places string values directly inside
``<parameter=...>`` blocks. Small local models occasionally serialize those scalar
values as JSON strings or vary harmless spelling details (case, separators, camel
case). The host may canonicalize only formatting-equivalent values that map uniquely
to one schema enum member. Any semantic mismatch is discarded before execution and
gets exactly one corrective generation attempt.
"""

import json
import re
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_qwen_enum_recovery_v1"
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


def _correction_message(error: QwenEnumValueError) -> Mapping[str, str]:
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


def install(llama_cpp_module: Any) -> None:
    """Install enum canonicalization and one bounded semantic retry."""

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
        except QwenEnumValueError as first_error:
            retry_request = replace(
                request,
                messages=(*tuple(request.messages), _correction_message(first_error)),
                parallel_tool_calls=False,
            )
            try:
                return current_tool_completion(adapter, server_url, retry_request)
            except QwenEnumValueError as retry_error:
                raise RuntimeError(
                    "Qwen emitted an invalid tool enum after one bounded corrective "
                    f"retry: {retry_error}"
                ) from retry_error

    setattr(decode_parameter_value, _MARKER, True)
    setattr(tool_semantic_completion, _MARKER, True)
    llama_cpp_module._decode_parameter_value = decode_parameter_value
    llama_cpp_module._tool_semantic_completion = tool_semantic_completion


__all__ = ["QwenEnumValueError", "canonical_string_enum", "install"]
