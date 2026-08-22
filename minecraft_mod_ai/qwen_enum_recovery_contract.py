from __future__ import annotations

"""Deterministic parser repair for Qwen tagged enum parameters.

Qwen's native tagged tool format places scalar values directly inside tagged
parameter blocks. Small local models occasionally serialize string enums with harmless
formatting drift or use a small set of lossless, tool-specific synonyms.

This module performs only deterministic parsing/canonicalization. It never asks the
model to generate again. Malformed calls that cannot be repaired without inventing
information are raised as typed protocol errors; causal_stale_tool_recovery_contract is
the sole owner of any bounded model re-synchronization.
"""

import json
import re
from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_qwen_enum_recovery_v2"
_NO_MATCH = object()
_MAX_VALUE_PREVIEW = 160

# Exact, lossless aliases observed from Qwen for this one protocol field. Keep this
# deliberately finite: no fuzzy matching and no cross-tool semantic guessing.
_TOOL_ENUM_ALIASES: dict[tuple[str, str], Mapping[str, str]] = {
    ("apply_source_edit", "operation"): {
        "overwrite_file": "replace_file",
        "update_file": "replace_file",
    },
}


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

    Generic normalization remains syntactic only. Semantic aliases are handled by the
    separate finite tool/parameter map below so they cannot leak across schemas.
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


def _canonical_tool_enum(
    tool_name: str,
    parameter_name: str,
    raw: str,
    allowed_values: Sequence[Any],
) -> Any:
    canonical = canonical_string_enum(raw, allowed_values)
    if canonical is not _NO_MATCH:
        return canonical
    if not allowed_values or not all(isinstance(value, str) for value in allowed_values):
        return _NO_MATCH

    aliases = _TOOL_ENUM_ALIASES.get((tool_name, parameter_name))
    if not aliases:
        return _NO_MATCH
    allowed = frozenset(str(value) for value in allowed_values)
    for candidate in _string_candidates(raw):
        target = aliases.get(_enum_key(candidate))
        if target is not None and target in allowed:
            return target
    return _NO_MATCH


def install(llama_cpp_module: Any) -> None:
    """Install parser-only enum canonicalization; never install a generation retry."""

    current_decode = llama_cpp_module._decode_parameter_value
    if bool(getattr(current_decode, _MARKER, False)):
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
            canonical = _canonical_tool_enum(tool_name, key, raw, enum)
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

    setattr(decode_parameter_value, _MARKER, True)
    llama_cpp_module._decode_parameter_value = decode_parameter_value


__all__ = [
    "QwenEnumValueError",
    "canonical_string_enum",
    "install",
]
