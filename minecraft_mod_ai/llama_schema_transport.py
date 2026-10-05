from __future__ import annotations

"""Project host JSON Schema into a conservative llama.cpp transport schema.

The full response schema is an application contract and stays host-owned.  This module
keeps only structural information useful to the sampler and deliberately drops
conditional/validation keywords that can make native grammar compilation brittle.
"""

import copy
from collections.abc import Mapping, Sequence
from typing import Any

_JSON_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)

_PINNED_GBNF_ESCAPES = frozenset({"t", "r", "n", "\\", '"', "[", "]"})
_REGEX_META_ESCAPES = frozenset("^$.[\\]()|{}*+?")
_HEX_ESCAPE_WIDTH = {"x": 2, "u": 4, "U": 8}
_SHORTHAND_OUTSIDE_CLASS = {
    "d": "[0-9]",
    "D": "[^0-9]",
    "w": "[A-Za-z0-9_]",
    "W": "[^A-Za-z0-9_]",
    "s": r"[ \t\r\n\x0B\x0C]",
    "S": r"[^ \t\r\n\x0B\x0C]",
}
_SHORTHAND_INSIDE_CLASS = {
    "d": "0-9",
    "w": "A-Za-z0-9_",
    "s": r" \t\r\n\x0B\x0C",
}


def _normalize_llama_pattern_escapes(pattern: str) -> str | None:
    """Translate regex escapes the pinned llama.cpp GBNF parser cannot read.

    llama.cpp 1d2869c copies unknown regex escapes into generated GBNF. Its
    grammar parser accepts a much smaller escape set, so digit/word/space
    shorthands can reach sampler initialization as invalid grammar.

    Preserve common JSON-Schema shorthand semantics. If an escape cannot be
    represented safely, return None so the decoder uses the finite length
    constraint while the host keeps the original pattern for validation.
    """

    parts: list[str] = []
    i = 0
    in_class = False
    length = len(pattern)

    while i < length:
        char = pattern[i]

        if in_class:
            if char == "]":
                in_class = False
                parts.append(char)
                i += 1
                continue
            if char != "\\":
                parts.append(char)
                i += 1
                continue
            if i + 1 >= length:
                return None

            escape = pattern[i + 1]
            width = _HEX_ESCAPE_WIDTH.get(escape)
            if width is not None:
                end = i + 2 + width
                if end > length or any(
                    digit not in "0123456789abcdefABCDEF"
                    for digit in pattern[i + 2 : end]
                ):
                    return None
                parts.append(pattern[i:end])
                i = end
                continue

            shorthand = _SHORTHAND_INSIDE_CLASS.get(escape)
            if shorthand is not None:
                parts.append(shorthand)
                i += 2
                continue
            if escape in {"D", "W", "S"}:
                return None
            if escape == "b":
                parts.append(r"\x08")
                i += 2
                continue
            if escape in {"f", "v"}:
                parts.append(r"\x0C" if escape == "f" else r"\x0B")
                i += 2
                continue
            if escape in _PINNED_GBNF_ESCAPES:
                parts.append(pattern[i : i + 2])
                i += 2
                continue
            if escape == "-":
                parts.append(r"\x2D")
                i += 2
                continue
            if escape.isalnum():
                return None

            codepoint = ord(escape)
            if codepoint <= 0x7F:
                parts.append(f"\\x{codepoint:02X}")
                i += 2
                continue
            return None

        if char == "[":
            in_class = True
            parts.append(char)
            i += 1
            continue
        if char != "\\":
            parts.append(char)
            i += 1
            continue
        if i + 1 >= length:
            return None

        escape = pattern[i + 1]
        width = _HEX_ESCAPE_WIDTH.get(escape)
        if width is not None:
            end = i + 2 + width
            if end > length or any(
                digit not in "0123456789abcdefABCDEF"
                for digit in pattern[i + 2 : end]
            ):
                return None
            parts.append(pattern[i:end])
            i = end
            continue

        shorthand = _SHORTHAND_OUTSIDE_CLASS.get(escape)
        if shorthand is not None:
            parts.append(shorthand)
            i += 2
            continue
        if escape in {"f", "v"}:
            parts.append(r"\x0C" if escape == "f" else r"\x0B")
            i += 2
            continue
        if escape in _PINNED_GBNF_ESCAPES or escape in _REGEX_META_ESCAPES:
            parts.append(pattern[i : i + 2])
            i += 2
            continue
        if escape.isalnum():
            return None

        parts.append(escape)
        i += 2

    return None if in_class else "".join(parts)


def _bound_pattern_quantifiers(pattern: str, max_length: int) -> str:
    """Fold maxLength into unbounded atom quantifiers.

    The pinned llama.cpp checks pattern before maxLength. Bounding every
    unbounded atom keeps sampler output finite without dropping the lexical
    constraint. Unsupported or malformed shapes raise ValueError and fall back
    to the finite length-only transport.
    """

    n = len(pattern)

    def _escape_end(pos: int) -> int:
        if pos >= n or pattern[pos] != "\\" or pos + 1 >= n:
            raise ValueError("invalid regex escape")
        width = _HEX_ESCAPE_WIDTH.get(pattern[pos + 1])
        if width is None:
            return pos + 2
        end = pos + 2 + width
        if end > n:
            raise ValueError("truncated hex escape")
        return end

    def _quantifier_at(pos: int) -> tuple[int, bool, int, int]:
        if pos >= n or pattern[pos] not in "*+?{":
            return pos, False, 0, 1
        char = pattern[pos]
        if char == "*":
            return pos + 1, True, 0, 0
        if char == "+":
            return pos + 1, True, 1, 0
        if char == "?":
            return pos + 1, False, 0, 1

        close = pattern.find("}", pos + 1)
        if close < 0:
            raise ValueError("unterminated quantifier")
        body = pattern[pos + 1 : close]
        if "," in body:
            lower, upper = body.split(",", 1)
            if not lower.isdigit() or not upper.isdigit():
                raise ValueError("open or malformed bounded quantifier")
            maximum = int(upper)
        elif body.isdigit():
            maximum = int(body)
        else:
            raise ValueError("malformed bounded quantifier")
        return close + 1, False, 0, maximum

    variable_count = 0
    variable_minimum = 0
    fixed_maximum = 0
    i = 0
    in_class = False
    group_depth = 0

    while i < n:
        char = pattern[i]

        if in_class:
            if char == "\\":
                i = _escape_end(i)
                continue
            if char == "]":
                in_class = False
                i += 1
                i, variable, minimum, fixed = _quantifier_at(i)
                if variable:
                    variable_count += 1
                    variable_minimum += minimum
                else:
                    fixed_maximum += fixed
                continue
            i += 1
            continue

        if char == "\\":
            i = _escape_end(i)
            i, variable, minimum, fixed = _quantifier_at(i)
            if variable:
                variable_count += 1
                variable_minimum += minimum
            else:
                fixed_maximum += fixed
            continue
        if char == "[":
            in_class = True
            i += 1
            continue
        if char in "^$|":
            i += 1
            continue
        if char == "(":
            if pattern.startswith("(?:", i):
                i += 3
            elif pattern.startswith("(?", i):
                raise ValueError("unsupported regex group")
            else:
                i += 1
            group_depth += 1
            continue
        if char == ")":
            if group_depth <= 0:
                raise ValueError("unbalanced regex group")
            group_depth -= 1
            i += 1
            if i < n and pattern[i] in "*+":
                raise ValueError("unbounded group repetition")
            if i < n and pattern[i] == "?":
                i += 1
            continue
        if char in "*+?{":
            raise ValueError("quantifier without a preceding atom")

        i += 1
        i, variable, minimum, fixed = _quantifier_at(i)
        if variable:
            variable_count += 1
            variable_minimum += minimum
        else:
            fixed_maximum += fixed

    if in_class:
        raise ValueError("unterminated character class")
    if group_depth:
        raise ValueError("unbalanced regex group")
    if variable_count == 0:
        return pattern

    remaining = max(0, max_length - fixed_maximum - variable_minimum)
    per_variable_extra = remaining // variable_count

    parts: list[str] = []
    i = 0
    in_class = False
    while i < n:
        char = pattern[i]

        if in_class:
            if char == "\\":
                end = _escape_end(i)
                parts.append(pattern[i:end])
                i = end
                continue
            parts.append(char)
            if char == "]":
                in_class = False
            i += 1
            continue

        if char == "\\":
            end = _escape_end(i)
            parts.append(pattern[i:end])
            i = end
            continue
        if char == "[":
            in_class = True
            parts.append(char)
            i += 1
            continue
        if char == "*":
            parts.append(f"{{0,{per_variable_extra}}}")
            i += 1
            continue
        if char == "+":
            parts.append(f"{{1,{1 + per_variable_extra}}}")
            i += 1
            continue
        parts.append(char)
        i += 1
    return "".join(parts)


def _llama_safe_pattern(pattern: str, max_length: int | None) -> str | None:
    """Return a pattern safe for the pinned llama.cpp grammar compiler."""

    if len(pattern) < 2 or not pattern.startswith("^") or not pattern.endswith("$"):
        return None
    normalized = _normalize_llama_pattern_escapes(pattern)
    if normalized is None:
        return None
    if max_length is None:
        return normalized
    try:
        return _bound_pattern_quantifiers(normalized, max_length)
    except (TypeError, ValueError):
        return None

def _project_type(value: Any) -> str | list[str] | None:
    if isinstance(value, str):
        return value if value in _JSON_TYPES else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        types = [item for item in value if isinstance(item, str) and item in _JSON_TYPES]
        return list(dict.fromkeys(types)) or None
    return None


def _project_enum(value: Any) -> list[Any] | None:
    if not isinstance(value, list) or not value:
        return None
    return copy.deepcopy(value)


def _fallback_branch(schema: Mapping[str, Any]) -> dict[str, Any]:
    variants = schema.get("allOf")
    if isinstance(variants, Sequence) and not isinstance(
        variants, (str, bytes, bytearray)
    ):
        for branch in variants:
            projected = project_llama_transport_schema(branch)
            if projected:
                return projected
    if "const" in schema:
        return {"const": copy.deepcopy(schema["const"])}
    return {}


def project_llama_transport_schema(schema: Any) -> dict[str, Any]:
    """Return a structural schema suitable for llama.cpp sampler initialization.

    Explicit base structure always wins over ``allOf``/``if`` branches. Structural
    variant unions (``oneOf``/``anyOf``) preserve their discriminated branches, required
    keys, and literal const values so the sampler cannot emit partial variants.
    """

    if not isinstance(schema, Mapping):
        return {}

    projected_type = _project_type(schema.get("type"))
    has_properties = isinstance(schema.get("properties"), Mapping)
    has_items = isinstance(schema.get("items"), Mapping)

    for combinator in ("oneOf", "anyOf"):
        raw_branches = schema.get(combinator)
        if isinstance(raw_branches, Sequence) and not isinstance(
            raw_branches, (str, bytes, bytearray)
        ):
            branches = [
                project_llama_transport_schema(branch)
                for branch in raw_branches
                if isinstance(branch, Mapping)
            ]
            branches = [b for b in branches if b]
            if branches:
                res: dict[str, Any] = {}
                if projected_type is not None:
                    res["type"] = projected_type
                if has_properties:
                    raw_properties = schema.get("properties")
                    if isinstance(raw_properties, Mapping):
                        res["properties"] = {
                            str(name): project_llama_transport_schema(child)
                            for name, child in raw_properties.items()
                            if isinstance(name, str)
                        }
                    raw_required = schema.get("required")
                    if isinstance(raw_required, Sequence) and not isinstance(
                        raw_required, (str, bytes, bytearray)
                    ):
                        req = [
                            name
                            for name in raw_required
                            if isinstance(name, str) and name in res.get("properties", {})
                        ]
                        if req:
                            res["required"] = req
                    additional = schema.get("additionalProperties")
                    if isinstance(additional, bool):
                        res["additionalProperties"] = additional
                    elif isinstance(additional, Mapping):
                        res["additionalProperties"] = project_llama_transport_schema(additional)
                res[combinator] = branches
                return res

    # Preserve an explicit/base object before considering combinators.  This is important
    # for schemas that append conditional allOf clauses to an otherwise normal object.
    if projected_type == "object" or has_properties:
        result: dict[str, Any] = {"type": "object"}
        raw_properties = schema.get("properties")
        if isinstance(raw_properties, Mapping):
            result["properties"] = {
                str(name): project_llama_transport_schema(child)
                for name, child in raw_properties.items()
                if isinstance(name, str)
            }
            raw_required = schema.get("required")
            if isinstance(raw_required, Sequence) and not isinstance(
                raw_required, (str, bytes, bytearray)
            ):
                required = [
                    name
                    for name in raw_required
                    if isinstance(name, str) and name in result["properties"]
                ]
                if required:
                    result["required"] = required

        additional = schema.get("additionalProperties")
        if isinstance(additional, bool):
            result["additionalProperties"] = additional
        elif isinstance(additional, Mapping):
            result["additionalProperties"] = project_llama_transport_schema(additional)

        if "maxProperties" in schema and isinstance(schema["maxProperties"], int):
            result["maxProperties"] = schema["maxProperties"]
        if "minProperties" in schema and isinstance(schema["minProperties"], int):
            result["minProperties"] = schema["minProperties"]
        return result

    if projected_type == "array" or has_items:
        result = {"type": "array"}
        items = schema.get("items")
        result["items"] = (
            project_llama_transport_schema(items) if isinstance(items, Mapping) else {}
        )
        if "maxItems" in schema and isinstance(schema["maxItems"], int):
            result["maxItems"] = schema["maxItems"]
        if "minItems" in schema and isinstance(schema["minItems"], int):
            result["minItems"] = schema["minItems"]
        return result

    result = {}
    if projected_type is not None:
        result["type"] = projected_type
    enum = _project_enum(schema.get("enum"))
    if enum is not None:
        result["enum"] = enum
    if "const" in schema:
        result["const"] = copy.deepcopy(schema["const"])
    # ── llama.cpp keyword precedence fix ──
    # llama.cpp processes "pattern" first and returns immediately, skipping
    # "maxLength" / "minLength".  Dropping either keyword breaks one contract:
    #   drop pattern  → sampler loses grammar enforcement (allows >=, etc.)
    #   drop maxLength → sampler produces unbounded output (token exhaustion)
    #
    # Solution: fold maxLength INTO the regex by replacing unbounded ``*``/``+``
    # with ``{0,N}``/``{1,N}``.  The resulting bounded pattern enforces both
    # grammar and length in a single llama.cpp grammar pass.  maxLength is kept
    # in the schema for the host-side token ceiling calculator.
    has_pattern = "pattern" in schema and isinstance(schema["pattern"], str)
    has_max_length = "maxLength" in schema and isinstance(schema["maxLength"], int)
    safe_pattern = (
        _llama_safe_pattern(
            schema["pattern"],
            schema["maxLength"] if has_max_length else None,
        )
        if has_pattern
        else None
    )
    if safe_pattern is not None:
        result["pattern"] = safe_pattern
    if has_max_length:
        result["maxLength"] = schema["maxLength"]
    if "minLength" in schema and isinstance(schema["minLength"], int):
        result["minLength"] = schema["minLength"]
    if result:
        return result

    return _fallback_branch(schema)


__all__ = ["project_llama_transport_schema"]
