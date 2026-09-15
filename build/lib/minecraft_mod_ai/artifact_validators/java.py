from __future__ import annotations

"""Java fragment syntax and structural validation."""

import re


class JavaValidationError(ValueError):
    pass


def validate_java_fragment(source: str, *, anchor: str = "") -> dict[str, object]:
    """Validate a rendered Java fragment for basic syntax balance and safety."""
    if not isinstance(source, str) or not source.strip():
        raise JavaValidationError("JAVA_EMPTY: Java source fragment cannot be empty")

    stripped = source.strip()
    stack = []
    pairs = {"{": "}", "(": ")", "[": "]"}
    in_string = False
    escape = False

    for i, ch in enumerate(stripped):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack.pop() != ch:
                raise JavaValidationError(
                    f"JAVA_SYNTAX: Unbalanced bracket '{ch}' at position {i}"
                )

    if stack:
        raise JavaValidationError(
            f"JAVA_SYNTAX: Unclosed bracket expected '{stack[-1]}'"
        )

    field_match = re.search(
        r"public\s+static\s+final\s+([A-Za-z0-9_<>]+)\s+([A-Za-z0-9_]+)\s*=",
        stripped,
    )
    declared_symbol = field_match.group(2) if field_match else ""

    return {
        "status": "PASS",
        "length": len(stripped),
        "declared_symbol": declared_symbol,
        "anchor": anchor,
    }
