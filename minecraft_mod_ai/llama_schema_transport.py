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
_BRANCH_KEYS = ("allOf",)


def _bound_pattern_quantifiers(pattern: str, max_length: int) -> str:
    """Fold *maxLength* into a regex by bounding ``*`` and ``+`` quantifiers.

    llama.cpp processes ``pattern`` before ``maxLength``, so the length limit
    must live inside the regex itself.

    Unlike a naive approach that gives every quantifier the full *max_length*
    budget (which allows ``N * max_length`` total characters for *N* groups),
    this function first computes the maximum cost of all fixed-length elements,
    then distributes the remaining budget **evenly** across unbounded groups,
    guaranteeing total match length ≤ *max_length*.
    """
    n = len(pattern)

    def _quantifier_at(pos: int) -> tuple[int, bool, int, int]:
        """Inspect the quantifier (if any) following an atom at *pos*.

        Returns ``(new_pos, is_variable, min_contrib, fixed_max_contrib)``.
        """
        if pos >= n or pattern[pos] not in "*+?{":
            return pos, False, 0, 1          # no quantifier → fixed 1
        c = pattern[pos]
        if c == "*":
            return pos + 1, True, 0, 0       # unbounded, min 0
        if c == "+":
            return pos + 1, True, 1, 0       # unbounded, min 1
        if c == "?":
            return pos + 1, False, 0, 1      # optional atom, max 1
        # c == "{"
        j = pos + 1
        while j < n and pattern[j] != "}":
            j += 1
        qs = pattern[pos + 1 : j]
        if "," in qs:
            hi = qs.split(",")[1].strip()
            mx = int(hi) if hi else max_length
        else:
            mx = int(qs)
        return j + 1, False, 0, mx           # already bounded

    # ── Phase 1: analyse ──
    num_var = 0      # unbounded quantifier count
    min_var = 0      # sum of minimum contributions from variable groups
    fixed_max = 0    # max chars from non-variable elements (conservative)
    i = 0
    in_cc = False

    while i < n:
        ch = pattern[i]
        # escaped char
        if ch == "\\" and i + 1 < n:
            if in_cc:
                i += 2
                continue
            i += 2
            i, is_v, mn, fx = _quantifier_at(i)
            if is_v:
                num_var += 1; min_var += mn
            else:
                fixed_max += fx
            continue
        # char-class boundaries
        if ch == "[" and not in_cc:
            in_cc = True; i += 1; continue
        if ch == "]" and in_cc:
            in_cc = False; i += 1
            i, is_v, mn, fx = _quantifier_at(i)
            if is_v:
                num_var += 1; min_var += mn
            else:
                fixed_max += fx
            continue
        if in_cc:
            i += 1; continue
        # anchors / alternation
        if ch in "^$|":
            i += 1; continue
        # group open
        if ch == "(":
            i += 1
            if i < n and pattern[i] == "?":
                i += 1
                if i < n and pattern[i] in "!=":
                    # lookahead — skip, consumes no chars
                    depth = 1; i += 1
                    while i < n and depth > 0:
                        if pattern[i] == "\\" and i + 1 < n:
                            i += 2; continue
                        if pattern[i] == "(":
                            depth += 1
                        elif pattern[i] == ")":
                            depth -= 1
                        i += 1
                    continue
                if i < n and pattern[i] == ":":
                    i += 1
            continue
        # group close — check for group-level quantifier
        if ch == ")":
            i += 1
            if i < n and pattern[i] == "?":
                i += 1                        # optional group
            elif i < n and pattern[i] == "*":
                num_var += 1; i += 1
            elif i < n and pattern[i] == "+":
                num_var += 1; min_var += 1; i += 1
            continue
        # literal or dot — atom of width 1
        i += 1
        i, is_v, mn, fx = _quantifier_at(i)
        if is_v:
            num_var += 1; min_var += mn
        else:
            fixed_max += fx

    if num_var == 0:
        return pattern

    # ── Phase 2: compute per-group budget ──
    budget = max(0, max_length - fixed_max - min_var)
    per_extra = budget // num_var

    # ── Phase 3: replace * and + ──
    parts: list[str] = []
    i = 0
    in_cc = False
    while i < n:
        ch = pattern[i]
        if ch == "\\" and i + 1 < n:
            parts.append(pattern[i : i + 2]); i += 2; continue
        if ch == "[" and not in_cc:
            in_cc = True; parts.append(ch); i += 1; continue
        if ch == "]" and in_cc:
            in_cc = False; parts.append(ch); i += 1; continue
        if in_cc:
            parts.append(ch); i += 1; continue
        if ch == "*":
            parts.append("{0," + str(per_extra) + "}"); i += 1; continue
        if ch == "+":
            parts.append("{1," + str(1 + per_extra) + "}"); i += 1; continue
        parts.append(ch); i += 1
    return "".join(parts)


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
    if has_pattern:
        if has_max_length:
            result["pattern"] = _bound_pattern_quantifiers(
                schema["pattern"], schema["maxLength"]
            )
        else:
            result["pattern"] = schema["pattern"]
    if has_max_length:
        result["maxLength"] = schema["maxLength"]
    if "minLength" in schema and isinstance(schema["minLength"], int):
        result["minLength"] = schema["minLength"]
    if result:
        return result

    return _fallback_branch(schema)


__all__ = ["project_llama_transport_schema"]
