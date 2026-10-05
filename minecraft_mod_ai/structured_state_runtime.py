from __future__ import annotations

"""Host compiler for structured state-model records.

Supported state expressions and mutations compile directly to Java. Anything
outside the canonical typed IR fails closed before source generation; there is no
secondary DSL, model callback, or Java-source fallback on this path.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import re
from typing import Any


_STATE_IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
_STATE_SCALAR_TYPES = ("boolean", "int", "long", "double", "string")
STATE_EXPRESSION_PATTERN = r"^.*$"
STATE_MUTATION_PATTERN = r"^.*$"

_STATE_EXECUTABLE_FIELDS = {
    "transitions": ("guard", "mutation"),
    "invariants": ("condition",),
    "initialization": ("initial_state",),
    "updates": ("mutation",),
    "cleanup": ("action",),
}


def state_variable_default_schema(
    value_family: str | None = None,
) -> dict[str, Any]:
    """Return the canonical planner/storage contract for one state default.

    Empty text is a valid string default. Families with a finite textual encoding
    narrow that base contract rather than relying on a blanket non-empty rule.
    """

    family = str(value_family or "").strip().casefold()
    if family in {"int", "long", "double"}:
        family = "number"
    schema: dict[str, Any] = {
        "type": "string",
        "minLength": 0,
        "maxLength": 128,
        "pattern": r"^[^{}\[\]]*$",
    }
    if family == "number":
        schema["minLength"] = 1
        schema["pattern"] = (
            r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$"
        )
    elif family == "boolean":
        schema["minLength"] = 1
        schema.pop("pattern", None)
        schema["enum"] = ["true", "false"]
    elif family == "map":
        schema["minLength"] = 2
        schema.pop("pattern", None)
        schema["enum"] = ["{}", "empty_map"]
    elif family == "list":
        schema["minLength"] = 2
        schema.pop("pattern", None)
        schema["enum"] = ["[]", "empty_list", "empty_set"]
    return schema


def constrain_state_record_schema(
    concern: str,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Decorate state-model record schemas with guidance descriptions."""
    result = deepcopy(dict(schema))
    properties = result.get("properties")
    if not isinstance(properties, dict):
        return result
    if concern == "variables":
        if isinstance(properties.get("name"), dict):
            properties["name"].update({
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
                "pattern": _STATE_IDENTIFIER_PATTERN,
                "description": (
                    "Stable ASCII identifier for exactly one independently mutable state value. "
                    "Do not pack multiple logical fields into one object-shaped variable."
                ),
            })
        if isinstance(properties.get("type"), dict):
            properties["type"].clear()
            properties["type"].update({
                "type": "string",
                "enum": list(_STATE_SCALAR_TYPES),
                "description": (
                    "Canonical scalar state type. No aliases or aggregate compatibility "
                    "types are accepted."
                ),
            })
        if isinstance(properties.get("default"), dict):
            description = properties["default"].get("description")
            properties["default"].clear()
            properties["default"].update(state_variable_default_schema())
            properties["default"]["description"] = (
                str(description)
                if description
                else (
                    "Scalar default matching this variable's declared type; never encode a JSON "
                    "object or array inside this string field."
                )
            )
    for field in _STATE_EXECUTABLE_FIELDS.get(concern, ()):
        target = properties.get(field)
        if not isinstance(target, dict):
            continue
        if field in {"mutation", "mutations", "initial_state", "action"}:
            target["description"] = (
                "Canonical typed state-mutation IR. Targets must be declared state "
                "variables; external subsystem actions do not belong in this field."
            )
        else:
            target["description"] = (
                "Canonical typed state-expression IR. State and context references "
                "remain distinct namespaces through production."
            )
    return result


def _prior_state_variable_names(
    prior_chunks: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    names: list[str] = []
    for chunk in prior_chunks:
        rows = chunk.get("variables")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name") or "").strip()
            if (
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                and name not in names
            ):
                names.append(name)
    return tuple(names)


def constrain_state_chunk_schema(
    schema: Mapping[str, Any],
    prior_chunks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Narrow mutation targets to variables already authored in earlier chunks."""
    result = deepcopy(dict(schema))
    names = _prior_state_variable_names(prior_chunks)
    if not names:
        return result
    properties = result.get("properties")
    if not isinstance(properties, dict):
        return result
    for concern in ("transitions", "initialization", "updates", "cleanup"):
        concern_schema = properties.get(concern)
        if not isinstance(concern_schema, dict):
            continue
        items = concern_schema.get("items")
        item_properties = items.get("properties") if isinstance(items, dict) else None
        if not isinstance(item_properties, dict):
            continue
        field = {
            "transitions": "mutation",
            "initialization": "initial_state",
            "updates": "mutation",
            "cleanup": "action",
        }[concern]
        field_schema = item_properties.get(field)
        if isinstance(field_schema, dict):
            field_schema["description"] = (
                "Canonical typed mutation IR. Assignment targets are restricted to "
                "already-authored state variables: " + ", ".join(names) + "."
            )
    return result

_SUPPORTED_STATE_FUNCTIONS = frozenset({
    "sum",
    "min",
    "max",
    "abs",
    "count",
    "size",
    "len",
})


def _java_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def validate_state_expr_ir(
    expr: Any,
    *,
    symbols: StateSymbolTable | set[str] | Sequence[str] | None = None,
) -> None:
    """Validate canonical typed expression IR against declared symbols."""
    if isinstance(symbols, StateSymbolTable):
        declared = symbols.declared_names
    elif isinstance(symbols, set):
        declared = symbols
    elif isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes, bytearray)):
        declared = set(symbols)
    else:
        declared = None

    if not isinstance(expr, Mapping):
        raise ValueError(
            "STRUCTURED_STATE_EXPRESSION: canonical expression must be an object; "
            f"got {type(expr).__name__}"
        )

    kind = str(expr.get("kind") or "").strip()
    if not kind:
        raise ValueError(
            f"STRUCTURED_STATE_EXPRESSION: missing 'kind' in expression object {expr!r}"
        )

    if kind in {"and", "or"}:
        terms = expr.get("terms")
        if terms is None:
            terms = ()
        elif not isinstance(terms, Sequence) or isinstance(terms, (str, bytes, bytearray)):
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: {kind} terms must be a list")
        for term in terms:
            validate_state_expr_ir(term, symbols=declared)
        return

    if kind == "compare":
        op = expr.get("op", "==")
        if op not in {"==", "!=", ">=", "<=", ">", "<"}:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: unsupported comparison op {op!r}")
        left = expr.get("left")
        right = expr.get("right")
        if left is None or right is None:
            raise ValueError("STRUCTURED_STATE_EXPRESSION: compare requires left and right operands")
        validate_state_expr_ir(left, symbols=declared)
        validate_state_expr_ir(right, symbols=declared)
        return

    if kind == "not":
        term = expr.get("term") or expr.get("operand") or expr.get("left")
        if term is None:
            raise ValueError("STRUCTURED_STATE_EXPRESSION: not requires a term")
        validate_state_expr_ir(term, symbols=declared)
        return

    if kind == "state_ref":
        name = expr.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("STRUCTURED_STATE_EXPRESSION: state_ref requires string 'name'")
        name = name.strip()
        if declared is not None and name not in declared:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: undeclared state variable {name!r}")
        return

    if kind == "context_ref":
        name = expr.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("STRUCTURED_STATE_EXPRESSION: context_ref requires string 'name'")
        return

    if kind == "number":
        val = str(expr.get("value") or "").strip()
        if not re.fullmatch(r"^-?[0-9]+([.][0-9]+)?$", val):
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: invalid number literal {val!r}")
        return

    if kind == "literal":
        value = expr.get("value")
        if isinstance(value, str):
            if any(char in value for char in "{}[]"):
                raise ValueError(
                    "STRUCTURED_STATE_EXPRESSION: structured containers may not be "
                    "serialized into a scalar string literal"
                )
            if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
                raise ValueError(
                    "STRUCTURED_STATE_EXPRESSION: string literal contains an unpaired surrogate"
                )
        return

    if kind in {"empty_map", "empty_list"}:
        return

    if kind == "arithmetic":
        op = expr.get("op", "+")
        if op not in {"+", "-", "*", "/", "%"}:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: unsupported arithmetic op {op!r}")
        validate_state_expr_ir(expr.get("left"), symbols=declared)
        validate_state_expr_ir(expr.get("right"), symbols=declared)
        return

    if kind == "call":
        name = str(expr.get("name") or "").casefold()
        if name not in _SUPPORTED_STATE_FUNCTIONS:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: unsupported function {name!r}")
        args = expr.get("args") or ()
        for arg in args:
            validate_state_expr_ir(arg, symbols=declared)
        return

    raise ValueError(f"STRUCTURED_STATE_EXPRESSION: unknown expression kind {kind!r}")


def compile_state_condition_ir(
    expr: Any,
    *,
    declared: set[str] | None = None,
    context: str = "context",
) -> str:
    """Compile any state expression in boolean/guard position."""

    if not isinstance(expr, Mapping):
        raise ValueError(
            "STRUCTURED_STATE_EXPRESSION: canonical condition must be an object; "
            f"got {type(expr).__name__}"
        )
    kind = str(expr.get("kind") or "").strip()
    if kind in {"and", "or", "not", "implies", "compare"}:
        return compile_state_expr_ir(
            expr,
            declared=declared,
            context=context,
        )
    compiled = compile_state_expr_ir(
        expr,
        declared=declared,
        context=context,
    )
    return f"$mmmTruthy({compiled})"


def compile_state_expr_ir(
    expr: Any,
    *,
    declared: set[str] | None = None,
    context: str = "context",
) -> str:
    """Compile canonical typed expression IR to Java."""
    if not isinstance(expr, Mapping):
        raise ValueError(
            "STRUCTURED_STATE_EXPRESSION: canonical expression must be an object; "
            f"got {type(expr).__name__}"
        )

    kind = str(expr.get("kind") or "").strip()
    if not kind:
        raise ValueError(
            f"STRUCTURED_STATE_EXPRESSION: missing 'kind' in expression object {expr!r}"
        )

    if kind == "literal":
        val = expr.get("value")
        if val is None or val == "null":
            return "null"
        if isinstance(val, bool):
            return "Boolean.TRUE" if val else "Boolean.FALSE"
        if isinstance(val, (int, float)):
            return f"Double.valueOf({val})"
        return _java_string(str(val))

    if kind == "state_ref":
        name = str(expr.get("name") or "").strip()
        if declared is not None and name not in declared:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: undeclared state variable {name!r}")
        return f"$mmmRead({_java_string(name)}, {context})"

    if kind == "context_ref":
        name = str(expr.get("name") or "").strip()
        return f"$mmmRead({_java_string(name)}, {context})"

    if kind == "empty_map":
        return "new java.util.LinkedHashMap<>()"

    if kind == "empty_list":
        return "new java.util.ArrayList<>()"

    if kind == "and":
        terms = expr.get("terms") or []
        if not terms:
            return "true"
        compiled_terms = [
            compile_state_condition_ir(t, declared=declared, context=context)
            for t in terms
        ]
        if len(compiled_terms) == 1:
            return compiled_terms[0]
        return "(" + " && ".join(f"({t})" for t in compiled_terms) + ")"

    if kind == "or":
        terms = expr.get("terms") or []
        if not terms:
            return "false"
        compiled_terms = [
            compile_state_condition_ir(t, declared=declared, context=context)
            for t in terms
        ]
        if len(compiled_terms) == 1:
            return compiled_terms[0]
        return "(" + " || ".join(f"({t})" for t in compiled_terms) + ")"

    if kind == "not":
        term = expr.get("term") or expr.get("operand") or expr.get("left")
        compiled = compile_state_expr_ir(term, declared=declared, context=context)
        return f"(!$mmmTruthy({compiled}))"

    if kind == "compare":
        op = expr.get("op", "==")
        left = compile_state_expr_ir(expr.get("left"), declared=declared, context=context)
        right = compile_state_expr_ir(expr.get("right"), declared=declared, context=context)
        if op == "==":
            return f"$mmmEquals({left}, {right})"
        if op == "!=":
            return f"(!$mmmEquals({left}, {right}))"
        if op in {">=", "<=", ">", "<"}:
            return f"($mmmCompare({left}, {right}) {op} 0)"
        return f"Boolean.valueOf({left} {op} {right})"

    if kind == "arithmetic":
        op = expr.get("op", "+")
        left = compile_state_expr_ir(expr.get("left"), declared=declared, context=context)
        right = compile_state_expr_ir(expr.get("right"), declared=declared, context=context)
        return f"$mmmArithmetic({_java_string(op)}, {left}, {right})"

    if kind == "number":
        value = str(expr.get("value") or "").strip()
        if not re.fullmatch(r"^-?[0-9]+(?:\.[0-9]+)?$", value):
            raise ValueError(
                f"STRUCTURED_STATE_EXPRESSION: invalid number literal {value!r}"
            )
        return f"Double.valueOf({_java_string(value)})"

    if kind == "call":
        name = str(expr.get("name") or "").casefold()
        if name not in _SUPPORTED_STATE_FUNCTIONS:
            raise ValueError(
                f"STRUCTURED_STATE_EXPRESSION: unsupported function {name!r}"
            )
        args = expr.get("args") or ()
        args_compiled = ", ".join(
            compile_state_expr_ir(a, declared=declared, context=context)
            for a in args
        )
        return f"$mmmFunction({_java_string(name)}, java.util.Arrays.asList({args_compiled}), {context})"

    raise ValueError(
        f"STRUCTURED_STATE_EXPRESSION: unknown expression kind {kind!r}"
    )


def _state_expr_value_family(
    expr: Any,
    symbols: StateSymbolTable | None = None,
) -> str:
    """Return a conservative value family for mutation type checking."""

    if expr is None:
        return "null"
    if isinstance(expr, bool):
        return "boolean"
    if isinstance(expr, (int, float)):
        return "number"
    if isinstance(expr, str):
        return "unknown"
    if not isinstance(expr, Mapping):
        return "unknown"

    kind = str(expr.get("kind") or "").strip()
    if kind == "number":
        return "number"
    if kind == "literal":
        value = expr.get("value")
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        return "string"
    if kind == "empty_map":
        return "map"
    if kind == "empty_list":
        return "list"
    if kind == "state_ref" and symbols is not None:
        name = str(expr.get("name") or "").strip()
        return _state_variable_value_kind(symbols.variables.get(name))
    return "unknown"


def validate_mutation_ir(
    mutation: Any,
    *,
    symbols: StateSymbolTable | set[str] | Sequence[str] | None = None,
) -> None:
    """Validate canonical typed mutation IR against declared symbols."""
    if mutation is None:
        return
    symbol_table = symbols if isinstance(symbols, StateSymbolTable) else None
    if isinstance(symbols, StateSymbolTable):
        declared = symbols.declared_names
    elif isinstance(symbols, set):
        declared = symbols
    elif isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes, bytearray)):
        declared = set(symbols)
    else:
        declared = None

    if isinstance(mutation, Mapping):
        mutations = [mutation]
    elif isinstance(mutation, Sequence) and not isinstance(mutation, (str, bytes, bytearray)):
        mutations = list(mutation)
    else:
        raise ValueError(
            f"STRUCTURED_STATE_MUTATION: expected list or object, got {type(mutation).__name__}"
        )

    for item in mutations:
        if not isinstance(item, Mapping):
            raise ValueError(
                f"STRUCTURED_STATE_MUTATION: mutation item must be an object, got {type(item).__name__}"
            )
        target = str(item.get("target") or "").strip()
        if not target:
            raise ValueError("STRUCTURED_STATE_MUTATION: mutation requires non-empty string 'target'")
        if declared is not None and target not in declared:
            raise ValueError(
                f"STRUCTURED_STATE_MUTATION: undeclared state variable {target!r}"
            )
        op = item.get("operator", "=")
        if op not in {"=", "+=", "-=", "*=", "/="}:
            raise ValueError(f"STRUCTURED_STATE_MUTATION: invalid operator {op!r}")
        normalized_op = op
        target_kind = (
            _state_variable_value_kind(symbol_table.variables.get(target))
            if symbol_table is not None
            else "unknown"
        )
        allowed_ops = (
            {"=", "+=", "-=", "*=", "/="}
            if target_kind == "number"
            else {"=", "+="}
            if target_kind == "string"
            else {"="}
            if target_kind in {"boolean", "map", "list"}
            else {"=", "+=", "-=", "*=", "/="}
        )
        if normalized_op not in allowed_ops:
            raise ValueError(
                "STRUCTURED_STATE_MUTATION: operator "
                f"{normalized_op!r} is incompatible with {target!r} "
                f"value family {target_kind!r}"
            )
        val = item.get("value")
        if val is not None:
            validate_state_expr_ir(val, symbols=declared)
            value_kind = _state_expr_value_family(val, symbol_table)
            if (
                target_kind not in {"unknown"}
                and value_kind not in {"unknown", "null", target_kind}
            ):
                raise ValueError(
                    "STRUCTURED_STATE_MUTATION: value family "
                    f"{value_kind!r} is incompatible with target {target!r} "
                    f"family {target_kind!r}"
                )


def compile_mutation_ir(
    mutation: Any,
    *,
    declared: set[str] | None = None,
    context: str = "context",
) -> str:
    """Compile canonical typed mutation IR to Java."""
    if mutation is None:
        return ""

    if isinstance(mutation, Mapping):
        mutations = [mutation]
    elif isinstance(mutation, Sequence) and not isinstance(mutation, (str, bytes, bytearray)):
        mutations = list(mutation)
    else:
        raise ValueError(
            "STRUCTURED_STATE_MUTATION: expected list, object, or null; "
            f"got {type(mutation).__name__}"
        )

    rows: list[str] = []
    for item in mutations:
        if not isinstance(item, Mapping):
            raise ValueError(
                "STRUCTURED_STATE_MUTATION: mutation item must be an object"
            )
        target = str(item.get("target") or "").strip()
        if not target:
            raise ValueError(
                "STRUCTURED_STATE_MUTATION: mutation target is required"
            )
        if declared is not None and target not in declared:
            raise ValueError(
                f"STRUCTURED_STATE_MUTATION: undeclared state variable {target!r}"
            )
        operator = item.get("operator", "=")
        val = item.get("value")
        right = compile_state_expr_ir(val, declared=declared, context=context)
        if operator == "=":
            rows.append(f"setState({_java_string(target)}, {right});")
        else:
            arithmetic = operator[0]
            rows.append(
                f"setState({_java_string(target)}, "
                f"$mmmArithmetic({_java_string(arithmetic)}, "
                f"$mmmRead({_java_string(target)}, {context}), {right}));"
            )
    return " ".join(rows)


def _state_name_schema(symbols: list[str]) -> dict[str, Any]:
    """Unified state-name schema shared by declarations and references.

    When *symbols* are known the schema is a closed ``enum`` — no ``maxLength``
    is needed because the enum already bounds every legal string.  When symbols
    are not yet available (open-world authoring) the schema falls back to
    ``maxLength=128`` which matches the declaration contract.
    """
    if symbols:
        return {"type": "string", "enum": symbols}
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": 128,
        "pattern": r"^[A-Za-z_][A-Za-z0-9_]*$",
    }


def _resolve_state_symbols(allowed_state_symbols: Any) -> list[str]:
    """Normalize symbol sources into a sorted list."""
    if isinstance(allowed_state_symbols, StateSymbolTable):
        return sorted(allowed_state_symbols.declared_names)
    if isinstance(allowed_state_symbols, (set, list, tuple)) and allowed_state_symbols:
        return sorted(set(allowed_state_symbols))
    return []


def _state_symbols_are_authoritative(allowed_state_symbols: Any) -> bool:
    """Whether the caller explicitly fixed the complete state-symbol universe."""

    return isinstance(allowed_state_symbols, StateSymbolTable) or isinstance(
        allowed_state_symbols,
        (set, list, tuple),
    )


def _state_variable_value_kind(record: Mapping[str, Any] | None) -> str:
    """Map authored type prose onto the finite host state-value families."""

    if not isinstance(record, Mapping):
        return "unknown"
    type_name = str(record.get("type") or "").strip()
    if type_name == "boolean":
        return "boolean"
    if type_name in {"int", "long", "double"}:
        return "number"
    if type_name == "string":
        return "string"
    return "unknown"


def _mutation_value_branches(
    allowed_state_symbols: Any,
    *,
    target_kind: str,
) -> list[dict[str, Any]]:
    """Return finite IR alternatives compatible with one declared target family."""

    symbols = _resolve_state_symbols(allowed_state_symbols)
    variables = getattr(allowed_state_symbols, "variables", {})
    compatible_names = [
        name
        for name in symbols
        if target_kind == "unknown"
        or _state_variable_value_kind(
            variables.get(name) if isinstance(variables, Mapping) else None
        ) in {target_kind, "unknown"}
    ]

    context_ref_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "context_ref"},
            "name": {
                "type": "string",
                "maxLength": 24,
                "pattern": r"^[A-Za-z_$][A-Za-z0-9_$.]*$",
                "minLength": 1,
            },
        },
        "required": ["kind", "name"],
        "additionalProperties": False,
    }
    string_literal_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "literal"},
            "value": {
                "type": "string",
                "maxLength": 24,
                "pattern": r"^[^{}\[\]]*$",
                "description": (
                    "One scalar text value only. Never serialize JSON, maps, arrays, "
                    "or another structured object into this string."
                ),
            },
        },
        "required": ["kind", "value"],
        "additionalProperties": False,
    }
    number_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "number"},
            "value": {
                "type": "string",
                "maxLength": 24,
                "pattern": r"^-?[0-9]+(\.[0-9]+)?$",
            },
        },
        "required": ["kind", "value"],
        "additionalProperties": False,
    }
    boolean_literal_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "literal"},
            "value": {"type": "boolean"},
        },
        "required": ["kind", "value"],
        "additionalProperties": False,
    }
    null_literal_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "literal"},
            "value": {"type": "null"},
        },
        "required": ["kind", "value"],
        "additionalProperties": False,
    }
    empty_map_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "empty_map"},
        },
        "required": ["kind"],
        "additionalProperties": False,
    }
    empty_list_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "empty_list"},
        },
        "required": ["kind"],
        "additionalProperties": False,
    }

    leaf_by_kind: list[dict[str, Any]]
    if target_kind == "number":
        leaf_by_kind = [number_branch, context_ref_branch]
    elif target_kind == "boolean":
        leaf_by_kind = [boolean_literal_branch, context_ref_branch]
    elif target_kind == "map":
        leaf_by_kind = [empty_map_branch, context_ref_branch]
    elif target_kind == "list":
        leaf_by_kind = [empty_list_branch, context_ref_branch]
    elif target_kind == "string":
        leaf_by_kind = [string_literal_branch, context_ref_branch]
    else:
        leaf_by_kind = [
            string_literal_branch,
            number_branch,
            boolean_literal_branch,
            empty_map_branch,
            empty_list_branch,
            context_ref_branch,
        ]

    if compatible_names:
        state_ref_name_schema = {
            "type": "string",
            "enum": compatible_names,
        }
    elif not _state_symbols_are_authoritative(allowed_state_symbols):
        state_ref_name_schema = _state_name_schema([])
    else:
        state_ref_name_schema = None

    if state_ref_name_schema is not None:
        leaf_by_kind.append({
            "type": "object",
            "properties": {
                "kind": {"type": "string", "const": "state_ref"},
                "name": state_ref_name_schema,
            },
            "required": ["kind", "name"],
            "additionalProperties": False,
        })

    branches = [*leaf_by_kind, null_literal_branch]
    if target_kind in {"number", "unknown"}:
        numeric_operands = [
            branch
            for branch in leaf_by_kind
            if branch.get("properties", {}).get("kind", {}).get("const")
            in {"number", "state_ref", "context_ref"}
        ]
        if numeric_operands:
            operand = {"oneOf": numeric_operands}
            branches.append({
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "arithmetic"},
                    "op": {
                        "type": "string",
                        "enum": ["+", "-", "*", "/", "%"],
                        "maxLength": 2,
                    },
                    "left": operand,
                    "right": operand,
                },
                "required": ["kind", "op", "left", "right"],
                "additionalProperties": False,
            })
            branches.append({
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "call"},
                    "name": {
                        "type": "string",
                        "enum": sorted(_SUPPORTED_STATE_FUNCTIONS),
                    },
                    "args": {
                        "type": "array",
                        "maxItems": 2,
                        "items": operand,
                    },
                },
                "required": ["kind", "name", "args"],
                "additionalProperties": False,
            })
    elif target_kind == "string":
        string_operands = [
            branch
            for branch in leaf_by_kind
            if branch.get("properties", {}).get("kind", {}).get("const")
            in {"literal", "state_ref", "context_ref"}
        ]
        if string_operands:
            operand = {"oneOf": string_operands}
            branches.append({
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "arithmetic"},
                    "op": {"type": "string", "const": "+"},
                    "left": operand,
                    "right": operand,
                },
                "required": ["kind", "op", "left", "right"],
                "additionalProperties": False,
            })

    return branches

def mutations_schema(allowed_state_symbols: Any = None) -> dict[str, Any]:
    symbols = _resolve_state_symbols(allowed_state_symbols)
    authoritative_symbols = _state_symbols_are_authoritative(
        allowed_state_symbols
    )
    if authoritative_symbols and not symbols:
        return {
            "type": "array",
            "minItems": 0,
            "maxItems": 0,
            "items": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }

    variables = getattr(allowed_state_symbols, "variables", {})
    if symbols:
        item_branches: list[dict[str, Any]] = []
        for target in symbols:
            record = variables.get(target) if isinstance(variables, Mapping) else None
            target_kind = _state_variable_value_kind(record)
            operators = (
                ["=", "+=", "-=", "*=", "/="]
                if target_kind == "number"
                else ["=", "+="]
                if target_kind == "string"
                else ["="]
            )
            item_branches.append({
                "type": "object",
                "properties": {
                    "target": {"type": "string", "const": target},
                    "operator": {
                        "type": "string",
                        "enum": operators,
                        "maxLength": 2,
                    },
                    "value": {
                        "oneOf": _mutation_value_branches(
                            allowed_state_symbols,
                            target_kind=target_kind,
                        )
                    },
                },
                "required": ["target", "operator", "value"],
                "additionalProperties": False,
            })
        item_schema: dict[str, Any] = {"oneOf": item_branches}
    else:
        item_schema = {
            "type": "object",
            "properties": {
                "target": _state_name_schema(symbols),
                "operator": {
                    "type": "string",
                    "enum": ["=", "+=", "-=", "*=", "/="],
                    "maxLength": 2,
                },
                "value": {
                    "oneOf": _mutation_value_branches(
                        allowed_state_symbols,
                        target_kind="unknown",
                    )
                },
            },
            "required": ["target", "operator", "value"],
            "additionalProperties": False,
        }

    return {
        "type": "array",
        "maxItems": 2,
        "items": item_schema,
    }

def state_expr_schema(allowed_state_symbols: Any = None) -> dict[str, Any]:
    symbols = _resolve_state_symbols(allowed_state_symbols)
    authoritative_symbols = _state_symbols_are_authoritative(
        allowed_state_symbols
    )
    name_schema = _state_name_schema(symbols)
    context_name_schema = {
        "type": "string",
        "maxLength": 24,
        "pattern": r"^[A-Za-z_$][A-Za-z0-9_$.]*$",
        "minLength": 1,
    }
    function_name_schema = {
        "type": "string",
        "enum": sorted(_SUPPORTED_STATE_FUNCTIONS),
    }
    literal_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "literal"},
            "value": {
                "type": ["string", "null"],
                "maxLength": 24,
                "pattern": r"^[^{}\[\]]*$",
                "description": (
                    "One scalar text/null literal only. Never serialize JSON, maps, "
                    "arrays, or another structured object into this string."
                ),
            },
        },
        "required": ["kind", "value"],
        "additionalProperties": False,
    }
    number_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "number"},
            "value": {
                "type": "string",
                "maxLength": 24,
                "pattern": r"^-?[0-9]+(\.[0-9]+)?$",
            },
        },
        "required": ["kind", "value"],
        "additionalProperties": False,
    }
    state_ref_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "state_ref"},
            "name": name_schema,
        },
        "required": ["kind", "name"],
        "additionalProperties": False,
    }
    context_ref_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "context_ref"},
            "name": context_name_schema,
        },
        "required": ["kind", "name"],
        "additionalProperties": False,
    }
    leaf_operand_branches = [
        *(
            [state_ref_branch]
            if symbols or not authoritative_symbols
            else []
        ),
        context_ref_branch,
        literal_branch,
        number_branch,
    ]
    leaf_operand = {
        "oneOf": leaf_operand_branches,
    }
    arithmetic_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "arithmetic"},
            "op": {"type": "string", "enum": ["+", "-", "*", "/", "%"], "maxLength": 2},
            "left": leaf_operand,
            "right": leaf_operand,
        },
        "required": ["kind", "op", "left", "right"],
        "additionalProperties": False,
    }
    call_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "call"},
            "name": function_name_schema,
            "args": {
                "type": "array",
                "maxItems": 2,
                "items": leaf_operand,
            },
        },
        "required": ["kind", "name", "args"],
        "additionalProperties": False,
    }
    operand_branches = [
        *leaf_operand_branches,
        arithmetic_branch,
        call_branch,
    ]
    operand_schema = {
        "oneOf": operand_branches,
    }
    compare_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "compare"},
            "op": {"type": "string", "enum": ["==", "!=", ">=", "<=", ">", "<"], "maxLength": 2},
            "left": operand_schema,
            "right": operand_schema,
        },
        "required": ["kind", "op", "left", "right"],
        "additionalProperties": False,
    }
    and_or_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["and", "or"], "maxLength": 16},
            "terms": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "oneOf": [
                        compare_branch,
                        *leaf_operand_branches,
                    ]
                },
            },
        },
        "required": ["kind", "terms"],
        "additionalProperties": False,
    }
    not_branch = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "not"},
            "term": operand_schema,
        },
        "required": ["kind", "term"],
        "additionalProperties": False,
    }
    return {
        "oneOf": [
            compare_branch,
            and_or_branch,
            not_branch,
            *operand_branches,
        ]
    }


def state_concern_schema(
    concern: str,
    *,
    allowed_state_symbols: Any = None,
) -> dict[str, Any]:
    symbols = _resolve_state_symbols(allowed_state_symbols)
    # None means the storage/worksheet schema does not yet have an authoritative
    # symbol table. Preserve that open-world state. An explicit empty
    # StateSymbolTable/list/set means something different: zero state variables
    # are authoritatively declared, so mutation arrays must then be empty.
    symbol_source = allowed_state_symbols

    def expression_field() -> dict[str, Any]:
        return state_expr_schema(symbol_source)

    def mutation_field() -> dict[str, Any]:
        return mutations_schema(symbol_source)
    if concern == "variables":
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                    "pattern": _STATE_IDENTIFIER_PATTERN,
                    "description": "Stable ASCII internal state identifier consumed by the host state compiler.",
                },
                "owner": {"type": "string", "minLength": 1, "maxLength": 256},
                "type": {"type": "string", "enum": list(_STATE_SCALAR_TYPES)},
                "unit": {"type": "string", "minLength": 1, "maxLength": 128},
                "default": state_variable_default_schema(),
                "domain": {"type": "string", "minLength": 1, "maxLength": 256},
            },
            "required": ["name", "owner", "type", "unit", "default", "domain"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "transitions":
        schema = {
            "type": "object",
            "properties": {
                "from_state": {"type": "string", "minLength": 1, "maxLength": 128},
                "trigger": {"type": "string", "minLength": 1, "maxLength": 128},
                "guard": expression_field(),
                "mutation": mutation_field(),
                "to_state": {"type": "string", "minLength": 1, "maxLength": 128},
            },
            "required": ["from_state", "trigger", "guard", "to_state"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "invariants":
        schema = {
            "type": "object",
            "properties": {
                "condition": expression_field(),
                "enforcement": {"type": "string", "minLength": 1, "maxLength": 512},
            },
            "required": ["condition", "enforcement"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "initialization":
        schema = {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "minLength": 1, "maxLength": 256},
                "trigger": {"type": "string", "minLength": 1, "maxLength": 128},
                "initial_state": mutation_field(),
            },
            "required": ["owner", "trigger", "initial_state"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "updates":
        schema = {
            "type": "object",
            "properties": {
                "trigger": {"type": "string", "minLength": 1, "maxLength": 128},
                "mutation": mutations_schema(symbol_source),
                "owner": {"type": "string", "minLength": 1, "maxLength": 256},
            },
            "required": ["trigger", "mutation", "owner"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "cleanup":
        schema = {
            "type": "object",
            "properties": {
                "event": {"type": "string", "minLength": 1, "maxLength": 128},
                "action": mutation_field(),
                "retained_state": {"type": "string", "minLength": 1, "maxLength": 256},
            },
            "required": ["event", "action", "retained_state"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "concurrency":
        schema = {
            "type": "object",
            "properties": {
                "entry_path": {"type": "string", "minLength": 1, "maxLength": 256},
                "ownership": {"type": "string", "minLength": 1, "maxLength": 256},
                "reentrancy_rule": {"type": "string", "minLength": 1, "maxLength": 512},
            },
            "required": ["entry_path", "ownership", "reentrancy_rule"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    raise ValueError(f"Unknown state concern: {concern!r}")


class StateSymbolTable:
    """Canonical symbol table for state variables declared in state_model."""

    def __init__(
        self,
        variables: Sequence[Mapping[str, Any]] | Sequence[str] | set[str] | StateSymbolTable = (),
    ) -> None:
        if isinstance(variables, StateSymbolTable):
            self.declared_names: set[str] = set(variables.declared_names)
            self.variables: dict[str, Mapping[str, Any]] = dict(variables.variables)
            return

        self.declared_names = set()
        self.variables = {}
        for item in variables or ():
            if isinstance(item, Mapping):
                name = str(item.get("name") or "").strip()
                if name:
                    self.variables[name] = item
                    self.declared_names.add(name)
            elif isinstance(item, str):
                name = item.strip()
                if name:
                    self.declared_names.add(name)

    def contains(self, name: str) -> bool:
        return name in self.declared_names

    def __contains__(self, name: str) -> bool:
        return name in self.declared_names

    def __iter__(self):
        return iter(sorted(self.declared_names))

    def __len__(self) -> int:
        return len(self.declared_names)

    def prompt_text(self) -> str:
        if not self.declared_names:
            return ""
        lines = ["Canonical state symbols:", "variables:"]
        for name in sorted(self.declared_names):
            record = self.variables.get(name)
            kind = _state_variable_value_kind(record)
            authored_type = (
                str(record.get("type") or "").strip()
                if isinstance(record, Mapping)
                else ""
            )
            type_text = authored_type or kind
            lines.append(f"- {name}: type={type_text}; value_family={kind}")
        return "\n".join(lines)


def validate_state_concern(
    concern: str,
    records: Sequence[Mapping[str, Any]],
    *,
    symbols: StateSymbolTable | set[str] | Sequence[str] | None = None,
) -> None:
    """Validate a single concern of state_model against the declared symbol table."""
    if isinstance(symbols, StateSymbolTable):
        declared = symbols.declared_names
    elif isinstance(symbols, set):
        declared = symbols
    elif isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes, bytearray)):
        declared = set(symbols)
    else:
        declared = set()

    rows = records if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)) else ()

    if concern == "variables":
        seen: set[str] = set()
        for record in rows:
            if not isinstance(record, Mapping):
                continue
            name = str(record.get("name") or "").strip()
            if re.fullmatch(_STATE_IDENTIFIER_PATTERN, name) is None:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_NAME: {name!r} is not a stable identifier"
                )
            if name in seen:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_DUPLICATE: {name!r}"
                )
            seen.add(name)

            family = _state_variable_value_kind(record)
            default = str(record.get("default") or "").strip()
            if any(0xD800 <= ord(char) <= 0xDFFF for char in default):
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_DEFAULT: {name!r} contains an unpaired surrogate"
                )
            if family == "number" and re.fullmatch(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                default,
            ) is None:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_DEFAULT: {name!r} requires a numeric default"
                )
            if family == "boolean" and default.casefold() not in {"true", "false"}:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_DEFAULT: {name!r} requires true or false"
                )
            if family == "map" and default.casefold() not in {"{}", "empty_map"}:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_DEFAULT: {name!r} supports only an empty map default"
                )
            if family == "list" and default.casefold() not in {"[]", "empty_list", "empty_set"}:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_DEFAULT: {name!r} supports only an empty collection default"
                )
            if family == "string" and any(char in default for char in "{}[]"):
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_DEFAULT: {name!r} may not contain serialized structured data"
                )
        return

    if concern == "transitions":
        for record in rows:
            if not isinstance(record, Mapping):
                continue
            guard_val = record.get("guard")
            if guard_val is not None:
                validate_state_expr_ir(guard_val, symbols=declared)
            mut_val = record.get("mutation")
            if mut_val is not None:
                validate_mutation_ir(mut_val, symbols=symbols)
        return

    if concern == "invariants":
        for record in rows:
            if not isinstance(record, Mapping):
                continue
            cond_val = record.get("condition")
            if cond_val is not None:
                validate_state_expr_ir(cond_val, symbols=declared)
        return

    if concern in {"initialization", "updates", "cleanup"}:
        field = {
            "initialization": "initial_state",
            "updates": "mutation",
            "cleanup": "action",
        }[concern]
        for record in rows:
            if not isinstance(record, Mapping):
                continue
            mut_val = record.get(field)
            if mut_val is not None:
                validate_mutation_ir(mut_val, symbols=symbols)
        return


def validate_structured_state_section(section: Mapping[str, Any]) -> None:
    """Fail closed on canonical state semantics before any production code is generated."""

    raw_specification = section.get("specification")
    specification = (
        raw_specification if isinstance(raw_specification, Mapping) else section
    )
    variables = specification.get("variables", [])
    validate_state_concern("variables", variables)
    symbols = StateSymbolTable(variables)

    for concern in ("transitions", "invariants", "initialization", "updates", "cleanup", "concurrency"):
        records = specification.get(concern, [])
        if records:
            validate_state_concern(concern, records, symbols=symbols)


def _obligations(task: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Decode host obligations without collapsing typed values into prose."""

    raw = task.get("implementation_obligations")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for item in raw:
        try:
            outer = json.loads(str(item))
            instruction = json.loads(str(outer.get("instruction") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if "structured_records" not in outer:
            continue
        name = str(instruction.get("concern") or "").strip()
        records = outer.get("structured_records")
        if not name or not isinstance(records, list):
            continue
        result[name] = [
            {str(key): deepcopy(value) for key, value in record.items()}
            for record in records
            if isinstance(record, Mapping)
        ]
    return result


def has_complete_structured_state(
    task: Mapping[str, Any],
    concerns: Sequence[Mapping[str, Any]],
) -> bool:
    available = _obligations(task)
    return bool(concerns) and all(
        str(item.get("concern") or "") in available
        for item in concerns
    )


def _default_value(record: Mapping[str, str]) -> str:
    type_name = str(record.get("type") or "").strip()
    value = str(record.get("default") or "").strip()
    lowered = value.casefold()
    if type_name == "boolean":
        if lowered not in {"true", "false"}:
            raise ValueError("STRUCTURED_STATE_DEFAULT_BOOLEAN_INVALID")
        return "Boolean.TRUE" if lowered == "true" else "Boolean.FALSE"
    if type_name in {"int", "long"}:
        if re.fullmatch(r"[-+]?\d+", value) is None:
            raise ValueError("STRUCTURED_STATE_DEFAULT_INTEGER_INVALID")
        return f"Long.valueOf({_java_string(value)})"
    if type_name == "double":
        if re.fullmatch(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            value,
        ) is None:
            raise ValueError("STRUCTURED_STATE_DEFAULT_DOUBLE_INVALID")
        return f"Double.valueOf({_java_string(value)})"
    if type_name == "string":
        return _java_string(value)
    raise ValueError(
        f"STRUCTURED_STATE_DEFAULT_TYPE_UNSUPPORTED: {type_name!r}"
    )


_COMMON = r"""
private static final java.util.Map<String, Object> $mmmState =
        new java.util.concurrent.ConcurrentHashMap<>();

@FunctionalInterface
private interface $mmmGuard {
    boolean test(java.util.Map<String, Object> context);
}

@FunctionalInterface
private interface $mmmAction {
    void apply(java.util.Map<String, Object> context);
}

private record $mmmTransition(
        String fromState,
        String trigger,
        $mmmGuard guard,
        $mmmAction action,
        String toState
) {}

private record $mmmInvariant($mmmGuard guard, String enforcement) {}
private record $mmmTriggeredAction(String trigger, $mmmAction action, String owner) {}
private record $mmmCleanupAction(
        String event,
        $mmmAction action,
        String retainedState
) {}

private static final java.util.List<$mmmTransition> $mmmTransitions =
        new java.util.concurrent.CopyOnWriteArrayList<>();
private static final java.util.List<$mmmInvariant> $mmmInvariants =
        new java.util.concurrent.CopyOnWriteArrayList<>();
private static final java.util.List<$mmmTriggeredAction> $mmmInitializers =
        new java.util.concurrent.CopyOnWriteArrayList<>();
private static final java.util.List<$mmmTriggeredAction> $mmmUpdates =
        new java.util.concurrent.CopyOnWriteArrayList<>();
private static final java.util.List<$mmmCleanupAction> $mmmCleanup =
        new java.util.concurrent.CopyOnWriteArrayList<>();
private static final java.util.List<String[]> $mmmConcurrency =
        new java.util.concurrent.CopyOnWriteArrayList<>();

public static synchronized Object getState(String name) {
    return $mmmState.get(name);
}

public static synchronized void setState(String name, Object value) {
    if (name == null || name.isBlank()) {
        return;
    }
    if (value == null) {
        $mmmState.remove(name);
    } else {
        $mmmState.put(name, value);
    }
}

public static synchronized Object getState(
        String name,
        java.util.Map<String, Object> context
) {
    return $mmmRead(name, context);
}

public static synchronized void setState(
        String name,
        Object value,
        java.util.Map<String, Object> context
) {
    setState(name, value);
    if (context == null || name == null || name.isBlank()) {
        return;
    }
    try {
        if (value == null) {
            context.remove(name);
        } else {
            context.put(name, value);
        }
    } catch (UnsupportedOperationException ignored) {
        // Immutable event contexts remain valid read overlays; persistent state
        // was already committed above.
    }
}

private static Object $mmmRead(
        String name,
        java.util.Map<String, Object> context
) {
    if (context != null && context.containsKey(name)) {
        return context.get(name);
    }
    return $mmmState.get(name);
}

private static double $mmmNumber(Object value) {
    if (value instanceof Number number) {
        return number.doubleValue();
    }
    if (value instanceof Boolean bool) {
        return bool ? 1.0d : 0.0d;
    }
    try {
        return Double.parseDouble(String.valueOf(value));
    } catch (RuntimeException ignored) {
        return 0.0d;
    }
}

private static boolean $mmmTruthy(Object value) {
    if (value == null) {
        return false;
    }
    if (value instanceof Boolean bool) {
        return bool;
    }
    if (value instanceof Number number) {
        return number.doubleValue() != 0.0d;
    }
    String text = String.valueOf(value);
    return !text.isBlank() && !"false".equalsIgnoreCase(text) && !"0".equals(text);
}

private static Object $mmmArithmetic(String op, Object left, Object right) {
    if ("+".equals(op) && (left instanceof String || right instanceof String)) {
        return String.valueOf(left) + String.valueOf(right);
    }
    double l = $mmmNumber(left);
    double r = $mmmNumber(right);
    return switch (op) {
        case "+" -> l + r;
        case "-" -> l - r;
        case "*" -> l * r;
        case "/" -> r == 0.0d ? l : l / r;
        case "%" -> r == 0.0d ? l : l % r;
        default -> l;
    };
}

private static boolean $mmmEquals(Object left, Object right) {
    if (left instanceof Number && right instanceof Number) {
        return Double.compare($mmmNumber(left), $mmmNumber(right)) == 0;
    }
    return java.util.Objects.equals(left, right);
}

private static int $mmmCompare(Object left, Object right) {
    if (left instanceof Number || right instanceof Number) {
        return Double.compare($mmmNumber(left), $mmmNumber(right));
    }
    return String.valueOf(left).compareTo(String.valueOf(right));
}

private static double $mmmSumValue(Object value) {
    if (value == null) {
        return 0.0d;
    }
    if (value instanceof java.lang.Iterable<?> iterable) {
        double total = 0.0d;
        for (Object item : iterable) {
            total += $mmmSumValue(item);
        }
        return total;
    }
    if (value instanceof java.util.Map<?, ?> map) {
        double total = 0.0d;
        for (Object item : map.values()) {
            total += $mmmSumValue(item);
        }
        return total;
    }
    if (value.getClass().isArray()) {
        double total = 0.0d;
        int length = java.lang.reflect.Array.getLength(value);
        for (int index = 0; index < length; index++) {
            total += $mmmSumValue(java.lang.reflect.Array.get(value, index));
        }
        return total;
    }
    return $mmmNumber(value);
}

private static int $mmmSize(Object value) {
    if (value == null) {
        return 0;
    }
    if (value instanceof java.util.Collection<?> collection) {
        return collection.size();
    }
    if (value instanceof java.util.Map<?, ?> map) {
        return map.size();
    }
    if (value instanceof java.lang.CharSequence text) {
        return text.length();
    }
    if (value.getClass().isArray()) {
        return java.lang.reflect.Array.getLength(value);
    }
    return 1;
}

private static Object $mmmFunction(
        String name,
        java.util.List<Object> args,
        java.util.Map<String, Object> context
) {
    return switch (name) {
        case "sum" -> {
            double total = 0.0d;
            for (Object arg : args) {
                total += $mmmSumValue(arg);
            }
            yield total;
        }
        case "min" -> {
            double result = Double.POSITIVE_INFINITY;
            for (Object arg : args) {
                result = Math.min(result, $mmmNumber(arg));
            }
            yield result == Double.POSITIVE_INFINITY ? 0.0d : result;
        }
        case "max" -> {
            double result = Double.NEGATIVE_INFINITY;
            for (Object arg : args) {
                result = Math.max(result, $mmmNumber(arg));
            }
            yield result == Double.NEGATIVE_INFINITY ? 0.0d : result;
        }
        case "abs" -> Math.abs(args.isEmpty() ? 0.0d : $mmmNumber(args.get(0)));
        case "count", "size", "len" ->
                args.isEmpty() ? 0 : $mmmSize(args.get(0));
        default -> throw new IllegalArgumentException(
                "Unsupported structured-state function: " + name
        );
    };
}

public static synchronized String transition(
        String fromState,
        String trigger,
        java.util.Map<String, Object> context
) {
    for ($mmmTransition transition : $mmmTransitions) {
        if (java.util.Objects.equals(transition.fromState(), fromState)
                && java.util.Objects.equals(transition.trigger(), trigger)
                && transition.guard().test(context)) {
            transition.action().apply(context);
            return transition.toState();
        }
    }
    return fromState;
}

public static synchronized boolean invariantsHold(
        java.util.Map<String, Object> context
) {
    for ($mmmInvariant invariant : $mmmInvariants) {
        if (!invariant.guard().test(context)) {
            return false;
        }
    }
    return true;
}

public static synchronized java.util.List<String> invariantFailures(
        java.util.Map<String, Object> context
) {
    java.util.List<String> failures = new java.util.ArrayList<>();
    for ($mmmInvariant invariant : $mmmInvariants) {
        if (!invariant.guard().test(context)) {
            failures.add(invariant.enforcement());
        }
    }
    return java.util.List.copyOf(failures);
}

public static synchronized void initializeState(
        String trigger,
        java.util.Map<String, Object> context
) {
    for ($mmmTriggeredAction row : $mmmInitializers) {
        if (java.util.Objects.equals(row.trigger(), trigger)) {
            row.action().apply(context);
        }
    }
}

public static synchronized void applyUpdate(
        String trigger,
        java.util.Map<String, Object> context
) {
    for ($mmmTriggeredAction row : $mmmUpdates) {
        if (java.util.Objects.equals(row.trigger(), trigger)) {
            row.action().apply(context);
        }
    }
}

public static synchronized void cleanupState(
        String event,
        java.util.Map<String, Object> context
) {
    for ($mmmCleanupAction row : $mmmCleanup) {
        if (java.util.Objects.equals(row.event(), event)) {
            row.action().apply(context);
        }
    }
}

public static synchronized java.util.List<String[]> concurrencyRules() {
    java.util.List<String[]> result = new java.util.ArrayList<>();
    for (String[] row : $mmmConcurrency) {
        result.add(row.clone());
    }
    return java.util.List.copyOf(result);
}
""".strip()


def render_state_model_concern(
    task: Mapping[str, Any],
    concern: str,
    *,
    include_runtime: bool,
) -> str | None:
    records_by_concern = _obligations(task)
    if concern not in records_by_concern:
        return None
    records = records_by_concern[concern]
    variables = records_by_concern.get("variables", [])
    declared = {
        str(record.get("name") or "").strip()
        for record in variables
        if str(record.get("name") or "").strip()
    }
    parts: list[str] = [_COMMON] if include_runtime else []

    def executable(index: int, record: Mapping[str, Any], field: str) -> str:
        del index
        if field in {"guard", "condition"}:
            return compile_state_condition_ir(
                record.get(field),
                declared=declared,
            )
        return compile_mutation_ir(record.get(field), declared=declared)

    if concern == "variables":
        lines = ["static {"]
        for record in records:
            name = str(record.get("name") or "").strip()
            if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name) is None:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_NAME: {name!r} is not a stable identifier"
                )
            default = _default_value(record)
            if default != "null":
                lines.append(
                    f"    $mmmState.put({_java_string(name)}, {default});"
                )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "transitions":
        lines = ["static {"]
        for index, record in enumerate(records):
            guard = executable(index, record, "guard")
            mutation = executable(index, record, "mutation")
            lines.append(
                "    $mmmTransitions.add(new $mmmTransition("
                + ", ".join((
                    _java_string(record.get("from_state", "")),
                    _java_string(record.get("trigger", "")),
                    f"context -> ({guard})",
                    f"context -> {{ {mutation} }}",
                    _java_string(record.get("to_state", "")),
                ))
                + "));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "invariants":
        lines = ["static {"]
        for index, record in enumerate(records):
            condition = executable(index, record, "condition")
            lines.append(
                "    $mmmInvariants.add(new $mmmInvariant("
                f"context -> ({condition}), "
                f"{_java_string(record.get('enforcement', ''))}));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "initialization":
        lines = ["static {"]
        for index, record in enumerate(records):
            action = executable(index, record, "initial_state")
            lines.append(
                "    $mmmInitializers.add(new $mmmTriggeredAction("
                f"{_java_string(record.get('trigger', ''))}, "
                f"context -> {{ {action} }}, "
                f"{_java_string(record.get('owner', ''))}));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "updates":
        lines = ["static {"]
        for index, record in enumerate(records):
            action = executable(index, record, "mutation")
            lines.append(
                "    $mmmUpdates.add(new $mmmTriggeredAction("
                f"{_java_string(record.get('trigger', ''))}, "
                f"context -> {{ {action} }}, "
                f"{_java_string(record.get('owner', ''))}));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "cleanup":
        lines = ["static {"]
        for index, record in enumerate(records):
            action = executable(index, record, "action")
            lines.append(
                "    $mmmCleanup.add(new $mmmCleanupAction("
                f"{_java_string(record.get('event', ''))}, "
                f"context -> {{ {action} }}, "
                f"{_java_string(record.get('retained_state', ''))}));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "concurrency":
        lines = ["static {"]
        for record in records:
            values = ", ".join(
                _java_string(record.get(field, ""))
                for field in ("entry_path", "ownership", "reentrancy_rule")
            )
            lines.append(
                f"    $mmmConcurrency.add(new String[]{{{values}}});"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    else:
        return None
    return "\n\n".join(part for part in parts if part).strip()


PUBLIC_API = (
    "public static synchronized Object getState(String name)",
    "public static synchronized Object getState(String name, java.util.Map<String, Object> context)",
    "public static synchronized void setState(String name, Object value)",
    "public static synchronized void setState(String name, Object value, java.util.Map<String, Object> context)",
    "public static synchronized String transition(String fromState, String trigger, java.util.Map<String, Object> context)",
    "public static synchronized boolean invariantsHold(java.util.Map<String, Object> context)",
    "public static synchronized java.util.List<String> invariantFailures(java.util.Map<String, Object> context)",
    "public static synchronized void initializeState(String trigger, java.util.Map<String, Object> context)",
    "public static synchronized void applyUpdate(String trigger, java.util.Map<String, Object> context)",
    "public static synchronized void cleanupState(String event, java.util.Map<String, Object> context)",
    "public static synchronized java.util.List<String[]> concurrencyRules()",
)


__all__ = [
    "PUBLIC_API",
    "STATE_EXPRESSION_PATTERN",
    "STATE_MUTATION_PATTERN",
    "StateSymbolTable",
    "compile_mutation_ir",
    "compile_state_expr_ir",
    "constrain_state_chunk_schema",
    "constrain_state_record_schema",
    "has_complete_structured_state",
    "mutations_schema",
    "render_state_model_concern",
    "state_concern_schema",
    "state_expr_schema",
    "validate_mutation_ir",
    "validate_state_concern",
    "validate_state_expr_ir",
    "validate_structured_state_section",
]
