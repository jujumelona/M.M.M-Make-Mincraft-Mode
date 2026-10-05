"""Author bounded semantic state choices; the host owns executable DSL syntax."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .fixed_template_generation import generate_fixed_template_value
from .model_output_atomicity_contract import structured_output_token_ceiling
from .planning_detail_slots import record_field_schema
from .structured_state_runtime import (
    _SUPPORTED_STATE_FUNCTIONS,
    StateSymbolTable,
    mutations_schema,
    state_expr_schema,
    validate_mutation_ir,
    validate_state_expr_ir,
    validate_state_expression,
)

STATE_EXPRESSION_FIELDS = frozenset({"guard", "condition"})
STATE_MUTATION_FIELDS = frozenset({"mutation", "initial_state", "action"})
STATE_EXECUTABLE_FIELDS = STATE_EXPRESSION_FIELDS | STATE_MUTATION_FIELDS
_MAX_EXPRESSION_NODES = 31


def _count_ir_nodes(expr: Any) -> int:
    if not isinstance(expr, Mapping):
        return 1
    count = 1
    for k in ("left", "right", "term", "operand"):
        if k in expr and expr[k] is not None:
            count += _count_ir_nodes(expr[k])
    for item in expr.get("terms") or ():
        count += _count_ir_nodes(item)
    for arg in expr.get("args") or ():
        count += _count_ir_nodes(arg)
    if "value" in expr and isinstance(expr["value"], Mapping):
        count += _count_ir_nodes(expr["value"])
    return count


def lower_state_expr_to_dsl(expr: Any) -> str:
    """Lower expression IR or literal to host DSL string."""
    if expr is None:
        return "true"
    if isinstance(expr, bool):
        return "true" if expr else "false"
    if isinstance(expr, (int, float)):
        return str(expr)
    if isinstance(expr, str):
        return expr.strip()
    if not isinstance(expr, Mapping):
        return "true"

    if _count_ir_nodes(expr) > _MAX_EXPRESSION_NODES:
        raise ValueError(
            f"PLANNER_STATE_EXPRESSION_BOUND: expression exceeds {_MAX_EXPRESSION_NODES} nodes"
        )

    kind = expr.get("kind")
    if not kind:
        if "terms" in expr:
            kind = "and"
        elif "op" in expr and "left" in expr and "right" in expr:
            kind = "compare" if expr["op"] in {"==", "!=", ">=", "<=", ">", "<", "="} else "arithmetic"
        elif "name" in expr:
            kind = "state_ref"
        elif "value" in expr:
            kind = "literal"
        else:
            kind = "literal"

    if kind == "number":
        val = str(expr.get("value") or "").strip()
        if not re.fullmatch(r"^-?[0-9]+([.][0-9]+)?$", val):
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: invalid number literal {val!r}")
        return val

    if kind == "literal":
        val = expr.get("value")
        if val is None or val == "null":
            return "null"
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, (int, float)):
            return str(val)
        val_str = str(val).strip()
        if val_str in {"true", "false", "null"}:
            return val_str
        try:
            float(val_str)
            return val_str
        except ValueError:
            return json.dumps(val_str, ensure_ascii=True)

    if kind == "empty_map":
        return "{}"

    if kind == "empty_list":
        return "[]"

    if kind in {"state_ref", "context_ref"}:
        return str(expr.get("name") or "").strip()

    if kind == "compare":
        op = str(expr.get("op") or "==").strip()
        op = "==" if op == "=" else op
        left = lower_state_expr_to_dsl(expr.get("left"))
        right = lower_state_expr_to_dsl(expr.get("right"))
        return f"({left} {op} {right})"

    if kind in {"and", "or"}:
        terms = expr.get("terms") or []
        if not terms:
            return "true" if kind == "and" else "false"
        lowered = [lower_state_expr_to_dsl(t) for t in terms]
        if len(lowered) == 1:
            return lowered[0]
        joiner = " && " if kind == "and" else " || "
        return f"({joiner.join(lowered)})"

    if kind in {"not", "negate"}:
        term = expr.get("term") or expr.get("operand") or expr.get("left")
        prefix = "!" if kind == "not" else "-"
        return f"({prefix}({lower_state_expr_to_dsl(term)}))"

    if kind == "arithmetic":
        op = str(expr.get("op") or "+").strip()
        left = lower_state_expr_to_dsl(expr.get("left"))
        right = lower_state_expr_to_dsl(expr.get("right"))
        return f"({left} {op} {right})"

    if kind == "call":
        name = str(expr.get("name") or "").strip()
        args = expr.get("args") or []
        if not isinstance(args, (list, tuple)):
            args = [args]
        if name in {"abs", "count", "size", "len"}:
            args = args[:1]
        lowered_args = ", ".join(lower_state_expr_to_dsl(a) for a in args)
        return f"{name}({lowered_args})"

    return "true"


def lower_mutations_to_dsl(mutations: Any) -> str:
    """Lower mutation IR list to host DSL string."""
    if mutations is None:
        return ""
    if isinstance(mutations, str):
        return mutations.strip()
    if isinstance(mutations, Mapping):
        mut_list = [mutations]
    elif isinstance(mutations, Sequence) and not isinstance(mutations, (str, bytes, bytearray)):
        mut_list = list(mutations)
    else:
        return ""

    statements: list[str] = []
    for item in mut_list:
        if not isinstance(item, Mapping):
            continue
        target = str(item.get("target") or "").strip()
        if not target:
            continue
        op = str(item.get("operator") or "=").strip()
        op = "=" if op == ":" else op
        val = item.get("value")
        val_dsl = lower_state_expr_to_dsl(val)
        statements.append(f"{target} {op} {val_dsl}")
    return "; ".join(statements)


def author_state_field_page(
    router: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    concern: str,
    field: str,
    count: int,
    symbols: Any,
) -> dict[str, Any]:
    """Return canonical string rows after finite, validated semantic decisions."""
    if count <= 0:
        return {concern: []}

    symbols_table = (
        symbols
        if isinstance(symbols, StateSymbolTable)
        else StateSymbolTable(symbols or ())
    )
    declared_names = sorted(symbols_table.declared_names)
    max_length = record_field_schema("state_model", concern, field)["maxLength"]
    rows: list[dict[str, str]] = []

    for index in range(count):
        path = f"state_model.{concern}[{index}].{field}"
        if field in STATE_EXPRESSION_FIELDS:
            inner_schema = state_expr_schema(symbols_table)
            row_schema = {
                "type": "object",
                "properties": {field: inner_schema},
                "required": [field],
                "additionalProperties": False,
            }
            instruction = (
                f"Author state expression for {path}. "
                f"Return complete semantic IR object. "
                f"Declared state variables: {', '.join(declared_names) or 'none'}."
            )
            raw = generate_fixed_template_value(
                router,
                "planner",
                (*messages, {
                    "role": "system",
                    "content": instruction,
                }),
                response_schema=row_schema,
                enable_tools=False,
                description=f"Author state {field} for row {index + 1} of {count} in {concern}.",
                output_token_ceiling=structured_output_token_ceiling(row_schema),
            )
            raw_value = raw.get(field, raw) if isinstance(raw, Mapping) else raw
            validate_state_expr_ir(raw_value, symbols=symbols_table)
            value = lower_state_expr_to_dsl(raw_value)
            validate_state_expression(value)
        else:
            inner_schema = mutations_schema(symbols_table)
            row_schema = {
                "type": "object",
                "properties": {field: inner_schema},
                "required": [field],
                "additionalProperties": False,
            }
            typed_symbols = symbols_table.prompt_text()
            instruction = (
                f"Author state mutation for {path}. "
                f"Return list of state assignments or empty list for no state mutation. "
                "Each assignment value must use the host IR branch compatible with its "
                "declared target type. Never serialize a JSON object/array into a string "
                "literal; compound state must use an explicitly supported container IR. "
                f"Declared state variables: {', '.join(declared_names) or 'none'}."
                + (f"\n{typed_symbols}" if typed_symbols else "")
            )
            raw = generate_fixed_template_value(
                router,
                "planner",
                (*messages, {
                    "role": "system",
                    "content": instruction,
                }),
                response_schema=row_schema,
                enable_tools=False,
                description=f"Author state {field} for row {index + 1} of {count} in {concern}.",
                output_token_ceiling=structured_output_token_ceiling(row_schema),
            )
            raw_value = raw.get(field, raw) if isinstance(raw, Mapping) else raw
            validate_mutation_ir(raw_value, symbols=symbols_table)
            value = lower_mutations_to_dsl(raw_value)
            if value:
                validate_mutation_ir(value, symbols=symbols_table)

        if len(value) > max_length:
            raise ValueError(
                f"PLANNER_STATE_FIELD_BOUND: {path} exceeds canonical {max_length} characters"
            )
        rows.append({field: value})

    return {concern: rows}
