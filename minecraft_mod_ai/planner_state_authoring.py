"""Author bounded semantic state choices; the host owns executable DSL syntax."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .fixed_template_generation import generate_fixed_template_value
from .planning_detail_slots import record_field_schema
from .structured_state_runtime import (
    _SUPPORTED_STATE_FUNCTIONS,
    StateSymbolTable,
    validate_mutation_ir,
    validate_state_expression,
)

STATE_EXPRESSION_FIELDS = frozenset({"guard", "condition"})
STATE_MUTATION_FIELDS = frozenset({"mutation", "initial_state", "action"})
STATE_EXECUTABLE_FIELDS = STATE_EXPRESSION_FIELDS | STATE_MUTATION_FIELDS
_MAX_EXPRESSION_NODES = 31
_BINARY_OPERATORS = ("+", "-", "*", "/", "%", "==", "!=", ">=", "<=", ">", "<", "&&", "||", "->")


def _choice(values: Sequence[Any]) -> dict[str, Any]:
    return {"type": "integer" if type(values[0]) is int else "string", "enum": list(values)}


class _StateAuthor:
    def __init__(self, router: Any, messages: Sequence[Mapping[str, Any]], symbols: Any) -> None:
        self.router = router
        self.messages = tuple(messages)
        self.names = sorted(StateSymbolTable(symbols or ()).declared_names)
        self.accepted: dict[str, Any] = {}
        self.nodes = 0

    def decide(self, path: str, instruction: str, properties: dict[str, Any]) -> dict[str, Any]:
        schema = {
            "type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False,
        }
        choices = "; ".join(
            f"{name}: {', '.join(str(value) for value in child['enum'])}"
            for name, child in properties.items() if "enum" in child
        )
        value = generate_fixed_template_value(
            self.router, "planner", (*self.messages, {
                "role": "system",
                "content": (
                    f"Current semantic state component: {path}. {instruction}\n"
                    "For this call return only the component fields: " + ", ".join(properties)
                    + ". The host assembles the outer worksheet row and all DSL syntax. "
                    "Do not return a worksheet page or a source-code fragment.\n"
                    + choices + "\nAccepted components (immutable): "
                    + json.dumps(self.accepted, ensure_ascii=False, separators=(",", ":"))
                ),
            }),
            response_schema=schema, enable_tools=False,
            description=f"Author state component {path}.",
        )
        self.accepted[path] = value
        return value

    def expression(self, path: str) -> str:
        self.nodes += 1
        if self.nodes > _MAX_EXPRESSION_NODES:
            raise ValueError(f"PLANNER_STATE_EXPRESSION_BOUND: {path} exceeds {_MAX_EXPRESSION_NODES} nodes")
        kind = self.decide(path, "Choose the semantic expression kind.", {
            "kind": _choice(("number", "string", "boolean", "null", "empty_map", "empty_list",
                             "state_ref", "context_ref", "binary", "not", "negate", "call")),
        })["kind"]
        if kind in {"null", "empty_map", "empty_list"}:
            return {"null": "null", "empty_map": "{}", "empty_list": "[]"}[kind]
        if kind == "binary":
            operator = self.decide(path + ".operator", "Choose the binary operation.", {
                "operator": _choice(_BINARY_OPERATORS),
            })["operator"]
            left = self.expression(path + ".left")
            right = self.expression(path + ".right")
            return f"({left} {operator} {right})"
        if kind in {"not", "negate"}:
            operand = self.expression(path + ".operand")
            return f"({'!' if kind == 'not' else '-'}({operand}))"
        if kind == "call":
            return self.function_call(path)
        if kind == "state_ref":
            if not self.names:
                raise ValueError("PLANNER_STATE_SYMBOLS_REQUIRED: state reference has no declared variables")
            scalar = _choice(self.names)
            instruction = "Select an already declared state variable."
        elif kind == "context_ref":
            scalar = {"type": "string", "minLength": 1, "maxLength": 64,
                      "pattern": r"^[A-Za-z_$][A-Za-z0-9_$.]*$"}
            instruction = "Name the event/context input described by this record; use an ASCII identifier."
        elif kind == "number":
            scalar = {"type": "string", "minLength": 1, "maxLength": 24,
                      "pattern": r"^-?[0-9]+([.][0-9]+)?$"}
            instruction = "Choose a numeric literal as decimal text, for example 100 or -0.5."
        elif kind == "boolean":
            scalar = {"type": "boolean"}
            instruction = "Choose the boolean literal."
        else:
            scalar = {"type": "string", "maxLength": 64}
            instruction = "Choose the literal string content; the host quotes and escapes it."
        value = self.decide(path + ".value", instruction, {"value": scalar})["value"]
        if kind in {"string", "boolean"}:
            return json.dumps(value, ensure_ascii=False)
        return value

    def function_call(self, path: str) -> str:
        name = self.decide(path + ".call", "Choose a host function.", {
            "name": _choice(sorted(_SUPPORTED_STATE_FUNCTIONS)),
        })["name"]
        # These runtime functions consume exactly one argument. Letting a model
        # choose their arity would silently discard every later operand.
        count = 1
        if name in {"sum", "min", "max"}:
            count = self.decide(path + ".arity", "Choose the number of function arguments.", {
                "count": _choice(tuple(range(1, 5))),
            })["count"]
        args = [self.expression(f"{path}.args[{index}]") for index in range(count)]
        return f"{name}({', '.join(args)})"

    def field(self, path: str, field: str) -> str:
        if field in STATE_EXPRESSION_FIELDS:
            value = self.expression(path)
            validate_state_expression(value)
            return value
        count = self.decide(path, "Choose the number of state assignments needed, 0 for no mutation.", {
            "count": _choice(tuple(range(9))),
        })["count"]
        if count and not self.names:
            raise ValueError("PLANNER_STATE_SYMBOLS_REQUIRED: mutation has no declared variables")
        statements = []
        for index in range(count):
            binding = f"{path}.assignments[{index}]"
            assignment = self.decide(binding, "Select the declared target and assignment operation.", {
                "target": _choice(self.names), "operator": _choice(("=", "+=", "-=", "*=", "/=")),
            })
            self.nodes = 0
            expression = self.expression(binding + ".value")
            statements.append(f"{assignment['target']} {assignment['operator']} {expression}")
        value = "; ".join(statements)
        validate_mutation_ir(value, symbols=set(self.names))
        return value


def author_state_field_page(
    router: Any, messages: Sequence[Mapping[str, Any]], *, concern: str,
    field: str, count: int, symbols: Any,
) -> dict[str, Any]:
    """Return canonical string rows after finite, validated semantic decisions."""
    rows = []
    max_length = record_field_schema("state_model", concern, field)["maxLength"]
    for index in range(count):
        author = _StateAuthor(router, (*messages, {
            "role": "system",
            "content": f"Author row {index + 1} of {count}. Prior completed rows: "
            + json.dumps(rows, ensure_ascii=False),
        }), symbols)
        path = f"state_model.{concern}[{index}].{field}"
        value = author.field(path, field)
        if len(value) > max_length:
            raise ValueError(f"PLANNER_STATE_FIELD_BOUND: {path} exceeds canonical {max_length} characters")
        rows.append({field: value})
    return {concern: rows}
