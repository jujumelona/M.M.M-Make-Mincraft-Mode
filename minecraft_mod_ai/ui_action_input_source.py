"""Derive UI input names and scalar types from source-owned state expressions."""
from __future__ import annotations

from collections.abc import Mapping
import sys

from .typed_action_contract import SCALARS

def _literal_type(node):
    value = node.get("value")
    if node.get("kind") == "number":
        return "double" if any(c in str(value).lower() for c in ".e") else "int"
    return {bool: "boolean", str: "string", float: "double", int: "int"}.get(type(value))


def _expression_type(node, state_types, inputs, *, literals=False):
    if not isinstance(node, Mapping):
        return None
    kind = node.get("kind")
    if kind == "state_ref":
        return state_types.get(node.get("name"))
    if kind == "context_ref":
        return inputs.get(node.get("name"))
    if kind in {"and", "or", "not", "compare"}:
        return "boolean"
    if kind in {"number", "literal"} and literals:
        return _literal_type(node)
    if kind == "arithmetic":
        return (_expression_type(node.get("left"), state_types, inputs, literals=literals)
                or _expression_type(node.get("right"), state_types, inputs, literals=literals))
    return None


def _comparison_type(operands, state_types, inputs, *, literals):
    for operand in operands:
        if isinstance(operand, Mapping) and operand.get("kind") == "state_ref":
            return state_types.get(operand.get("name"))
    for operand in operands:
        typ = _expression_type(operand, state_types, inputs)
        if typ:
            return typ
    if literals:
        for operand in operands:
            typ = _expression_type(operand, state_types, inputs, literals=True)
            if typ:
                return typ
    return None


def _collect_inputs(node, expected, state_types, inputs, *, literals):
    if not isinstance(node, Mapping):
        return
    kind = node.get("kind")
    if kind == "context_ref":
        name = node.get("name")
        if expected is None:
            return  # A later authoritative expression can bind this input.
        if expected not in SCALARS:
            raise ValueError("ACTION_SOURCE_INPUT_TYPE_UNBOUND: " + str(name))
        if name in inputs and inputs[name] != expected:
            raise ValueError("ACTION_SOURCE_INPUT_TYPE_CONFLICT: " + str(name))
        inputs[name] = expected
    elif kind == "arithmetic":
        for side in ("left", "right"):
            _collect_inputs(node.get(side), expected, state_types, inputs, literals=literals)
    elif kind == "compare":
        operands = (node.get("left"), node.get("right"))
        typ = _comparison_type(operands, state_types, inputs, literals=literals)
        for operand in operands:
            _collect_inputs(operand, typ, state_types, inputs, literals=literals)
    elif kind in {"and", "or"}:
        for term in node.get("terms", ()):
            _collect_inputs(term, "boolean", state_types, inputs, literals=literals)
    elif kind == "not":
        _collect_inputs(node.get("term") or node.get("operand"),
                        "boolean", state_types, inputs, literals=literals)


def _input_contract(name, typ):
    row = {"name": name, "type": typ, "label": name.replace("_", " ")}
    if typ in {"int", "long"}:
        bits = 32 if typ == "int" else 64
        row.update(minimum=-(2 ** (bits - 1)), maximum=2 ** (bits - 1) - 1)
    elif typ == "double":
        row.update(minimum=-sys.float_info.max, maximum=sys.float_info.max)
    elif typ == "string":
        row.update(max_length=160, choices=[])
    return row


def _source_rules(record, state, interactions):
    trigger = str(record.get("trigger") or "").strip().casefold()
    if not trigger:
        return None
    if sum(str(row.get("trigger") or "").strip().casefold() == trigger
           for row in interactions) > 1:
        # Trigger equality cannot distinguish two different transactions.
        return None
    return [row for concern in ("updates", "transitions")
             for row in state.get(concern, ())
             if str(row.get("trigger") or "").strip().casefold() == trigger
             and row.get("mutation")]


def _source_expressions(rules, state_types):
    expressions = []
    for rule in rules:
        for mutation in rule["mutation"]:
            target = mutation.get("target")
            typ = state_types.get(target)
            if typ not in SCALARS:
                raise ValueError("ACTION_HOST_STATE_UNBOUND: " + str(target))
            expressions.append((mutation.get("value"), typ))
        if rule.get("guard") is not None:
            expressions.append((rule["guard"], "boolean"))
    return expressions


def source_action_inputs(record, state, interactions, *, validate_expression):
    """Derive scalar form fields from normalized same-trigger state rules.

    State reads never become client values. Only context_ref expressions in
    executable writes and their guards introduce fields. The caller supplies
    the existing expression compiler to validate the inferred environment.
    """
    rules = _source_rules(record, state, interactions)
    if not rules:
        return None
    state_types = {row["name"]: row["type"] for row in state.get("variables", ())}
    expressions = _source_expressions(rules, state_types)
    inputs = {}
    # Gather all authoritative constraints before using a literal's default
    # family. Revisit comparisons when a later mutation binds one operand;
    # this monotone process adds at most one entry per source input.
    for literals in (False, True):
        while True:
            before = len(inputs)
            for node, typ in expressions:
                _collect_inputs(node, typ, state_types, inputs, literals=literals)
            if len(inputs) == before:
                break
    for node, typ in expressions:
        validate_expression(node, typ, inputs, state_types)
    return [_input_contract(name, typ) for name, typ in inputs.items()]
