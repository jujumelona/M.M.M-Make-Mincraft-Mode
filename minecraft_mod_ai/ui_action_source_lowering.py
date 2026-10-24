"""Deterministic compiler for source-owned typed UI transactions.

The host refuses guessed effects, cross-trigger state writes, and duplicate
transactions. No recursive model authoring is performed in this module.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from .request_record_ownership import same_action_source


def _literal(typ, value):
    from .typed_plan_authoring import (
        _decode_double_literal, _decode_int_literal, _decode_long_literal,
    )
    if typ == "int":
        value = _decode_int_literal(value, scope="ui_action.literal")
    elif typ == "long":
        value = _decode_long_literal(value, scope="ui_action.literal")
    elif typ == "double":
        value = _decode_double_literal(value, scope="ui_action.literal")
    elif typ == "boolean":
        if type(value) is not bool:
            raise ValueError("ACTION_HOST_LITERAL_TYPE: expected boolean")
    elif typ == "string":
        if not isinstance(value, str):
            raise ValueError("ACTION_HOST_LITERAL_TYPE: expected string")
    else:
        raise ValueError("ACTION_HOST_LITERAL_TYPE: unsupported " + str(typ))
    return {"op": "literal", "type": typ, "value": value}


def _selected_mutations(record, owned_refs, sections):
    from .authored_structured_design import active_concern_records

    state = active_concern_records(sections, "state_model")
    available = [
        (f"state_model.{concern}[{index}]", row)
        for concern in ("updates", "transitions")
        for index, row in enumerate(state.get(concern, ()))
        if isinstance(row, Mapping)
    ]
    selected = [(ref, row) for ref, row in available if ref in owned_refs]
    if any(not same_action_source(row, record)
           for _, row in selected):
        raise ValueError(
            "ACTION_HOST_OWNER_TRIGGER_MISMATCH: a runtime owner references "
            "a state rule belonging to a different UI trigger"
        )
    if not selected and not owned_refs:
        interactions = active_concern_records(sections, "resources_and_ui").get(
            "interactions", ()
        )
        actions = [
            row for row in interactions
            if isinstance(row, Mapping)
            and isinstance(row.get("server_action"), Mapping)
        ]
        if len(actions) == 1 and len(available) == 1:
            candidate = available[0]
            if candidate[1].get("trigger") and same_action_source(candidate[1], record):
                selected = [candidate]
    if not selected:
        raise ValueError(
            "ACTION_HOST_LOWERING_UNAVAILABLE: no uniquely assigned canonical "
            "state mutation for this UI action"
        )
    signatures = [
        json.dumps(row.get("mutation"), sort_keys=True, ensure_ascii=False)
        for _, row in selected
    ]
    if len(signatures) != len(set(signatures)):
        raise ValueError(
            "ACTION_HOST_DUPLICATE_MUTATION: identical transaction records "
            "were assigned to the same UI action"
        )
    return selected


def _comparison(node, inputs, state_types):
    op = node.get("op")
    if op not in {"==", "!=", "<", "<=", ">", ">="}:
        raise ValueError("ACTION_HOST_COMPARE_OPERATOR")
    left, right = node.get("left"), node.get("right")
    hint = None
    for operand in (left, right):
        if not isinstance(operand, Mapping):
            continue
        if operand.get("kind") == "state_ref":
            hint = state_types.get(operand.get("name"))
            break
        if operand.get("kind") == "context_ref":
            hint = inputs.get(operand.get("name"))
            break
    if hint is None:
        raise ValueError("ACTION_HOST_COMPARE_UNTYPED")
    return {
        "op": "binary", "operator": op,
        "left": _expression(left, hint, inputs, state_types),
        "right": _expression(right, hint, inputs, state_types),
    }


def _boolean_terms(node, inputs, state_types):
    terms = node.get("terms")
    if not isinstance(terms, list) or len(terms) < 2:
        raise ValueError("ACTION_HOST_BOOLEAN_TERMS")
    acc = _expression(terms[0], "boolean", inputs, state_types)
    for term in terms[1:]:
        acc = {
            "op": "binary",
            "operator": "&&" if node["kind"] == "and" else "||",
            "left": acc,
            "right": _expression(term, "boolean", inputs, state_types),
        }
    return acc


def _expression(node, expected, inputs, state_types):
    if not isinstance(node, Mapping):
        raise ValueError("ACTION_HOST_EXPRESSION_REQUIRED: canonical state IR")
    kind = node.get("kind")
    if kind == "context_ref":
        name = str(node.get("name") or "")
        if name not in inputs or (expected and inputs[name] != expected):
            raise ValueError("ACTION_HOST_INPUT_UNBOUND: " + name)
        return {"op": "ref", "name": name}
    if kind == "state_ref":
        name = str(node.get("name") or "")
        if name not in state_types or (expected and state_types[name] != expected):
            raise ValueError("ACTION_HOST_STATE_UNBOUND: " + name)
        return {
            "op": "state_get",
            "key": _literal("string", name),
            "type": state_types[name],
            "context": {"op": "map", "entries": []},
        }
    if kind in {"literal", "number"}:
        value = node.get("value")
        inferred = expected or (
            "boolean" if type(value) is bool else
            "double" if isinstance(value, float) or
            (kind == "number" and "." in str(value)) else
            "int" if kind == "number" or type(value) is int else
            "string" if isinstance(value, str) else None
        )
        return _literal(inferred, value)
    if kind == "arithmetic":
        if expected not in {"int", "long", "double"}:
            raise ValueError("ACTION_HOST_ARITHMETIC_TYPE_UNBOUND")
        op = node.get("op")
        if op not in {"+", "-", "*", "/", "%"}:
            raise ValueError("ACTION_HOST_ARITHMETIC_OPERATOR")
        return {
            "op": "binary", "operator": op,
            "left": _expression(node.get("left"), expected, inputs, state_types),
            "right": _expression(node.get("right"), expected, inputs, state_types),
        }
    if kind == "compare":
        if expected not in {None, "boolean"}:
            raise ValueError("ACTION_HOST_COMPARE_TYPE")
        return _comparison(node, inputs, state_types)
    if kind == "not":
        return {
            "op": "unary", "operator": "!",
            "value": _expression(
                node.get("term") or node.get("operand"),
                "boolean", inputs, state_types,
            ),
        }
    if kind in {"and", "or"}:
        return _boolean_terms(node, inputs, state_types)
    raise ValueError("ACTION_HOST_EXPRESSION_UNSUPPORTED: " + str(kind))


def _mutation_writes(ref, mutations, inputs, state_types):
    if not isinstance(mutations, list) or not mutations:
        raise ValueError("ACTION_HOST_MUTATION_REQUIRED: " + ref)
    writes = []
    for mutation in mutations:
        if not isinstance(mutation, Mapping):
            raise ValueError("ACTION_HOST_MUTATION_INVALID: " + ref)
        name = str(mutation.get("target") or "")
        if name not in state_types:
            raise ValueError("ACTION_HOST_STATE_UNBOUND: " + name)
        typ = state_types[name]
        op = mutation.get("operator", "=")
        right = _expression(mutation.get("value"), typ, inputs, state_types)
        if op != "=":
            if op not in {"+=", "-=", "*=", "/="} or typ not in {
                "int", "long", "double",
            }:
                raise ValueError("ACTION_HOST_MUTATION_OPERATOR")
            right = {
                "op": "binary", "operator": op[0],
                "left": _expression(
                    {"kind": "state_ref", "name": name}, typ, inputs, state_types,
                ),
                "right": right,
            }
        writes.append({
            "op": "state_set",
            "key": _literal("string", name),
            "value": right,
            "context": {"op": "map", "entries": []},
        })
    return writes


def _host_lower_ui_action_body(record, metadata, owned_refs, sections, state_types):
    """Compile a validated UI transaction without inferring execution semantics."""
    selected = _selected_mutations(record, owned_refs, sections)
    inputs = {row["name"]: row["type"] for row in metadata["inputs"]}
    statements = []
    for ref, row in selected:
        writes = _mutation_writes(ref, row.get("mutation"), inputs, state_types)
        guard = row.get("guard")
        if guard is None:
            statements.extend(writes)
        else:
            statements.append({
                "op": "if",
                "condition": _expression(guard, "boolean", inputs, state_types),
                "then": writes,
                "else": [{"op": "return", "value": _literal("int", 0)}],
            })
    statements.append({"op": "return", "value": _literal("int", 1)})
    return statements
