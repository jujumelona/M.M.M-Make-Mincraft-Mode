"""Deterministic source-owned lowering for typed UI transactions.

Only canonical authored state IR is eligible. Never infer purchases or
capture another trigger's effects from presentation metadata.
"""
from __future__ import annotations

from collections.abc import Mapping
import json

def _host_lower_ui_action_body(
    record, metadata, owned_refs, sections, state_types,
):
    """Lower an already authored, unambiguous state mutation without a model.

    This is a recovery path for a small model that loops while writing a
    recursive AST. Only source-owned canonical state mutations are eligible.
    Unsupported, unassigned or ambiguous effects must fail closed, rather
    than inventing a gameplay success.
    """
    from .authored_structured_design import active_concern_records
    from .typed_plan_authoring import (
        _decode_double_literal, _decode_int_literal, _decode_long_literal,
    )

    state = active_concern_records(sections, "state_model")
    interactions = active_concern_records(sections, "resources_and_ui").get(
        "interactions", ()
    )
    inputs = {row["name"]: row["type"] for row in metadata["inputs"]}
    available = []
    for concern in ("updates", "transitions"):
        for index, row in enumerate(state.get(concern, ())):
            if isinstance(row, Mapping):
                available.append((f"state_model.{concern}[{index}]", row))

    action_trigger = str(record.get("trigger") or "").strip().casefold()
    selected = [(ref, row) for ref, row in available if ref in owned_refs]
    if any(str(row.get("trigger") or "").strip().casefold() != action_trigger
           for _ref, row in selected):
        raise ValueError(
            "ACTION_HOST_OWNER_TRIGGER_MISMATCH: a runtime owner references "
            "a state rule belonging to a different UI trigger"
        )
    if not selected and not owned_refs:
        # Legacy single-action designs have no explicit runtime owner. Only
        # admit an exactly matching, unique state mutation in that case.
        action_rows = [
            row for row in interactions
            if isinstance(row, Mapping)
            and isinstance(row.get("server_action"), Mapping)
        ]
        if len(action_rows) == 1 and len(available) == 1:
            candidate = available[0]
            trigger = str(candidate[1].get("trigger") or "").strip().casefold()
            requested = str(record.get("trigger") or "").strip().casefold()
            if trigger and requested and trigger == requested:
                selected = [candidate]
    if not selected:
        raise ValueError(
            "ACTION_HOST_LOWERING_UNAVAILABLE: no uniquely assigned canonical "
            "state mutation for this UI action"
        )
    # Charging or applying the same persisted transaction twice is never
    # acceptable. Detect duplicate source rows before lowering.
    signatures = [
        json.dumps(row.get("mutation"), sort_keys=True, ensure_ascii=False)
        for _ref, row in selected
    ]
    if len(signatures) != len(set(signatures)):
        raise ValueError(
            "ACTION_HOST_DUPLICATE_MUTATION: identical transaction records "
            "were assigned to the same UI action"
        )

    def literal(typ, value):
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

    def expression(node, expected=None):
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
                "key": literal("string", name),
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
            return literal(inferred, value)
        if kind == "arithmetic":
            if expected not in {"int", "long", "double"}:
                raise ValueError("ACTION_HOST_ARITHMETIC_TYPE_UNBOUND")
            op = node.get("op")
            if op not in {"+", "-", "*", "/", "%"}:
                raise ValueError("ACTION_HOST_ARITHMETIC_OPERATOR")
            return {
                "op": "binary", "operator": op,
                "left": expression(node.get("left"), expected),
                "right": expression(node.get("right"), expected),
            }
        if kind == "compare":
            if expected not in {None, "boolean"}:
                raise ValueError("ACTION_HOST_COMPARE_TYPE")
            op = node.get("op")
            if op not in {"==", "!=", "<", "<=", ">", ">="}:
                raise ValueError("ACTION_HOST_COMPARE_OPERATOR")
            # State references and player inputs carry their declared types.
            left = node.get("left")
            right = node.get("right")
            hint = None
            for operand in (left, right):
                if isinstance(operand, Mapping):
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
                "left": expression(left, hint),
                "right": expression(right, hint),
            }
        if kind == "not":
            return {
                "op": "unary", "operator": "!",
                "value": expression(node.get("term") or node.get("operand"), "boolean"),
            }
        if kind in {"and", "or"}:
            terms = node.get("terms")
            if not isinstance(terms, list) or len(terms) < 2:
                raise ValueError("ACTION_HOST_BOOLEAN_TERMS")
            acc = expression(terms[0], "boolean")
            for term in terms[1:]:
                acc = {
                    "op": "binary",
                    "operator": "&&" if kind == "and" else "||",
                    "left": acc, "right": expression(term, "boolean"),
                }
            return acc
        raise ValueError("ACTION_HOST_EXPRESSION_UNSUPPORTED: " + str(kind))

    all_statements = []
    for ref, row in selected:
        mutations = row.get("mutation")
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
            right = expression(mutation.get("value"), typ)
            if op != "=":
                if op not in {"+=", "-=", "*=", "/="} or typ not in {
                    "int", "long", "double"
                }:
                    raise ValueError("ACTION_HOST_MUTATION_OPERATOR")
                right = {
                    "op": "binary", "operator": op[0],
                    "left": expression({"kind": "state_ref", "name": name}, typ),
                    "right": right,
                }
            writes.append({
                "op": "state_set",
                "key": literal("string", name),
                "value": right,
                "context": {"op": "map", "entries": []},
            })
        guard = row.get("guard")
        if guard is not None:
            all_statements.append({
                "op": "if",
                "condition": expression(guard, "boolean"),
                "then": writes,
                "else": [{"op": "return", "value": literal("int", 0)}],
            })
        else:
            all_statements.extend(writes)
    all_statements.append({"op": "return", "value": literal("int", 1)})
    return all_statements
