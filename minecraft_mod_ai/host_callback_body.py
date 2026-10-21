"""Compile small semantic callback choices into host-owned Java statements.

Expressions are values, never source declarations or statement fragments. The
host owns local names, semicolons, block delimiters, predicates' return syntax
and completion. Effects have no return operation. Final javac admission still
checks actual receiver/argument types and does not certify gameplay behavior.
"""
from __future__ import annotations

import json

from .construction_source_repair import validate_generated_java_source
from .fixed_template_generation import _validate_native_arguments
from .java_operation_ir import identifier
from .java_region_parser import _parse, _text
from .structured_output import StructuredOutputValidationError


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_EXPRESSION = {"type": "string", "minLength": 1, "maxLength": 2048}
_EXPRESSION_SCHEMA = _object({"expression": _EXPRESSION})
_SCHEMAS = {
    "call": _object({"receiver": _EXPRESSION,
                     "method": {"type": "string", "minLength": 1,
                                "maxLength": 128, "pattern": r"^[A-Za-z_$][A-Za-z0-9_$]*$"},
                     "arguments": {"type": "array", "maxItems": 16, "items": _EXPRESSION}}),
    "assign": _object({"target": _EXPRESSION,
                       "operator": {"type": "string", "enum": ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="]},
                       "value": _EXPRESSION}),
    "declare": _EXPRESSION_SCHEMA,
    "if": _EXPRESSION_SCHEMA,
    "while": _EXPRESSION_SCHEMA,
    "foreach": _EXPRESSION_SCHEMA,
}
_FORBIDDEN = frozenset({"lambda_expression", "class_body", "class_declaration",
                        "method_declaration", "assignment_expression", "update_expression"})


def parse_expression(value: str, *, predicate=False) -> str:
    """Admit exactly one Java value and reject statement/declaration escapes."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("CALLBACK_EXPRESSION_REQUIRED")
    raw, root = _parse("class MmmExpression { Object value() { return " + value + "; } }")
    owners = [n for n in root.named_children if n.type == "class_declaration"]
    if len(owners) != 1:
        raise ValueError("CALLBACK_EXPRESSION_SCOPE")
    body = owners[0].child_by_field_name("body")
    methods = [n for n in body.named_children if n.type == "method_declaration"]
    if len(methods) != 1:
        raise ValueError("CALLBACK_EXPRESSION_SCOPE")
    statements = [n for n in methods[0].child_by_field_name("body").named_children
                  if n.type not in {"line_comment", "block_comment"}]
    if len(statements) != 1 or statements[0].type != "return_statement":
        raise ValueError("CALLBACK_EXPRESSION_NOT_A_VALUE")
    values = [n for n in statements[0].named_children
              if n.type not in {"line_comment", "block_comment"}]
    if len(values) != 1 or _text(raw, values[0]).strip() != value.strip():
        raise ValueError("CALLBACK_EXPRESSION_NOT_A_VALUE")
    pending = [values[0]]
    while pending:
        node = pending.pop()
        if node.type in _FORBIDDEN:
            raise ValueError("CALLBACK_EXPRESSION_SIDE_EFFECT_OR_DECLARATION")
        if predicate and node.type == "method_invocation":
            name = _text(raw, node.child_by_field_name("name"))
            if name.startswith("writeState_") or name == "setState":
                raise ValueError("CALLBACK_PREDICATE_STATE_WRITE")
        pending.extend(node.named_children)
    return value.strip()


def _validate_state(source, frame):
    from .host_typed_state_access import (
        validate_behavior_state_provenance,
        validate_host_state_access,
    )
    contracts = frame.get("host_state_access", ())
    validate_host_state_access(source, contracts)
    validate_behavior_state_provenance(source, contracts, frame.get("host_behavior_contract"))


def _ask(router, messages, *, name, schema, context, lower):
    rejected, error = None, ""
    for attempt in range(2):
        payload = {**context, "last_error": error, "rejected_payload": rejected}
        response = None
        try:
            response = router.generate_structured_data(
                "coder", [*messages, {"role": "user", "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"))}],
                data_name=f"{name}_try_{attempt}", parameters=schema,
            )
            checked = _validate_native_arguments(response, schema)
            return lower(checked)
        except (ValueError, StructuredOutputValidationError) as exc:
            rejected, error = response if response is not None else getattr(exc, "output", None), str(exc)
            if attempt:
                failure = ValueError("CALLBACK_OPERATION_INVALID: " + error)
                failure.failed_response = json.dumps(response, ensure_ascii=False)
                raise failure from exc
    raise AssertionError("unreachable")


def author_callback_body(router, messages, *, data_name, frame, predicate=False) -> str:
    """Author expressions and operation parameters, never Java statements."""
    from .construction_java_context import scoped_java_member_messages
    scoped = list(scoped_java_member_messages(messages, frame))
    # Remove old source/statement transport instructions, retaining their
    # authoritative user payloads and the single coherent compiler protocol.
    scoped = [m for m in scoped if m.get("role") != "system"]
    instruction = (
        "The host compiles your semantic choices into Java. Supply only the requested "
        "JSON values, never statements, semicolons, return keywords, braces, method "
        "declarations or imports. Use actual callback parameters, inspected APIs and "
        "typed canonical state accessors. Preserve the assigned trigger and effects. "
        "Each expression is one value; lambdas and embedded assignments are forbidden. "
        "For call choose receiver, method name and argument values; the host writes "
        "the invocation. For assign choose an existing field/array target, operator "
        "and value. Declare creates a host-named local from your initializer. "
        "if/while/foreach open host-owned blocks; end closes one block; else opens "
        "the alternative of an if. Complete only after ALL assigned effects are "
        "implemented. Callback results and returns are owned by the host and cannot "
        "be supplied as effects. Do not replace missing state with an unrelated key."
    )
    scoped.insert(0, {"role": "system", "content": instruction})
    if predicate:
        def lower_predicate(response):
            expression = parse_expression(response["expression"], predicate=True)
            source = frame["prefix"] + "return " + expression + ";" + frame["suffix"]
            _validate_state(source, frame)
            return validate_generated_java_source(source, frame, frame["prefix"], frame["suffix"])
        return _ask(router, scoped, name=data_name + "_predicate", schema=_EXPRESSION_SCHEMA,
                    context={"instruction": "One pure boolean expression for the original trigger and preconditions."},
                    lower=lower_predicate)

    statements, blocks, locals_ = [], [], []
    effects = 0
    for index in range(64):
        choices = ["call", "assign", "declare", "if", "while", "foreach"]
        if blocks:
            choices.append("end")
            if blocks[-1] == "if":
                choices.append("else")
        if any(b in {"while", "foreach"} for b in blocks):
            choices.extend(["break", "continue"])
        if effects and not blocks:
            choices.append("complete")
        context = {"accepted_host_statements": statements, "host_locals": locals_,
                   "open_blocks": blocks, "next_operation": index}
        action = _ask(router, scoped, name=f"{data_name}_operation_{index}",
                      schema=_object({"action": {"type": "string", "enum": choices}}),
                      context=context, lower=lambda row: row["action"])
        if action == "complete":
            source = frame["prefix"] + "\n".join(statements) + frame["suffix"]
            return validate_generated_java_source(source, frame, frame["prefix"], frame["suffix"])
        new_local = f"mmmValue{len(locals_)}"

        def lower_operation(row, action=action, new_local=new_local):
            if action == "call":
                receiver = parse_expression(row["receiver"])
                method = identifier(row["method"])
                arguments = ", ".join(parse_expression(v) for v in row["arguments"])
                code = f"{receiver}.{method}({arguments});"
            elif action == "assign":
                target = parse_expression(row["target"])
                value = parse_expression(row["value"])
                code = f"{target} {row['operator']} {value};"
            else:
                expression = parse_expression(row["expression"], predicate=action in {"if", "while"})
                if action == "declare":
                    code = f"var {new_local} = {expression};"
                elif action == "foreach":
                    code = f"for (var {new_local} : {expression}) {{"
                else:
                    code = f"{action} ({expression}) {{"
            # Check a balanced trial, including all already accepted blocks.
            trial_blocks = len(blocks) + int(action in {"if", "while", "foreach"})
            source = (frame["prefix"] + "\n".join([*statements, code])
                      + "\n}" * trial_blocks + frame["suffix"])
            validate_generated_java_source(source, frame, frame["prefix"], frame["suffix"])
            _validate_state(source, frame)
            return code

        if action in _SCHEMAS:
            code = _ask(router, scoped, name=f"{data_name}_operation_{index}_{action}",
                        schema=_SCHEMAS[action], context={**context, "action": action,
                            "host_local_name": new_local if action in {"declare", "foreach"} else None},
                        lower=lower_operation)
            statements.append(code)
            effects += int(action in {"call", "assign"})
            if action in {"declare", "foreach"}:
                locals_.append(new_local)
            if action in {"if", "while", "foreach"}:
                blocks.append(action)
        elif action == "end":
            blocks.pop()
            statements.append("}")
        elif action == "else":
            blocks[-1] = "else"
            statements.append("} else {")
        else:
            statements.append(action + ";")
    raise ValueError("CALLBACK_OPERATION_LIMIT: assigned effect requires decomposition")


__all__ = ["author_callback_body", "parse_expression"]
