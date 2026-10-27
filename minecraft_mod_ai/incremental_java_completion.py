"""Continue incomplete generated blocks in their original lexical environment."""
from __future__ import annotations

from .incremental_java_model import initial_scope
from .incremental_java_scope import scope_at, nested_scopes


def _append_return(builder, body, return_type):
    builder.block = body
    body.append({"id": builder.allocate("n"), "op": "return", "value": builder.expression("value", return_type)})


def _complete_lambda(builder, member, node):
    outer = nested_scopes(node, builder.scope)[0][1]
    temporary = {**member, "body": node["body"], "metadata": {
        **member["metadata"], "return_type": node["return_type"], "parameters": outer["locals"]}}
    builder.scope = scope_at(temporary)
    builder.path = [*builder.path, node["id"], "body"]
    _append_return(builder, node["body"], node["return_type"])


def _complete_member(builder, member, diagnostic, missing_return):
    if member["kind"] in {"class", "record", "enum"} and "does not override abstract method" in diagnostic:
        builder.author_block(member, initial_scope(member), [], body=member["body"])
    elif "CONSTRUCTION_FILE_BEHAVIOR_" in diagnostic and member.get("registrar"):
        member["body"][:] = [item for item in member["body"] if item["op"] != "return" or item.get("value")]
        builder.author_block(member, scope_at(member), [], body=member["body"])
    elif missing_return and member["metadata"].get("return_type", "void") != "void":
        builder.scope = scope_at(member)
        _append_return(builder, member["body"], member["metadata"]["return_type"])
    elif member["kind"] == "constructor" and not any(n["op"] == "constructor_call" for n in member["body"]):
        from .incremental_java_operations import author_operation
        member["body"].insert(0, author_operation(builder, "constructor_call"))
    else:
        return False
    return True


def complete_program_block(builder, member, node, diagnostic):
    missing_return = any(text in diagnostic for text in ("missing return value", "missing return statement"))
    if node is None:
        return _complete_member(builder, member, diagnostic, missing_return)
    if node["op"] != "lambda" or not missing_return or node["return_type"] == "void":
        return False
    _complete_lambda(builder, member, node)
    return True
