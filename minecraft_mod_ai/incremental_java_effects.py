"""Necessary structural effects, distinct from compiler/gameplay acceptance."""
from __future__ import annotations

from functools import lru_cache
import json

from .incremental_java_history import child_blocks
from .incremental_java_render import _expression_node


@lru_cache(maxsize=2048)
def _expression_has_call(expression):
    """Cache parser work, not an implementation budget or gameplay proof."""
    try:
        _, _, root = _expression_node(expression, allow_update=True)
    except ValueError:
        return False
    pending = [root]
    while pending:
        current = pending.pop()
        if current.type in {"method_invocation", "object_creation_expression"}:
            return True
        pending.extend(current.named_children)
    return False


def expression_effect_key(node):
    operands = {key: node[key] for key in ("value", "condition", "iterable", "selector", "update", "monitor")
                if isinstance(node.get(key), str) and _expression_has_call(node[key])}
    return json.dumps({"op": node.get("op"), "operands": operands}, sort_keys=True) if operands else None


def _executed_nodes(nodes):
    for node in nodes:
        yield node
        # Defining a closure does not invoke its deferred body.
        if node.get("op") == "lambda":
            continue
        for _, children in child_blocks(node):
            yield from _executed_nodes(children)


def _remember_local(node, names):
    if node.get("op") in {"local", "lambda", "for", "foreach"}:
        names.add(node.get("name"))
    elif node.get("op") == "call" and node.get("result_type"):
        names.add(node.get("name"))


def _node_has_potential_effect(node, local_names):
    if expression_effect_key(node) is not None:
        return True
    operation = node.get("op")
    if operation == "assign":
        target = str(node.get("target") or "").strip()
        return bool(target) and target not in local_names
    if operation in {"call", "register", "constructor_call", "throw"}:
        return True
    return operation == "return" and bool(node.get("value"))


def _member_has_potential_effect(member):
    local_names = {row.get("name") for row in member.get("metadata", {}).get("parameters", ())}
    for node in _executed_nodes(member.get("body") or ()):
        _remember_local(node, local_names)
        if _node_has_potential_effect(node, local_names):
            return True
    return False
