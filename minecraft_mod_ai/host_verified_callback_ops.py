"""Compile model-selected callback operations from actual indexed Java signatures.

This is deliberately not a source-string grammar. A model picks an opaque ID
for a verified receiver/method/argument tuple assembled by the host. No method,
receiver, or argument expression is model-authored along this path.
"""
from __future__ import annotations

import json
import re

from .java_region_parser import _parse, _text


def _callback_parameters(frame):
    prefix = frame.get("prefix", "")
    raw, root = _parse("class MmmVerifiedCallback {\n" + prefix + "\n}\n}")
    body = next(node for node in root.named_children if node.type == "class_declaration").child_by_field_name("body")
    methods = [node for node in body.named_children if node.type == "method_declaration"]
    if len(methods) != 1:
        return []
    result = []
    for param in methods[0].child_by_field_name("parameters").named_children:
        if param.type != "formal_parameter":
            continue
        kind, name = param.child_by_field_name("type"), param.child_by_field_name("name")
        if kind is None or name is None:
            continue
        result.append((_text(raw, kind), _text(raw, name)))
    return result


def verified_callback_choices(messages, frame, *, predicate=False):
    """Return bounded (ID, Java expression, ABI) choices with exact typed operands."""
    task = None
    for message in messages:
        if message.get("role") != "user":
            continue
        try:
            row = json.loads(message.get("content") or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(row, dict) and isinstance(row.get("verified_event_callback"), dict):
            task = row
    if task is None or not isinstance(task.get("callback_input_api"), list):
        return []
    params = _callback_parameters(frame)
    declared = task["verified_event_callback"].get("parameters", [])
    if [kind for kind, _ in params] != declared:
        # A method's immutable actual inputs must match the inspected callback.
        return []
    inputs = {}
    for kind, name in params:
        inputs.setdefault(kind, []).append(name)
    choices, seen = [], set()
    for record in task["callback_input_api"]:
        if not isinstance(record, dict):
            continue
        owner = record.get("verified_class")
        receivers = inputs.get(owner, ())
        if not receivers:
            continue
        for method in record.get("members", ()):
            if not isinstance(method, dict) or method.get("kind") != "METHOD" or method.get("static"):
                continue
            name = method.get("name")
            if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name) is None:
                continue
            # Listener installation belongs to onInitialize. An already-fired
            # event handler calling Event.register() is neither the requested
            # gameplay mutation nor a valid execution owner, even if an ABI
            # method with that name exists in the inspected classpath.
            if (name == "register" and
                    owner.startswith("net.fabricmc.fabric.api.")):
                continue
            arguments = method.get("parameters")
            if not isinstance(arguments, list) or len(arguments) > 3:
                continue
            if predicate and (method.get("return_type") != "boolean" or arguments):
                continue
            if not predicate and name.startswith(
                ("get", "is", "has", "can", "should", "read", "peek")
            ):
                continue
            if any(not inputs.get(kind) for kind in arguments):
                continue
            bound_args = [inputs[kind][0] for kind in arguments]
            for receiver in receivers:
                expression = receiver + "." + name + "(" + ", ".join(bound_args) + ")"
                if expression in seen:
                    continue
                seen.add(expression)
                choices.append({
                    "id": "v" + str(len(choices)),
                    "expression": expression,
                    "signature": owner + "#" + name + "(" + ", ".join(arguments) + ")",
                    "return_type": method.get("return_type"),
                })
    return choices[:32]
