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
    declared = task["verified_event_callback"].get("parameters", [])
    # The host fixed and validated the declaration metadata before any body
    # model call. Prefer that typed ownership record over reparsing one
    # synthetic half-open method prefix. The prefix parser can return no
    # method for a valid incomplete Java frame, silently removing all
    # verified calls and forcing the 9B coder to invent a method invocation.
    metadata = task.get("declaration_metadata") or {}
    declared_parameters = metadata.get("parameters") or ()
    authored_params = [
        (row.get("type"), row.get("name"))
        for row in declared_parameters if isinstance(row, dict)
    ]
    if (len(authored_params) == len(declared)
            and all(isinstance(typ, str) and isinstance(name, str)
                    and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name)
                    for typ, name in authored_params)
            and [typ for typ, _ in authored_params] == declared):
        parsed = _callback_parameters(frame)
        if parsed and parsed != authored_params:
            return []  # A disagreement with the actual frame is not authority.
        params = authored_params
    else:
        params = _callback_parameters(frame)
    if [kind for kind, _ in params] != declared:
        # The host may not substitute a different receiver just because it
        # exposes a conveniently named method. Every argument is exact ABI.
        return []
    inputs = {}
    for kind, name in params:
        inputs.setdefault(kind, []).append(name)
    choices, seen = [], set()
    api = {
        row.get("verified_class"): row
        for row in task["callback_input_api"]
        if isinstance(row, dict) and isinstance(row.get("verified_class"), str)
    }
    # Source owners are discovered by walking *inspected* signatures only.
    # Do not ask the 9B model to write `player.getWorld()` as an arbitrary
    # receiver expression. A callback input may provide a verified one-hop
    # receiver into a Level/inventory/project controller.
    for record in task["callback_input_api"]:
        if not isinstance(record, dict):
            continue
        owner = record.get("verified_class")
        receivers = list(inputs.get(owner, ()))
        if not receivers:
            bridge = record.get("verified_receiver_path")
            if not isinstance(bridge, dict) or bridge.get("destination_type") != owner:
                continue
            source_type, method_name = bridge.get("source_type"), bridge.get("source_method")
            origin = api.get(source_type, {})
            if not isinstance(method_name, str) or re.fullmatch(
                    r"[A-Za-z_$][A-Za-z0-9_$]*", method_name) is None:
                continue
            verified = any(
                row.get("kind") == "METHOD" and row.get("name") == method_name
                and row.get("return_type") == owner
                and not row.get("parameters") and not row.get("static")
                for row in origin.get("members", ())
                if isinstance(row, dict)
            )
            if not verified:
                continue
            receivers = [
                expression + "." + method_name + "()"
                for expression in inputs.get(source_type, ())
            ]
            if receivers:
                inputs[owner] = receivers
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


def _checked_receiver_getter(apis, start, method, owner):
    return any(
        isinstance(row, dict) and row.get("kind") == "METHOD"
        and row.get("name") == method and row.get("return_type") == owner
        and not row.get("parameters") and not row.get("static")
        and not int(row.get("access", 0)) & 8
        for row in apis.get(start, {}).get("members", ())
    )


def _valid_native_bridge(bridge, owner, declared, apis):
    if not isinstance(bridge, dict) or bridge.get("destination_type") != owner:
        return False
    start, method = bridge.get("source_type"), bridge.get("source_method")
    return (start in declared and isinstance(method, str)
            and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", method)
            and _checked_receiver_getter(apis, start, method, owner))


def _extend_native_receivers(records, declared, params, apis, inputs):
    """Every one-hop path starts from an immutable original event input."""
    for record in records:
        if not isinstance(record, dict):
            continue
        owner = record.get("verified_class")
        bridges = list(record.get("verified_receiver_paths") or ())
        if isinstance(record.get("verified_receiver_path"), dict):
            bridges.append(record["verified_receiver_path"])
        for bridge in bridges:
            if not _valid_native_bridge(bridge, owner, declared, apis):
                continue
            for origin in (name for kind, name in params if kind == bridge["source_type"]):
                receiver = origin + "." + bridge["source_method"] + "()"
                if receiver not in inputs.setdefault(owner, []):
                    inputs[owner].append(receiver)


def _runtime_argument_slots(arguments, inputs):
    slots = []
    for kind in arguments:
        values = inputs.get(kind, ())
        slot = {"type": kind, "expression": values[0] if len(values) == 1 else None}
        if len(values) > 1:
            slot["verified_values"] = tuple(values)
        slots.append(slot)
    return slots


def verified_runtime_mutation_candidates(messages, frame, *, api_task=None):
    """Expose grounded receiver+method ABIs with typed *value* holes.

    A Fabric callback rarely provides `float damage` or `int amount` as a
    direct parameter. The old bound-call-only menu silently lost methods like
    `setHealth(float)` even if indexed on the real callback's player receiver.
    This catalog lets a separate host slot author such scalar values while
    keeping method identity, receiver and argument count immutable.
    """
    task = api_task
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
    declared = task["verified_event_callback"].get("parameters", [])
    metadata = task.get("declaration_metadata") or {}
    typed = metadata.get("parameters") or ()
    authoritative = [
        (row.get("type"), row.get("name"))
        for row in typed if isinstance(row, dict)
    ]
    parsed = _callback_parameters(frame)
    if (len(authoritative) == len(declared)
            and [kind for kind, _ in authoritative] == declared
            and all(isinstance(name, str) and re.fullmatch(
                r"[A-Za-z_$][A-Za-z0-9_$]*", name) for _, name in authoritative)):
        if parsed and parsed != authoritative:
            return []
        params = authoritative
    else:
        params = parsed
    if [kind for kind, _ in params] != declared:
        return []

    inputs: dict[str, list[str]] = {}
    for kind, name in params:
        inputs.setdefault(kind, []).append(name)
    apis = {
        row.get("verified_class"): row
        for row in task["callback_input_api"]
        if isinstance(row, dict) and isinstance(row.get("verified_class"), str)
    }
    scalar_types = {
        "boolean", "byte", "short", "char", "int", "long", "float", "double",
        "java.lang.String", "String",
        "java.lang.Boolean", "java.lang.Integer", "java.lang.Long",
        "java.lang.Float", "java.lang.Double",
    }
    forbidden = {
        "equals", "hashCode", "toString", "compareTo", "contains",
        "containsKey", "containsValue", "size", "length", "iterator",
        "stream", "spliterator", "clone", "copy", "matches",
    }
    _extend_native_receivers(task["callback_input_api"], declared, params, apis, inputs)
    candidates, seen = [], set()
    for record in task["callback_input_api"]:
        if not isinstance(record, dict):
            continue
        owner = record.get("verified_class")
        if not isinstance(owner, str):
            continue
        receivers = list(inputs.get(owner, ()))
        if not receivers:
            continue
        for row in record.get("members", ()):
            if (not isinstance(row, dict) or row.get("kind") != "METHOD"
                    or row.get("static") or int(row.get("access", 0)) & 8
                    or row.get("owner") == "java.lang.Object"):
                continue
            name = row.get("name")
            if (not isinstance(name, str) or not re.fullmatch(
                    r"[A-Za-z_$][A-Za-z0-9_$]*", name)
                    or name.startswith(("get", "is", "has", "can", "should", "read", "peek"))
                    or name in forbidden):
                continue
            if name == "register" and owner.startswith("net.fabricmc.fabric.api."):
                continue
            arguments = row.get("parameters")
            if not isinstance(arguments, list):
                continue
            # Non-scalar parameters must already be real callback inputs.
            # Scalar value slots are bounded and type checked at compilation.
            if any(kind not in inputs and kind not in scalar_types for kind in arguments):
                continue
            signature = owner + "#" + name + "(" + ", ".join(arguments) + ")"
            for receiver in receivers:
                identity = (receiver, signature)
                if identity in seen:
                    continue
                seen.add(identity)
                candidates.append({
                    "id": "m" + str(len(candidates)),
                    "receiver": receiver, "name": name, "signature": signature,
                    "arguments": _runtime_argument_slots(arguments, inputs),
                    "return_type": row.get("return_type"),
                })
    return candidates

