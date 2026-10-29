"""Edits to typed operands/declarations without replacing accepted method bodies."""
from __future__ import annotations

from .incremental_java_schema import TYPE, BOOL, obj, choice
from .incremental_java_navigation import record_navigation_transition
from .incremental_java_render import validate_type


def replace_expression(builder, node, slot):
    if slot.startswith("argument:"):
        index = int(slot.split(":")[1])
        node["arguments"][index] = builder.expression(slot)
    elif slot.startswith("case:"):
        index = int(slot.split(":")[1])
        node["cases"][index]["label"] = builder.expression(slot)
    else:
        expected = "boolean" if slot == "condition" else "java.lang.Object"
        node[slot] = builder.expression(slot, expected, allow_update=slot == "update")


def replace_node(builder, node, slot):
    if slot.startswith(("catch:", "parameter:")):
        kind, index, _ = slot.split(":")
        container = "catches" if kind == "catch" else "parameters"
        node[container][int(index)]["type"] = _type(builder, allow_union=kind == "catch")
    elif slot in {"type", "result_type", "return_type"}:
        node[slot] = _type(builder, allow_void=slot == "return_type")
    elif slot in {"method", "receiver", "event_field", "handler"}:
        replace_call_head(builder, node)
    else:
        replace_expression(builder, node, slot)


def replace_call_head(builder, node):
    """A changed receiver/member needs a fresh exact ABI, never a code string."""
    from .incremental_java_calls import _is_varargs, _source_parameters
    from .incremental_java_values import author_call_head
    if node["op"] == "register":
        revise_operation(builder, node)
        return
    before = node.get("call_contract") or {}
    author_call_head(builder, node)
    contract = node["call_contract"]
    previous, current = _source_parameters(before), _source_parameters(contract)
    if (not before or previous != current
            or _is_varargs(before, previous) != _is_varargs(contract, current)):
        builder.arguments(node)
    if contract.get("return_type") == "void":
        node["result_type"] = ""


def _type(builder, **type_options):
    value = builder.typed(obj({"type": TYPE}), "repair_type", "type", **type_options)["type"]
    return validate_type(value)


def declaration_slots(member):
    metadata = member["metadata"]
    slots = {}
    # Actual selected event ABI stays immutable. Its body may need a return or
    # different operands, never a fabricated replacement callback signature.
    if member.get("event"):
        return slots
    for field in ("type", "return_type", "superclass"):
        if metadata.get(field):
            slots[field] = metadata[field]
    for index, parameter in enumerate(metadata.get("parameters", [])):
        slots[f"parameter:{index}"] = parameter["type"]
    for index, component in enumerate(metadata.get("components", [])):
        slots[f"component:{index}"] = component["type"]
    for index, interface in enumerate(metadata.get("interfaces", [])):
        slots[f"interface:{index}"] = interface
    for flag in ("static", "final"):
        if flag in metadata:
            slots[flag] = metadata[flag]
    return slots


def replace_declaration(builder, member, slot):
    if slot == "initializer":
        member["initializer"] = builder.expression("initializer", member["metadata"]["type"])
        return
    slots = declaration_slots(member)
    if not slots:
        raise ValueError("JAVA_INCREMENTAL_SELECTED_ABI_IS_HOST_OWNED")
    selected = builder.ask(choice("declaration_slot", slots), "repair_declaration_slot",
                           {"declaration_type_slots": slots})["declaration_slot"]
    metadata = member["metadata"]
    if selected in {"static", "final"}:
        metadata[selected] = builder.ask(obj({selected: BOOL}), "repair_" + selected)[selected]
    elif selected.startswith("parameter:"):
        index = int(selected.split(":")[1])
        metadata["parameters"][index]["type"] = _type(builder, allow_varargs=index == len(metadata["parameters"]) - 1)
    elif selected.startswith("interface:"):
        metadata["interfaces"][int(selected.split(":")[1])] = _type(builder)
    elif selected.startswith("component:"):
        index = int(selected.split(":")[1])
        metadata["components"][index]["type"] = _type(builder, allow_varargs=index == len(metadata["components"]) - 1)
    else:
        metadata[selected] = _type(builder, allow_void=selected == "return_type")


def repair_related_declaration(builder, current):
    rows = [row for row in builder.program["members"] if declaration_slots(row)]
    offset, size = 0, 10
    transitions = {}
    while True:
        page = rows[offset:offset + size]
        options = [row["id"] for row in page]
        if offset + size < len(rows):
            options.append("@next")
        if offset:
            options.append("@previous")
        selected = builder.ask(choice("declaration_id", options), "repair_related_declaration", {
            "declaration_candidates": [{key: row.get(key) for key in ("id", "name", "kind", "owner", "metadata", "purpose")}
                                       for row in page], "declaration_offset": offset,
            "declaration_total": len(rows), "diagnostic_member": current["id"]})["declaration_id"]
        target = next((row for row in page if row["id"] == selected), None)
        if target:
            with builder.declaration_scope(target):
                replace_declaration(builder, target, "declaration")
            return
        record_navigation_transition(
            transitions, offset, selected, error="JAVA_REPAIR_DECLARATION_NAVIGATION_CYCLE",
        )
        offset += size if selected == "@next" else -size


def revise_operation(builder, node):
    if node["op"] in {"call", "register"}:
        before = node.copy()
        revised = _replacement_operation(builder, node["op"])
        revised["id"] = before["id"]
        if before.get("name") and revised.get("result_type"):
            revised["name"] = before["name"]
        node.clear()
        node.update(revised)
    else:
        replace_node(builder, node, builder.rejection["repair_slot"])


def _replacement_operation(builder, operation):
    from .incremental_java_operations import author_operation
    # A register operation can be cancelled or unavailable (no indexed event).
    # Re-entering it forever is not a correction and can spin without any model
    # request when the event catalogue is empty.
    for _ in range(2):
        revised = author_operation(builder, operation)
        if revised is not None:
            return revised
    raise ValueError("JAVA_INCREMENTAL_REPAIR_OPERATION_UNAVAILABLE: " + operation)


def create_helper(builder, node):
    from .incremental_java_declarations import author_declaration
    from .incremental_java_calls import _signature
    method = author_declaration(builder, "method")
    node["method"] = method["name"]
    node["receiver"] = "" if method["metadata"].get("static") else "this"
    node["call_contract"] = _signature({"kind": "METHOD", "owner": method["owner"],
                                        "name": method["name"], **method["metadata"]})
    builder.arguments(node)
