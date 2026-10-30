"""Host-owned work phases: acquire runtime inputs, then implement their effects."""
from __future__ import annotations

from .incremental_java_effects import _member_has_potential_effect
from .incremental_java_history import walk_nodes
from .java_operation_ir import parameter_types


def _world_bound(builder, member):
    behavior = member.get("behavior") or {}
    names = [*(behavior.get("runtime_state_reads") or ()), *(behavior.get("runtime_state_writes") or ())]
    return any(builder.api.bindings.get(name, {}).get("binding") == "world_state" for name in names)


def _needs_observation(builder, member, body):
    # A missing callback is an unsatisfied producer obligation, not permission
    # to author a lifecycle method that never observes a gameplay occurrence.
    return (bool(member.get("registrar"))
            and not any(node.get("op") == "register" for node in walk_nodes(body)))


def work_view(builder):
    member = builder.member or {}
    if member.get("work_role") == "supporting_abi":
        return {"phase": "implement_supporting_abi", "purpose": member.get("purpose", ""),
                "next_step": "Implement this exact interface hook. A supporting no-op hook may "
                             "complete without inventing effects; required gameplay remains with its owning work."}
    if not _world_bound(builder, member):
        return {"phase": "implement_current_work", "purpose": member.get("purpose", "")}
    if _needs_observation(builder, member, member.get("body") or builder.block):
        return {"phase": "bind_observation", "next_step": (
            "Select a real observation event for the original trigger first. Its callback inputs "
            "are the operands for implementation. Host lifecycle registrars are not gameplay helpers.")}
    return {"phase": "implement_effects", "next_step": (
        "Use these actual callback inputs, the saved feature_work_order and verified state/API "
        "declarations to implement the required effects. Local copies and searches do not "
        "complete this method. Nested rejection returns remain available.")}


def work_options(builder, member, scope, path, body, options):
    if (path or member.get("kind") != "method" or member.get("work_role") == "supporting_abi"
            or not _world_bound(builder, member)):
        return options
    if _needs_observation(builder, member, body):
        # The indexed callback selector owns event search/pagination. Generic
        # Java lookup cannot install an occurrence hook and was selected 32
        # times before the production model tried the actual event selector.
        return [op for op in options if op == "register"]
    if _member_has_potential_effect({**member, "body": body}):
        return options
    blocked = {"complete"}
    if scope.get("return_type", "void") == "void":
        blocked.add("return")
    return [op for op in options if op not in blocked]


def method_unavailable(builder, member):
    return (bool(member.get("registrar"))
            or member.get("id", object()) == (builder.member or {}).get("id")
            or member.get("implementation_status") == "authoring")


def _host_owners(builder):
    name = builder.program["class_name"]
    return {name, builder.program["package"] + "." + name}


def host_lifecycle_api(builder, receiver, row):
    return row.get("kind") == "METHOD" and _initializer_call(
        builder, receiver.java_type, row.get("name"), row)


def _initializer_call(builder, receiver, name, contract):
    return ((receiver in _host_owners(builder) or contract.get("owner") == builder.program.get("interface"))
            and name == builder.program.get("lifecycle")
            and not parameter_types(contract))


def lifecycle_call_diagnostic(builder, node):
    if not node or node.get("op") != "call":
        return ""
    contract = node.get("call_contract") or {}
    receiver = node.get("receiver_type") or node.get("receiver")
    if receiver in {"", "this"}:
        receiver = (builder.member or {}).get("owner")
    if _initializer_call(builder, receiver, node.get("method"), contract):
        return "JAVA_HOST_LIFECYCLE_CALL: " + builder.program["lifecycle"]
    owner = contract.get("owner") or receiver
    for member in builder.program["members"]:
        if not member.get("registrar") or member.get("name") != node.get("method"):
            continue
        if owner not in {member["owner"], builder.program["package"] + "." + member["owner"]}:
            continue
        if parameter_types(contract) == parameter_types(member["metadata"]):
            return "JAVA_HOST_LIFECYCLE_CALL: " + member["id"]
    return ""


def reject_lifecycle_call(builder, node):
    diagnostic = lifecycle_call_diagnostic(builder, node)
    if diagnostic:
        raise RuntimeError(diagnostic)
