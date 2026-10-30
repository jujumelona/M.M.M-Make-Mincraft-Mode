"""Source-derived proof gates for host-authored incremental Fabric features.

An AST call is not a gameplay test. These checks establish that each behavior
is connected to an inspected occurrence and that each required native-state
write has an explicitly selected inspected call in a reachable listener.
"""
from __future__ import annotations

from collections.abc import Mapping

from .incremental_java_author import _reachable_effect_nodes
from .incremental_java_calls import _owner_matches
from .java_operation_ir import parameter_types

_READ_ONLY = ("get", "is", "has", "can", "should", "read", "peek")
_OBSERVERS = frozenset({
    "equals", "hashCode", "toString", "compareTo", "contains",
    "containsKey", "containsValue", "size", "length", "iterator",
    "stream", "spliterator", "clone", "copy", "matches",
})


def native_write_targets(behavior, bindings):
    """Keep custom model state separate from Minecraft-owned runtime state."""
    from .construction_behavior_state import _runtime_owned_state_reference

    targets = []
    for name in behavior.get("runtime_state_writes") or ():
        if not isinstance(name, str) or not name:
            continue
        binding = bindings.get(name) if isinstance(bindings, Mapping) else None
        if isinstance(binding, Mapping):
            status = binding.get("semantic_status")
            if status == "host_native_runtime_owner" or binding.get("binding") == "world_state":
                targets.append(name)
            elif status not in {"verified_scalar_state", "host_owned_state_model"} and _runtime_owned_state_reference(name):
                targets.append(name)
        elif _runtime_owned_state_reference(name):
            targets.append(name)
    return tuple(dict.fromkeys(targets))


def _registration_nodes(program, registrar):
    """Follow completed generated methods, without accepting method-name guesses."""
    by_id = {row["id"]: row for row in program["members"] if row.get("kind") == "method"}
    seen, pending = set(), [registrar]
    while pending:
        current = pending.pop()
        if current["id"] in seen:
            continue
        seen.add(current["id"])
        for node in _reachable_effect_nodes(current.get("body") or ()):
            if node.get("op") == "register":
                yield node
            if node.get("op") == "call":
                target_id = node.get("generated_member_id")
                if target_id in by_id:
                    pending.append(by_id[target_id])


def _verified_event(event, allowed, behavior):
    key = str(event.get("event_owner") or "") + "#" + str(event.get("event_field") or "")
    checked = allowed.get(key)
    if checked is None:
        return False
    explicit = str(behavior.get("event_binding") or "")
    if explicit and explicit != key:
        return False
    if checked.get("provenance") in {"verified_source_site_mixin", "host_typed_ui_action"}:
        if checked.get("behavior_id") != behavior.get("behavior_id"):
            return False
    return True


def _matching_registration(program, registrar, behavior, events):
    allowed = {row["event_id"]: row for row in events if isinstance(row, dict) and row.get("event_id")}
    handlers = {member.get("name"): member for member in program["members"] if member.get("event")}
    for node in _registration_nodes(program, registrar):
        if not _verified_event(node, allowed, behavior):
            continue
        handler = handlers.get(node.get("handler"))
        if handler is None or handler.get("behavior", {}).get("behavior_id") != behavior["behavior_id"]:
            continue
        event = handler.get("event") or {}
        if (event.get("event_owner") == node.get("event_owner")
                and event.get("event_field") == node.get("event_field")):
            yield handler


def _inspected_call(node, symbols):
    """Verify callable owner+overload against the checked target API, not prose."""
    if node.get("generated_member_id") or node.get("op") != "call":
        return False
    contract = node.get("call_contract") or {}
    owner, method = contract.get("owner"), contract.get("name")
    if not owner or method != node.get("method"):
        return False
    if method.startswith(_READ_ONLY) or method in _OBSERVERS:
        return False
    params = tuple(parameter_types(contract))
    return any(
        isinstance(row, Mapping) and row.get("kind") == "METHOD"
        and _owner_matches(row.get("owner"), owner) and row.get("name") == method
        and tuple(parameter_types(row)) == params
        and row.get("provenance") != "construction_plan"
        for row in symbols.values()
    )


def _native_effects(program, handler, symbols):
    """Proven call placement, not a substitute for in-game mutation tests."""
    generated = {row["id"]: row for row in program["members"] if row.get("kind") == "method"}
    seen, pending, found = set(), [handler], set()
    while pending:
        current = pending.pop()
        if current["id"] in seen:
            continue
        seen.add(current["id"])
        for node in _reachable_effect_nodes(current.get("body") or ()):
            if node.get("op") != "call":
                continue
            identity = node.get("generated_member_id")
            if identity in generated:
                pending.append(generated[identity])
            elif node.get("native_state_effect") and _inspected_call(node, symbols):
                found.add(node["native_state_effect"])
    return found


def verify_incremental_feature_execution(program, behaviors, callbacks, bindings, symbols):
    """Fail closed for missing occurrence or ungrounded required native effect."""
    registrars = {
        str((row.get("behavior") or {}).get("behavior_id")): row
        for row in program["members"] if row.get("registrar")
    }
    for behavior in behaviors:
        identity = str(behavior.get("behavior_id") or "")
        registrar = registrars.get(identity)
        if not identity or registrar is None:
            raise ValueError("INCREMENTAL_JAVA_BEHAVIOR_REGISTRAR_MISSING: " + identity)
        handlers = list(_matching_registration(program, registrar, behavior, callbacks))
        if not handlers:
            raise ValueError("INCREMENTAL_JAVA_INSPECTED_OCCURRENCE_MISSING: " + identity)
        required = set(native_write_targets(behavior, bindings))
        if not required:
            continue
        actual = set()
        for handler in handlers:
            actual.update(_native_effects(program, handler, symbols))
        missing = required - actual
        if missing:
            raise ValueError("INCREMENTAL_JAVA_NATIVE_MUTATION_UNPROVEN: "
                             + identity + " states=" + repr(sorted(missing)))
