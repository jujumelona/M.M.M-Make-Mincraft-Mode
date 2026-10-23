"""Bind missing player identities from the host's event and call argument ABI."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


def _nodes(value):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _nodes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _nodes(child)


def _guarded_nodes(value, event=None):
    if isinstance(value, Mapping):
        yield value, event
        condition = value.get("condition", {}) if value.get("op") == "if" else {}
        selected = None
        if (
            condition.get("op") == "binary"
            and condition.get("operator") == "=="
            and condition.get("left") == {"op": "ref", "name": "event"}
            and condition.get("right", {}).get("type") == "string"
            and condition["right"].get("op") == "literal"
        ):
            selected = condition["right"]["value"]
        for key, child in value.items():
            yield from _guarded_nodes(
                child, selected if key == "then" and selected is not None else event
            )
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _guarded_nodes(child, event)


def bind_player_state_context(plan, state_section, capabilities):
    variables = (state_section or {}).get("specification", {}).get("variables", ())
    scoped = {row["name"] for row in variables if row.get("owner") == "player"}
    if not scoped:
        return plan, capabilities
    result, contracts = deepcopy(plan), dict(capabilities or {})
    functions = {row["id"]: row for row in result.get("functions", ())}
    actors = {}
    nonplayer_roots = set()
    positions = {
        "command": 0,
        "player_join": 0,
        "player_disconnect": 0,
        "player_respawn": 1,
    }
    for binding in result.get("event_bindings", ()):
        position = positions.get(binding["event"])
        if position is not None:
            fn = functions[binding["function"]]
            actors[fn["id"]] = fn["parameters"][position]["name"]
        else:
            nonplayer_roots.add(binding["function"])
    dispatchers = set()
    # This is the complete host-owned semantic dispatcher ABI, including saved plans.
    for fn in functions.values():
        if fn["id"] in {"logic_dispatch", "ui_action_dispatch"} and fn[
            "parameters"
        ] == [
            {"name": "event", "type": "string"},
            {"name": "primary", "type": "object"},
            {"name": "secondary", "type": "object"},
            {"name": "flag", "type": "boolean"},
        ]:
            actors[fn["id"]] = "primary"
            dispatchers.add(fn["id"])
    calls = [
        (fn["id"], node, event)
        for fn in [
            *functions.values(),
            {"id": "<initialize>", "body": result.get("initialize", [])},
        ]
        for node, event in _guarded_nodes(fn["body"])
        if node.get("op") == "call"
    ]
    safe_events = {
        key: {"player_join", "player_disconnect", "player_respawn", "command"}
        for key in dispatchers
    }
    unsafe_events = {key: set() for key in dispatchers}
    incoming_events = {key: set() for key in dispatchers}
    for caller, call, _ in calls:
        target = call["function"]
        if target not in dispatchers:
            continue
        event_arg = call["args"][0]
        if event_arg.get("op") != "literal" or event_arg.get("type") != "string":
            raise ValueError(
                "TYPED_PLAYER_STATE_ACTOR_AMBIGUOUS: dynamic semantic event"
            )
        event = event_arg["value"]
        incoming_events[target].add(event)
        if (
            caller not in dispatchers
            and caller not in nonplayer_roots
            and call["args"][1]
            == {
                "op": "ref",
                "name": actors.get(caller),
            }
        ):
            safe_events[target].add(event)
        else:
            unsafe_events[target].add(event)

    def actor_available(caller, event):
        if caller not in actors or caller in nonplayer_roots:
            return False
        if caller not in dispatchers:
            return True
        allowed = safe_events[caller] - unsafe_events[caller]
        return (
            event in allowed
            if event is not None
            else bool(incoming_events[caller]) and incoming_events[caller] <= allowed
        )

    changed = True
    while changed:
        changed = False
        for caller, node, event in calls:
            if not actor_available(caller, event):
                continue
            actor = actors[caller]
            target = functions.get(node["function"])
            if target is None or target["id"] in actors:
                continue
            for parameter, argument in zip(target["parameters"], node["args"]):
                if parameter["type"] == "object" and argument == {
                    "op": "ref",
                    "name": actor,
                }:
                    actors[target["id"]] = parameter["name"]
                    changed = True
                    break

    def needs_identity(node):
        if node.get("op") not in {"state_get", "state_set"}:
            return False
        key, context = node["key"], node["context"]
        return (
            key.get("op") == "literal"
            and key.get("value") in scoped
            and context.get("op") == "map"
            and not any(
                entry["key"] == {"op": "literal", "type": "string", "value": "player"}
                for entry in context["entries"]
            )
        )

    needed = {
        fn["id"]
        for fn in functions.values()
        if any(needs_identity(node) for node in _nodes(fn["body"]))
    }
    changed = True
    while changed:
        changed = False
        for caller, call, _ in calls:
            if (
                call["function"] in needed
                and call["function"] not in dispatchers
                and caller not in needed
            ):
                needed.add(caller)
                changed = True
    if needed & nonplayer_roots:
        raise ValueError(
            "TYPED_PLAYER_STATE_ACTOR_AMBIGUOUS: mixed player/server event roots: "
            + ", ".join(sorted(needed & nonplayer_roots))
        )
    for caller, call, event in calls:
        target = functions.get(call["function"])
        if target is None or target["id"] not in needed or target["id"] in dispatchers:
            continue
        actor = actors.get(target["id"])
        parameter_names = [p["name"] for p in target["parameters"]]
        if (
            not actor_available(caller, event)
            or actor not in parameter_names
            or call["args"][parameter_names.index(actor)]
            != {"op": "ref", "name": actors[caller]}
        ):
            raise ValueError("TYPED_PLAYER_STATE_ACTOR_AMBIGUOUS: " + target["id"])
    injected = False
    for fn in [
        *functions.values(),
        {"id": "<initialize>", "body": result.get("initialize", [])},
    ]:
        for node, event in _guarded_nodes(fn["body"]):
            if not needs_identity(node):
                continue
            key = node["key"]
            if key.get("op") != "literal" or key.get("value") not in scoped:
                continue
            context = node["context"]
            if context.get("op") != "map":
                # An explicit computed context already has its own typed contract.
                continue
            if any(
                entry["key"] == {"op": "literal", "type": "string", "value": "player"}
                for entry in context["entries"]
            ):
                continue
            actor = actors.get(fn["id"])
            if actor is None or not actor_available(fn["id"], event):
                raise ValueError(
                    "TYPED_PLAYER_STATE_ACTOR_REQUIRED: "
                    + fn["id"]
                    + ": "
                    + key["value"]
                )
            context["entries"].append(
                {
                    "key": {"op": "literal", "type": "string", "value": "player"},
                    "value": {
                        "op": "capability",
                        "id": "player.uuid",
                        "args": [{"op": "ref", "name": actor}],
                    },
                }
            )
            injected = True
    if injected:
        from .typed_host_capabilities import typed_host_capability_contracts
        from .typed_plan_ir import validate_typed_plan_ir

        contracts.setdefault(
            "player.uuid", typed_host_capability_contracts()["player.uuid"]
        )
        result = validate_typed_plan_ir(result, capabilities=contracts)
    return result, contracts
