"""Lower semantic observation operators to the inspected Fabric ABI.

An application state transition is implemented by a rule, not by finding a
native event with the application's name. Native occurrences retain their
actual hook and cancellation/ordering semantics; they never fall back to ticks.
"""
from __future__ import annotations

import json

from .fixed_template_generation import _validate_native_arguments


PAGE_SIZE = 12
_TICKS = "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents"
_WORLD_INPUTS = {"net.minecraft.server.level.ServerLevel", "net.minecraft.server.world.ServerWorld"}
_SERVER_INPUT = "net.minecraft.server.MinecraftServer"


def choose_rule_data(router, name, context, properties, instruction):
    schema = {"type": "object", "properties": properties,
              "required": list(properties), "additionalProperties": False}
    response = router.generate_structured_data(
        "coder", [{"role": "system", "content": instruction},
                  {"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
        data_name=name, parameters=schema)
    return _validate_native_arguments(response, schema)


def bind_rule_observation(router, behavior, candidates):
    by_id = {row["event_id"]: row for row in candidates}
    explicit = behavior.get("event_binding")
    if explicit:
        if explicit not in by_id:
            raise ValueError("RULE_UNINSPECTED_EVENT: " + str(explicit))
        return {"mode": "native_event", "event": by_id[explicit]}
    # These are adapter operations, not prompt classifiers. Require the exact
    # inspected owner, field, input and void SAM before emitting registration.
    ticks = {}
    for row in candidates:
        if row.get("event_owner") != _TICKS or row.get("return_type") != "void":
            continue
        params = row.get("parameters", [])
        field = row.get("event_field")
        if field in {"END_LEVEL_TICK", "END_WORLD_TICK"} and len(params) == 1 and params[0] in _WORLD_INPUTS:
            ticks["world_state"] = row
        elif field == "END_SERVER_TICK" and params == [_SERVER_INPUT]:
            ticks["server_state"] = row
    options = [name for name in ("world_state", "server_state") if name in ticks]
    options.append("native_event")
    context = {"behavior": {key: behavior[key] for key in (
        "behavior_id", "intent", "trigger", "effects", "state_reads", "state_writes")
        if key in behavior}}
    mode = choose_rule_data(router, "rule_observation", context,
        {"observation": {"type": "string", "enum": options}},
        "Choose how to observe ONE original behavior. world_state polls each real world; "
        "server_state polls the server. Time boundaries, changed state and combinations "
        "of observable conditions are rule predicates, not named native events. "
        "The host supplies checked clock/state inputs and separately compiles predicates, "
        "change detection and effects. native_event requires the actual occurrence's "
        "ordering, interaction inputs or cancellation (for example before damage). "
        "Preserve every part of an OR trigger. No API names or Java.")
    mode = mode["observation"]
    if mode in ticks:
        return {"mode": mode, "event": ticks[mode]}
    ordered = sorted(candidates, key=lambda row: row["event_id"])
    for offset in range(0, len(ordered), PAGE_SIZE):
        page = ordered[offset:offset + PAGE_SIZE]
        selected = choose_rule_data(router, "rule_native_event_" + str(offset // PAGE_SIZE),
            {**context, "events": [
                {key: row[key] for key in ("event_id", "parameters", "return_type")}
                for row in page], "later_events": len(ordered) - offset - len(page)},
            {"event_id": {"type": "string", "enum": [
                *[row["event_id"] for row in page], "next_page"]}},
            "Select the inspected occurrence that preserves the original trigger and "
            "ordering, or next_page to examine other real hooks. Never select an "
            "unrelated hook just to make progress. No Java.")["event_id"]
        if selected != "next_page":
            return {"mode": mode, "event": by_id[selected]}
    raise ValueError("RULE_NATIVE_HOOK_REQUIRED: " + str(behavior.get("behavior_id")))
