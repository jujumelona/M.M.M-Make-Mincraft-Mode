"""Add missing host-bound UI action commands to an existing saved Typed PlanIR.

Repair only the executable action overlay; do not replace the saved program,
invent gameplay effects, or downgrade the production gameplay coverage gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .authored_structured_design import active_concern_records


def _mutating_ui_actions(sections: Mapping[str, Any]) -> tuple[str, ...]:
    algorithm = active_concern_records(sections, "algorithm")
    state = active_concern_records(sections, "state_model")
    if not (
        algorithm.get("atomic_mutations")
        or state.get("transitions")
        or state.get("updates")
    ):
        return ()
    rows = active_concern_records(sections, "resources_and_ui").get(
        "interactions", ()
    )
    return tuple(dict.fromkeys(
        str(row.get("server_action") or "").strip()
        for row in rows
        if isinstance(row, Mapping) and str(row.get("server_action") or "").strip()
    ))


def authored_ui_action_repair_needed(
    plan: Any, *, source_reuse: Mapping[str, Any] | None = None,
) -> bool:
    actions = _mutating_ui_actions(plan.structured_sections)
    if len(actions) < 2:
        return False
    if isinstance(source_reuse, Mapping) and any(
        isinstance(row, Mapping) and row.get("mode") == "source_transplant"
        for row in source_reuse.get("capabilities", ())
    ):
        return False
    functions = {
        str(fn.get("id") or ""): fn
        for fn in plan.typed_plan_ir.get("functions", ())
        if isinstance(fn, Mapping)
    }
    bindings = {
        str(binding.get("function") or ""): binding
        for binding in plan.typed_plan_ir.get("event_bindings", ())
        if isinstance(binding, Mapping)
    }
    return not all(
        (name := f"uiAction{index}_command") in functions
        and bindings.get(name, {}).get("event") == "command"
        for index in range(1, len(actions) + 1)
    )


def repair_authored_ui_action_ir(
    router: Any,
    source_text: str,
    structured_sections: Mapping[str, Any],
    typed_plan_ir: Mapping[str, Any],
    *,
    capabilities: Mapping[str, Any] | None = None,
    budget: Any = None,
) -> dict[str, Any]:
    """Build missing real commands from saved actions without changing old logic."""
    from .typed_plan_authoring import author_semantic_game_dispatch
    from .typed_plan_ir import validate_typed_plan_ir

    actions = _mutating_ui_actions(structured_sections)
    if len(actions) < 2:
        raise ValueError("AUTHORED_UI_ACTION_REPAIR_NOT_REQUIRED")

    repaired = deepcopy(dict(typed_plan_ir))
    functions = repaired.get("functions", [])
    bindings = repaired.get("event_bindings", [])
    if not isinstance(functions, list) or not isinstance(bindings, list):
        raise ValueError("AUTHORED_UI_ACTION_IR_MALFORMED")

    if not any(
        isinstance(binding, Mapping) and binding.get("event") == "command"
        for binding in bindings
    ):
        raise ValueError(
            "TYPED_UI_ACTION_COMMAND_UNBOUND: saved gameplay design has no "
            "player-invokable command. Repair the integration entry points."
        )

    existing_ids = {
        str(fn.get("id") or "")
        for fn in functions if isinstance(fn, Mapping)
    }
    if "ui_action_dispatch" in existing_ids or any(
        f"uiAction{index}_command" in existing_ids
        for index in range(1, len(actions) + 1)
    ):
        raise ValueError(
            "AUTHORED_UI_ACTION_PARTIAL_REPAIR: existing action binding needs "
            "explicit reconciliation, not a duplicate handler"
        )
    reserved_indices = {
        binding.get("entry_point_index")
        for binding in bindings if isinstance(binding, Mapping)
    }
    # The persisted design already indexes its authored integration hooks.
    integration = active_concern_records(
        structured_sections, "integration",
    ).get("entry_points", ())
    synthetic_first_index = len(integration)
    if any(
        synthetic_first_index + index in reserved_indices
        for index in range(len(actions))
    ):
        raise ValueError("AUTHORED_UI_ACTION_INDEX_COLLISION")

    rules = author_semantic_game_dispatch(
        router, source_text, structured_sections, capabilities,
        coverage_refs=("resources_and_ui.interactions",),
        budget=budget, bound_events=("command",),
    )
    if not rules or rules[-1].get("op") != "return":
        raise ValueError("AUTHORED_UI_ACTION_DISPATCH_INVALID")

    # Require a distinct guarded *mutation* for every action, not just a GUI
    # registration or the unrelated generic command state counter.
    from .typed_host_capabilities import typed_host_capability_contracts
    mutating_caps = {
        key for key, contract in typed_host_capability_contracts().items()
        if contract.get("gameplay_mutation") is True
    }
    guarded: set[str] = set()

    def nodes(value: Any):
        if isinstance(value, Mapping):
            yield value
            for child in value.values():
                yield from nodes(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from nodes(child)

    for stmt in rules[:-1]:
        if not isinstance(stmt, Mapping) or stmt.get("op") != "if":
            continue
        condition = stmt.get("condition", {})
        literals = {
            str(node.get("value") or "")
            for node in nodes(condition)
            if node.get("op") == "literal" and node.get("type") == "string"
        }
        mutated = any(
            node.get("op") == "state_set"
            or (node.get("op") == "capability"
                and str(node.get("id") or "") in mutating_caps)
            for node in nodes(stmt.get("then", ()))
        )
        if mutated:
            guarded.update(literals & set(actions))
    missing = set(actions) - guarded
    if missing:
        raise ValueError(
            "AUTHORED_UI_ACTION_REPAIR_INCOMPLETE: "
            + ", ".join(sorted(missing))
        )

    params = [
        {"name": "event", "type": "string"},
        {"name": "primary", "type": "object"},
        {"name": "secondary", "type": "object"},
        {"name": "flag", "type": "boolean"},
    ]
    functions.append({
        "id": "ui_action_dispatch",
        "parameters": params,
        "return_type": "int",
        "body": rules,
        "covers": ["resources_and_ui.interactions"],
    })
    used_literals = {
        str(binding.get("config", {}).get("literal") or "")
        for binding in bindings if isinstance(binding, Mapping)
        and binding.get("event") == "command"
    }

    def literal(kind: str, value: Any) -> dict[str, Any]:
        return {"op": "literal", "type": kind, "value": value}

    for index, action in enumerate(actions, start=1):
        name = f"uiAction{index}_command"
        command = f"mmm_action_{index}"
        while command in used_literals:
            command += "_ui"
        used_literals.add(command)
        functions.append({
            "id": name,
            "parameters": [{"name": "source", "type": "object"}],
            "return_type": "int",
            "covers": ["resources_and_ui.interactions"],
            "body": [{
                "op": "return",
                "value": {
                    "op": "call",
                    "function": "ui_action_dispatch",
                    "args": [
                        literal("string", action),
                        {"op": "ref", "name": "source"},
                        literal("object", None),
                        literal("boolean", False),
                    ],
                },
            }],
        })
        bindings.append({
            "event": "command", "function": name,
            "entry_point_index": synthetic_first_index + index - 1,
            "config": {"literal": command, "permission_level": 0},
        })
    return validate_typed_plan_ir(repaired, capabilities=capabilities)


__all__ = [
    "authored_ui_action_repair_needed", "repair_authored_ui_action_ir",
]
