"""Add missing host-bound UI action commands to an existing saved Typed PlanIR.

Repair only the executable action overlay; do not replace the saved program,
invent gameplay effects, or downgrade the production gameplay coverage gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .authored_structured_design import active_concern_records
from .authored_ui_action_bindings import _mutating_ui_actions



def authored_ui_action_repair_needed(
    plan: Any,
) -> bool:
    actions = _mutating_ui_actions(plan.structured_sections)
    if len(actions) < 2:
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

    repaired, synthetic_first_index = _validate_repair_target(
        structured_sections, typed_plan_ir, actions,
    )

    rules = author_semantic_game_dispatch(
        router, source_text, structured_sections, capabilities,
        coverage_refs=("resources_and_ui.interactions",),
        budget=budget, bound_events=("command",),
    )
    if not rules or rules[-1].get("op") != "return":
        raise ValueError("AUTHORED_UI_ACTION_DISPATCH_INVALID")

    _assert_action_mutation_guards(rules, actions)
    _append_repaired_ui_bindings(repaired, rules, actions, synthetic_first_index)


    return validate_typed_plan_ir(repaired, capabilities=capabilities)


__all__ = [
    "authored_ui_action_repair_needed", "repair_authored_ui_action_ir",
]



def _validate_repair_target(
    sections: Mapping[str, Any], ir: Mapping[str, Any],
    actions: tuple[str, ...],
) -> tuple[dict[str, Any], int]:
    """Validate existing command and reserve disjoint synthetic event indexes."""
    repaired = deepcopy(dict(ir))
    functions = repaired.get("functions", [])
    bindings = repaired.get("event_bindings", [])
    if not isinstance(functions, list) or not isinstance(bindings, list):
        raise ValueError("AUTHORED_UI_ACTION_IR_MALFORMED")
    if not any(
        isinstance(b, Mapping) and b.get("event") == "command"
        for b in bindings
    ):
        raise ValueError(
            "TYPED_UI_ACTION_COMMAND_UNBOUND: saved gameplay design has no "
            "player-invokable command. Repair the integration entry points."
        )
    _assert_no_existing_action_bindings(functions, actions)
    integration = active_concern_records(
        sections, "integration",
    ).get("entry_points", ())
    first_index = len(integration)
    _assert_action_entry_slots_free(bindings, first_index, len(actions))
    return repaired, first_index


def _assert_no_existing_action_bindings(
    functions: list[dict[str, Any]], actions: tuple[str, ...],
) -> None:
    ids = {
        str(fn.get("id") or "") for fn in functions
        if isinstance(fn, Mapping)
    }
    if "ui_action_dispatch" in ids or any(
        f"uiAction{index}_command" in ids
        for index in range(1, len(actions) + 1)
    ):
        raise ValueError(
            "AUTHORED_UI_ACTION_PARTIAL_REPAIR: existing action binding needs "
            "explicit reconciliation, not a duplicate handler"
        )


def _assert_action_entry_slots_free(
    bindings: list[dict[str, Any]], first_index: int, count: int,
) -> None:
    occupied = {
        binding.get("entry_point_index")
        for binding in bindings if isinstance(binding, Mapping)
    }
    if any(first_index + index in occupied for index in range(count)):
        raise ValueError("AUTHORED_UI_ACTION_INDEX_COLLISION")

def _action_guard_literal(stmt: Mapping[str, Any]) -> str:
    """Only exact equality against the current dispatcher event is a guard."""
    if stmt.get("op") != "if":
        return ""
    condition = stmt.get("condition")
    if not isinstance(condition, Mapping) or condition.get("op") != "binary":
        return ""
    if condition.get("operator") != "==":
        return ""
    left, right = condition.get("left"), condition.get("right")
    for reference, literal in ((left, right), (right, left)):
        if not isinstance(reference, Mapping) or not isinstance(literal, Mapping):
            continue
        if reference.get("op") == "ref" and reference.get("name") == "event":
            if literal.get("op") == "literal" and literal.get("type") == "string":
                return str(literal.get("value") or "")
    return ""


def _walk_ir_nodes(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_ir_nodes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_ir_nodes(child)


def _is_mutating_action(body: Any, mutating_caps: set[str]) -> bool:
    return any(
        node.get("op") == "state_set"
        or (node.get("op") == "capability"
            and str(node.get("id") or "") in mutating_caps)
        for node in _walk_ir_nodes(body)
    )


def _assert_action_mutation_guards(
    rules: list[dict[str, Any]], actions: tuple[str, ...],
) -> None:
    """Demand an exact guarded gameplay write for every UI action."""
    from .typed_host_capabilities import typed_host_capability_contracts

    mutating_caps = {
        key for key, contract in typed_host_capability_contracts().items()
        if contract.get("gameplay_mutation") is True
    }
    covered = {
        action
        for stmt in rules[:-1]
        if isinstance(stmt, Mapping)
        if (action := _action_guard_literal(stmt)) in actions
        if _is_mutating_action(stmt.get("then", ()), mutating_caps)
    }
    missing = set(actions) - covered
    if missing:
        raise ValueError(
            "AUTHORED_UI_ACTION_REPAIR_INCOMPLETE: "
            + ", ".join(sorted(missing))
        )

def _append_repaired_ui_bindings(
    repaired: dict[str, Any], rules: list[dict[str, Any]],
    actions: tuple[str, ...], synthetic_first_index: int,
) -> None:
    functions = repaired["functions"]
    bindings = repaired["event_bindings"]
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
                        {"op": "literal", "type": "string", "value": action},
                        {"op": "ref", "name": "source"},
                        {"op": "literal", "type": "object", "value": None},
                        {"op": "literal", "type": "boolean", "value": False},
                    ],
                },
            }],
        })
        bindings.append({
            "event": "command", "function": name,
            "entry_point_index": synthetic_first_index + index - 1,
            "config": {"literal": command, "permission_level": 0},
        })

def repair_authored_ui_action_plan(
    router: Any, plan: Any, *, minecraft_version: str = "",
) -> Any:
    """Retain saved design but add missing executable UI action bindings."""
    from .typed_action_authoring import migrate_ui_action_plan
    migrated = migrate_ui_action_plan(router, plan, minecraft_version=minecraft_version)
    if migrated is not plan:
        return migrated
    if any("action" in b.get("config", {}) for b in plan.typed_plan_ir.get("event_bindings", ())):
        return plan
    from dataclasses import replace
    from .typed_host_capabilities import typed_host_capability_contracts
    from .root_cause_trace import emit_root_cause

    if not authored_ui_action_repair_needed(plan):
        return plan
    updated = repair_authored_ui_action_ir(
        router, plan.text, plan.structured_sections, plan.typed_plan_ir,
        capabilities=typed_host_capability_contracts(),
    )
    repaired_plan = replace(plan, typed_plan_ir=updated)
    emit_root_cause(
        "authored_ui_action_ir_repaired",
        stage="production", operation="repair_missing_ui_handlers",
        result="PASS",
        details={
            "policy": "add_only_action_handlers",
            "typed_functions": len(updated.get("functions", ())),
        },
    )
    return repaired_plan
