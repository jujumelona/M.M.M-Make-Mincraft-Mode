"""Shared host-owned action binding helpers without planner dependencies."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from .request_record_ownership import action_identity_payload


def _mutating_ui_actions(sections: Mapping[str, Any]) -> tuple[str, ...]:
    from .authored_structured_design import active_concern_records

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
        action_record_identity(row)
        for row in rows
        if isinstance(row, Mapping) and action_record_identity(row)
    ))


def action_record_identity(row):
    action = row.get("server_action")
    if isinstance(action, Mapping):
        from .typed_action_authoring import action_identity, validate_action_metadata
        return action_identity(action_identity_payload(row, validate_action_metadata(action)))
    return str(action or "").strip()



def _append_ui_action_bindings(
    *, actions: tuple[str, ...], entry_points: Any,
    specs: list[dict[str, Any]], bindings: list[dict[str, Any]],
    known_ids: set[str],
) -> dict[str, str]:
    """Bind distinct generated player commands to authored action identities."""
    from .typed_event_ir import EVENT_PARAMETERS, EVENT_SIGNATURES

    used_literals = {
        str(binding["config"]["literal"])
        for binding in bindings if binding["event"] == "command"
    }
    action_by_function: dict[str, str] = {}
    for ordinal, action in enumerate(actions, start=1):
        command_literal = f"mmm_action_{ordinal}"
        while command_literal in used_literals:
            command_literal += "_ui"
        used_literals.add(command_literal)
        function_id = f"uiAction{ordinal}_command"
        if function_id in known_ids:
            raise ValueError("TYPED_UI_ACTION_FUNCTION_COLLISION")
        known_ids.add(function_id)
        specs.append({
            "id": function_id,
            "parameters": [
                {"name": name, "type": type_name}
                for name, type_name in EVENT_PARAMETERS["command"]
            ],
            "return_type": EVENT_SIGNATURES["command"][1],
            "covers": ["resources_and_ui.interactions"],
        })
        bindings.append({
            "event": "command",
            "function": function_id,
            "entry_point_index": len(entry_points) + ordinal - 1,
            "config": {"literal": command_literal, "permission_level": 0},
        })
        action_by_function[function_id] = action
    return action_by_function


