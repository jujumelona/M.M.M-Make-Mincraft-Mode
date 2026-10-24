"""Shared host-owned action binding helpers without planner dependencies."""
from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any
from .request_record_ownership import action_identity_payload, same_action_source


def is_presentation_only_interaction(record, state) -> bool:
    """Identify a *pure* screen-open request, never a gameplay transaction.

    A button displayed on the resulting screen is not the action of pressing
    that button. A screen-open operation has no typed server command body; a
    later purchase/upgrade/launch must have its own source-owned state rule.

    This is deliberately conservative. Ambiguous actions still fail the
    executable-effect preflight rather than silently disappearing.
    """
    if not isinstance(record, Mapping):
        return False
    action = record.get("server_action")
    if not isinstance(action, (str, Mapping)):
        return False
    trigger = str(record.get("trigger") or "").strip()
    intent = str(action.get("intent") if isinstance(action, Mapping) else action).strip()
    if not trigger or not intent:
        return False

    # Navigation is a specific player event, not merely the presence of the
    # word "screen" in a transaction such as "purchase from upgrade screen".
    navigation = re.match(
        r"^(?:(?:the )?(?:player|user|client)\s+)?"
        r"(?:opens?|views?|displays?|shows?)\s+",
        trigger, re.I,
    )
    screen = re.search(r"\b(?:screen|menu|panel|window|interface|dialog|gui|list|inventory|catalog|catalogue|overview|details)\b",
                       trigger, re.I)
    if not (navigation and screen):
        return False
    if not (intent.casefold() == trigger.casefold() or re.match(
        r"^(?:open|view|display|show)\s+(?:the\s+|a\s+)?"
        r"(?:screen|menu|panel|window|interface|dialog|gui|list|inventory|catalog|catalogue|overview|details)\b",
        intent, re.I,
    )):
        return False
    # Never classify an explicit multi-step purchase/launch as read-only.
    if re.search(
        r"\b(?:and|then)\s+(?:buy|buys|purchase|purchases|sell|sells|"
        r"craft|crafts|construct|constructs|upgrade|upgrades|"
        r"launch|launches|equip|equips|change|changes|confirm|confirms|"
        r"spend|spends|deduct|deducts|save|saves)\b",
        trigger + " " + intent, re.I,
    ):
        return False

    for concern in ("updates", "transitions"):
        for change in state.get(concern, ()):
            if (isinstance(change, Mapping)
                    and same_action_source(change, record)
                    and change.get("mutation")):
                # Even an "open screen" verb is executable when the source
                # explicitly assigns it a state transition.
                return False
    return True


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
        if (isinstance(row, Mapping) and action_record_identity(row)
            and not is_presentation_only_interaction(row, state))
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


