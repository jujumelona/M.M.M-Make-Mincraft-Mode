"""Host-owned handoff from the original typed UI action to source Java."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy

from .request_record_ownership import (
    SOURCE_REQUIREMENT,
    action_identity_payload,
    same_action_source,
)
from .typed_action_contract import validate_action_contract

SOURCE_UI_CAPABILITY = "ui.invoke_source_action"
_SOURCE_UI_CONTRACT = {
    "owner": "AuthoredSourceActions", "method": "invokeSourceAction",
    "parameters": ["string", "object", "map"], "return_type": "int",
    "gameplay_mutation": False,
}
_JAVA_SCALARS = {
    "boolean": "boolean", "int": "int", "long": "long",
    "double": "double", "string": "java.lang.String",
}


def _effective_state_mutation(mutation, declared):
    """A source write must demonstrably change an owned canonical state."""
    if not isinstance(mutation, Mapping) or mutation.get("target") not in declared:
        return False
    value = mutation.get("value")
    if not isinstance(value, Mapping):
        return False
    op = mutation.get("operator", "=")
    if (op == "=" and value.get("kind") == "state_ref"
            and value.get("name") == mutation.get("target")):
        return False
    if op in {"+=", "-=", "*=", "/="} and value.get("kind") in {"literal", "number"}:
        from decimal import Decimal, InvalidOperation
        raw = value.get("value")
        try:
            numeric = Decimal(str(raw))
        except (InvalidOperation, ValueError, TypeError):
            return True  # Rejected separately by the typed state validator.
        if (op in {"+=", "-="} and numeric == 0) or (
            op in {"*=", "/="} and numeric == 1
        ):
            return False
    return True


def action_identity(metadata):
    encoded = json.dumps(
        metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return "action_" + hashlib.sha256(encoded.encode()).hexdigest()[:20]


def validate_action_metadata(metadata):
    if not isinstance(metadata, Mapping) or set(metadata) != {"intent", "inputs"}:
        raise ValueError("ACTION_METADATA_FIELDS")
    validate_action_contract(
        {
            "id": action_identity(metadata),
            "label": metadata["intent"],
            "inputs": metadata["inputs"],
        }
    )
    if any(row["name"] in {"source", "action_context"} for row in metadata["inputs"]):
        raise ValueError("ACTION_INPUT_RESERVED_ACTOR")
    return deepcopy(dict(metadata))


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
        trigger, re.IGNORECASE,
    )
    screen = re.search(r"\b(?:screen|menu|panel|window|interface|dialog|gui|list|inventory|catalog|catalogue|overview|details)\b",
                       trigger, re.IGNORECASE)
    if not (navigation and screen):
        return False
    if not (intent.casefold() == trigger.casefold() or re.match(
        r"^(?:open|view|display|show)\s+(?:the\s+|a\s+)?"
        r"(?:screen|menu|panel|window|interface|dialog|gui|list|inventory|catalog|catalogue|overview|details)\b",
        intent, re.IGNORECASE,
    )):
        return False
    # Never classify an explicit multi-step purchase/launch as read-only.
    if re.search(
        r"\b(?:and|then)\s+(?:buy|buys|purchase|purchases|sell|sells|"
        r"craft|crafts|construct|constructs|upgrade|upgrades|"
        r"launch|launches|equip|equips|change|changes|confirm|confirms|"
        r"spend|spends|deduct|deducts|save|saves)\b",
        trigger + " " + intent, re.IGNORECASE,
    ):
        return False

    return not _presentation_state_effect(record, state)


def _presentation_state_effect(record, state):
    # Even an "open screen" verb is executable when the source explicitly
    # assigns it a state transition.
    return any(
        isinstance(change, Mapping) and same_action_source(change, record)
        and change.get("mutation")
        for concern in ("updates", "transitions")
        for change in state.get(concern, ())
    )


def action_record_identity(row):
    action = row.get("server_action")
    if isinstance(action, Mapping):
        return action_identity(action_identity_payload(row, validate_action_metadata(action)))
    return str(action or "").strip()


def _section_records(sections, section):
    """Read canonical specification records after the caller's schema boundary."""
    return (sections or {}).get(section, {}).get("specification", {})


def _source_effects(state):
    declared = {row.get("name") for row in state.get("variables", ())}
    return [
        row for concern in ("updates", "transitions")
        for row in state.get(concern, ())
        if any(_effective_state_mutation(change, declared)
               for change in (row.get("mutation") or ()))
    ]


def _eligible_interactions(rows, state, requirement_ids):
    for row in rows:
        action = row.get("server_action")
        if not isinstance(action, (str, Mapping)) or not action:
            continue
        if isinstance(action, Mapping):
            validate_action_metadata(action)
        if is_presentation_only_interaction(row, state):
            continue
        source = row.get(SOURCE_REQUIREMENT)
        if requirement_ids is not None and source not in requirement_ids:
            raise ValueError("SOURCE_UI_SOURCE_UNKNOWN: " + str(source))
        if isinstance(source, str) and source.strip():
            yield row


def deferred_ui_interactions(sections, *, requirement_ids=None) -> list[dict]:
    """Find source operations with no effective canonical transaction of their own."""
    state = _section_records(sections, "state_model")
    rows = _section_records(sections, "resources_and_ui").get("interactions", ())
    effects = _source_effects(state)
    return [
        deepcopy(row) for row in _eligible_interactions(rows, state, requirement_ids)
        if not any(same_action_source(effect, row) for effect in effects)
    ]


def source_ui_capabilities(capabilities, sections) -> dict:
    """Extend a request's capabilities only when source transport is necessary."""
    result = deepcopy(dict(capabilities or {}))
    if deferred_ui_interactions(sections):
        result[SOURCE_UI_CAPABILITY] = deepcopy(_SOURCE_UI_CONTRACT)
    return result


def _source_ui_delegate_body(action_id, inputs):
    return [{"op": "return", "value": {
        "op": "capability", "id": SOURCE_UI_CAPABILITY, "args": [
            {"op": "literal", "type": "string", "value": action_id},
            {"op": "ref", "name": "source"},
            {"op": "map", "entries": [{
                "key": {"op": "literal", "type": "string", "value": row["name"]},
                "value": {"op": "ref", "name": row["name"]},
            } for row in inputs]},
        ],
    }}]


def source_ui_body(record, sections, capabilities) -> list | None:
    """Lower only admitted deferred actions; keep the original input and actor ABI."""
    if SOURCE_UI_CAPABILITY not in (capabilities or {}):
        return None
    if not isinstance(record.get("server_action"), Mapping):
        return None
    if record not in deferred_ui_interactions(sections):
        return None
    metadata = validate_action_metadata(record["server_action"])
    return _source_ui_delegate_body(action_record_identity(record), metadata["inputs"])


def source_ui_preflight_sections(sections, *, enabled=True) -> dict:
    """Provide a legacy mutation-presence view without changing source obligations."""
    result = deepcopy(dict(sections or {}))
    if not enabled:
        return result
    deferred = deferred_ui_interactions(sections)
    specification = result.get("resources_and_ui", {}).get("specification", {})
    if "interactions" in specification:
        specification["interactions"] = [
            row for row in specification["interactions"] if row not in deferred
        ]
    return result


def _source_ui_event_abi(action_id, inputs, package):
    suffix = action_id.removeprefix("action_")
    owner = package + ".AuthoredSourceActions"
    event_field = "EVENT_" + suffix
    return {
        "event_id": owner + "#" + event_field,
        "event_owner": owner, "event_field": event_field,
        "event_interface": owner + ".Listener_" + suffix,
        "method": "onAction",
        "parameters": ["net.minecraft.commands.CommandSourceStack",
                       *(_JAVA_SCALARS[row["type"]] for row in inputs)],
        "parameter_names": ["source", *("input_" + str(i) for i in range(len(inputs)))],
        "return_type": "int", "throws": [], "type_parameters": [],
    }


def source_ui_callback(record, *, package) -> dict:
    """Return the exact registered Fabric Event ABI used by the Java renderer."""
    source = record.get(SOURCE_REQUIREMENT)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("SOURCE_UI_SOURCE_UNKNOWN: " + str(source))
    metadata = validate_action_metadata(record["server_action"])
    action_id = action_record_identity(record)
    return {
        **_source_ui_event_abi(action_id, metadata["inputs"], package),
        "action_id": action_id, SOURCE_REQUIREMENT: source,
        "trigger": record.get("trigger", ""), "inputs": metadata["inputs"],
    }
