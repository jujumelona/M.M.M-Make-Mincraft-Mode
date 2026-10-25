"""Plan an original player action independently of its eligibility condition.

Exact quotes establish origin, not semantic or gameplay certification. The
planner interprets the complete behavior; the host preserves it and binds the
interpretation to its bytes before producing the form and server event.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

from .player_action_modality import ACTION_FIELDS, assess_player_action_modality
from .source_occurrence_error import SourceHookUnavailable
from .structured_output import StructuredOutputValidationError, validate_structured_output

_SCHEMA = "mmm/player-action-evidence-v1"
_FIELDS = ACTION_FIELDS


def _behavior_hash(behavior):
    return "sha256:" + hashlib.sha256(json.dumps(
        dict(behavior), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _unavailable(behavior, reason):
    return SourceHookUnavailable(str(behavior.get("behavior_id") or ""),
                                 reason=reason, inspected=0)


def _require_original_quote(behavior, field, quote):
    if (field not in _FIELDS or not isinstance(quote, str) or not quote.strip()
            or not isinstance(behavior.get(field), str) or quote not in behavior[field]):
        raise _unavailable(behavior, "action_evidence_quote_not_in_original_field")


def validate_player_action_evidence(behavior, evidence):
    if (not isinstance(evidence, Mapping)
            or set(evidence) != {"schema", "behavior_sha256", "request_field", "request_quote", "activation_scope"}
            or evidence.get("schema") != _SCHEMA):
        raise ValueError("ACTION_EVIDENCE_SCHEMA_INVALID")
    if evidence["behavior_sha256"] != _behavior_hash(behavior):
        raise ValueError("ACTION_EVIDENCE_BEHAVIOR_MISMATCH")
    field, quote = evidence["request_field"], evidence["request_quote"]
    _require_original_quote(behavior, field, quote)
    if evidence["activation_scope"] != assess_player_action_modality(behavior, quote):
        raise ValueError("ACTION_EVIDENCE_SCOPE_MISMATCH")


def author_player_action_evidence(router, behavior):
    schema = {"type": "object", "properties": {
        "activation": {"type": "string", "enum": ["typed_ui_action", "unavailable"]},
        "request_field": {"type": "string", "enum": list(_FIELDS)},
        "request_quote": {"type": "string", "maxLength": 512},
    }, "required": ["activation", "request_field", "request_quote"], "additionalProperties": False}
    response = router.generate_structured_data(
        "planner", [{"role": "system", "content": (
            "Determine whether the ORIGINAL behavior requests a voluntary player action. "
            "Read trigger, intent, effects and source_quote together. The trigger may "
            "state eligibility (having resources, being ready) while the requested "
            "action is expressed in another field. A missing method or Fabric event "
            "does not prohibit implementing that original interaction. Choose "
            "typed_ui_action only when the contract calls for the player to perform "
            "the action. Select its original field and quote the exact action clause "
            "from that field, preserving its subject and negation. The host will "
            "provide the form, validated server transport and event; all original "
            "conditions and effects remain required. Having/wanting something, an "
            "arrival, damage, a tick, a world condition, or an automatic effect alone "
            "is not a voluntary action. Never invent a command, key, click, input "
            "or action quote to replace a passive trigger. If the complete contract "
            "has no voluntary action, choose unavailable and an empty quote. "
            "This is an unverified implementation plan, not proof of gameplay.")},
            {"role": "user", "content": json.dumps(dict(behavior), ensure_ascii=False)}],
        data_name="source_owned_player_action_evidence", parameters=schema,
    )
    try:
        selected = json.loads(validate_structured_output(
            json.dumps(response, ensure_ascii=False), response_format="json", response_schema=schema))
    except StructuredOutputValidationError:
        # A malformed or abstaining model cannot veto a positive, immutable
        # source action. The host can independently recover only an explicitly
        # player-owned verb that agrees with the original intent.
        selected = {"activation": "unavailable", "request_field": "trigger",
                    "request_quote": ""}
    if selected["activation"] != "typed_ui_action":
        from .host_player_action_recovery import recover_player_action_quote

        recovered = recover_player_action_quote(behavior)
        if recovered is None:
            raise _unavailable(behavior, "original_trigger_has_no_owned_ui_activation")
        field, quote = recovered
        selected = {"activation": "typed_ui_action", "request_field": field,
                    "request_quote": quote}
    evidence = {"schema": _SCHEMA, "behavior_sha256": _behavior_hash(behavior),
                "request_field": selected["request_field"],
                "request_quote": selected["request_quote"].strip()}
    _require_original_quote(behavior, evidence["request_field"], evidence["request_quote"])
    evidence["activation_scope"] = assess_player_action_modality(behavior, evidence["request_quote"])
    validate_player_action_evidence(behavior, evidence)
    return evidence
