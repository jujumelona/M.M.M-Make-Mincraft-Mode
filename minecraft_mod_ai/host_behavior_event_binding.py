"""Lower a gameplay trigger to an inspected hook, retaining its condition.

The hook supplies observable inputs; it need not be named after the gameplay
concept. A predicate and effects implement that concept separately. The host
retains the original trigger and never substitutes an arbitrary fallback event.
"""
from __future__ import annotations

from collections import defaultdict
import json


class HostEventBindingError(ValueError):
    def __init__(self, behavior, response):
        self.failed_response = json.dumps(response, ensure_ascii=False, default=str)
        super().__init__("JAVA_HOST_BEHAVIOR_EVENT_UNBOUND: "
                         + str(behavior.get("behavior_id"))
                         + "; trigger=" + str(behavior.get("trigger"))
                         + "; selection=" + self.failed_response)


_INSTRUCTION = (
    "Bind the original gameplay trigger to a verified Fabric execution hook. "
    "The hook is an OBSERVATION POINT supplying typed inputs, not the complete "
    "gameplay condition. Do not require a Fabric event named after the gameplay "
    "concept: an alien encounter, spacecraft placement or a transaction is a "
    "domain condition, not a literal Fabric callback name. Prefer the specific "
    "interaction/entity/block event when its inputs allow checking that condition. "
    "For ongoing world conditions, a verified level/server tick may observe "
    "the condition; the host runs a separate boolean predicate before effects "
    "for void callbacks. Do not replace the original trigger with unconditional "
    "ticks or arbitrary interactions. Non-void callbacks must preserve the "
    "callback's result semantics and check the original condition in their body. "
    "Use UNBOUND only if no offered hook exposes inputs from which the original "
    "trigger can be implemented using verified APIs. Do not invent any event. "
    "Select only the requested identifier; no Java source."
)


def select_behavior_callback(router, behavior, candidates):
    by_event = {row["event_id"]: row for row in candidates}
    requested = str(behavior.get("event_binding") or "")
    if not requested and behavior.get("event_interface"):
        matches = [row["event_id"] for row in candidates
                   if row["event_interface"] == behavior["event_interface"]]
        if len(matches) == 1:
            requested = matches[0]
    if requested:
        if requested not in by_event:
            raise HostEventBindingError(behavior, {"event_binding": requested})
        return by_event[requested]

    context = {key: behavior.get(key) for key in (
        "behavior_id", "intent", "trigger", "effects", "state_reads", "state_writes")}
    groups = defaultdict(list)
    for row in candidates:
        groups[row["event_owner"]].append(row)
    available = list(candidates)
    # 26.2 has over a hundred common event fields. Sending every SAM on every
    # choice repeats a 55K-character catalogue. Select a real holder first,
    # then send only that holder's field contracts, without truncating data.
    if len(groups) > 1:
        payload = {**context, "verified_event_groups": [
            {"event_owner": owner, "fields": [row["event_field"] for row in rows]}
            for owner, rows in sorted(groups.items())]}
        selection = _select(router, payload, "event_owner", sorted(groups),
                            "select_verified_fabric_event_owner")
        owner = selection.get("event_owner")
        if owner not in groups:
            raise HostEventBindingError(behavior, selection)
        available = groups[owner]
    selection = _select(router, {**context, "verified_listeners": available},
                        "event_binding", [row["event_id"] for row in available],
                        "select_verified_fabric_event")
    selected = selection.get("event_binding")
    if selected not in {row["event_id"] for row in available}:
        raise HostEventBindingError(behavior, selection)
    return by_event[selected]


def _select(router, payload, field, choices, action):
    return router.generate_structured_data(
        "coder", [{"role": "system", "content": _INSTRUCTION},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
        data_name=action, parameters={"type": "object", "properties": {
            field: {"type": "string", "enum": ["UNBOUND", *choices]}},
            "required": [field], "additionalProperties": False})
