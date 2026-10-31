"""Lower a gameplay trigger to an inspected hook, retaining its condition.

The hook supplies observable inputs; it need not be named after the gameplay
concept. A predicate and effects implement that concept separately. The host
retains the original trigger and never substitutes an arbitrary fallback event.
"""
from __future__ import annotations

from collections import defaultdict
import json
import re


# These are *intrinsic* occurrence semantics of a verified Fabric hook.
# Creative pick-block (middle-click / clone-block) is not block mining,
# item collection, or a completed block-break occurrence. A model may not
# turn a verified but wrong listener into proof of the requested trigger.
_PICK_ITEM_OWNER = "net.fabricmc.fabric.api.event.player.PlayerPickItemEvents"
_PICK_BLOCK_ACTION = re.compile(
    r"\b(?:pick[- ]block|pick[- ]item[- ]from[- ]block|clone[- ]block|"
    r"middle[- ]click(?:s|ing|ed)?\s+(?:on\s+)?(?:a\s+|the\s+)?block)\b"
    r"|블록\s*선택",
    re.IGNORECASE,
)
_PICK_ENTITY_ACTION = re.compile(
    r"\b(?:pick[- ]entity|clone[- ]entity|"
    r"middle[- ]click(?:s|ing|ed)?\s+(?:on\s+)?(?:an?\s+|the\s+)?entity)\b"
    r"|개체\s*선택",
    re.IGNORECASE,
)


def _occurrence_admissible(behavior, event):
    owner = str(event.get("event_owner") or "")
    if owner != _PICK_ITEM_OWNER:
        return True
    # Pick-item event only fires when the pick-block control is used. This
    # must be an actual requested occurrence, not merely a block-related
    # noun in a mining/gathering, UI or resource-farming requirement.
    requested = " ".join(str(behavior.get(k) or "") for k in (
        "trigger", "intent", "effects", "source_quote"))
    field = str(event.get("event_field") or "").upper()
    if field == "BLOCK":
        return bool(_PICK_BLOCK_ACTION.search(requested))
    if field == "ENTITY":
        return bool(_PICK_ENTITY_ACTION.search(requested))
    return False


_BLOCK_BREAK_OWNER = "net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents"
_BLOCK_MINING_TRIGGER = re.compile(
    r"\b(?:mine|mines|mined|mining|harvest(?:s|ed|ing)?\s+(?:a\s+|the\s+)?block|"
    r"break(?:s|ing)?\s+(?:a\s+|the\s+)?block|"
    r"dig(?:s|ging)?\s+(?:a\s+|the\s+)?block)\b"
    r"|채굴|블록\s*파괴",
    re.IGNORECASE,
)
_NEGATED_BLOCK_MINING = re.compile(
    r"\b(?:not|never|without)\s+(?:\w+\s+){0,2}"
    r"(?:mine|mines|mined|mining|harvest|break|dig)\b",
    re.IGNORECASE,
)


def _verified_block_mining_occurrence(behavior, candidates):
    """Use a known Fabric occurrence rather than a model-chosen nearby event.

    Only the exact inspected AFTER hook is eligible. It fires after a player
    has actually broken a block, unlike PickItemFromBlock (creative selection),
    USE_ON, item clicks or ticks. This is a general event-kind adapter;
    it does not choose a block, resource, payout, world state or game logic.
    Other/ambiguous triggers still use ordinary verified event selection.
    """
    trigger = str(behavior.get("trigger") or "")
    if (not _BLOCK_MINING_TRIGGER.search(trigger)
            or _NEGATED_BLOCK_MINING.search(trigger)):
        return None
    matches = [row for row in candidates
               if row.get("event_owner") == _BLOCK_BREAK_OWNER
               and row.get("event_field") == "AFTER"]
    return matches[0] if len(matches) == 1 else None


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
    "for every callback and separately compiles its result branches. Do not "
    "replace the original trigger with unconditional ticks or arbitrary interactions. "
    "Use UNBOUND only if no offered hook exposes inputs from which the original "
    "trigger can be implemented using verified APIs. Do not invent any event. "
    "Select only the requested identifier; no Java source."
)


def select_behavior_callback(router, behavior, candidates):
    candidates = [row for row in candidates
                  if _occurrence_admissible(behavior, row)]
    # Do not let the model replace a known block-break occurrence with
    # middle-click pick-block, an unrelated item callback or an idle tick.
    # The event still needs a host-compiled predicate and gameplay effects.
    mined = _verified_block_mining_occurrence(behavior, candidates)
    if mined is not None and not (
            behavior.get("event_binding") or behavior.get("event_interface")):
        return mined
    by_event = {row["event_id"]: row for row in candidates}
    if not by_event:
        raise HostEventBindingError(behavior, {
            "event_binding": "UNBOUND",
            "reason": "no_intrinsically_compatible_verified_occurrence",
        })
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
    # With exactly one inspected hook, there is no search or ranking problem.
    # Keep the established single-hook ABI (and one model decision) rather than
    # issuing a new global-index request that existing callers cannot fulfill.
    if len(by_event) == 1:
        only = next(iter(by_event))
        decision = _select(router, {**context, "verified_listeners": candidates},
                           "event_binding", [only], "select_verified_fabric_event")
        if decision.get("event_binding") != only:
            raise HostEventBindingError(behavior, decision)
        return by_event[only]

    # Choose from a compact, *global* index before grouping. The group-first
    # search loses the correct event when an earlier owner is misclassified.
    # Supplying just verified IDs and callback input types is much smaller than
    # repeating the full 26.2 functional-interface catalogue (>55 KB).
    compact = [
        {"event_id": row["event_id"], "inputs": row.get("parameters", ())}
        for row in sorted(candidates, key=lambda item: item["event_id"])
    ]
    direct = _select(
        router, {**context, "verified_hook_index": compact},
        "event_binding", [row["event_id"] for row in compact],
        "select_verified_fabric_event_index",
    )
    selected = direct.get("event_binding")
    if selected in by_event:
        return by_event[selected]
    if selected != "UNBOUND":
        raise HostEventBindingError(behavior, direct)
    # A model's UNBOUND assessment is not proof that the global index contains
    # no usable hook. Re-examine bounded event-owner groups with independent
    # names and preserve all rejection evidence rather than inventing an event.
    groups = defaultdict(list)
    for row in candidates:
        groups[row["event_owner"]].append(row)
    # A hierarchy of event owners is a search index, not proof that the first
    # owner contains an adequate hook. Earlier behavior stopped after choosing
    # ServerLifecycleEvents and then UNBOUND, although ServerTickEvents was in
    # the original inspected catalogue. Keep the complete candidate index and
    # search *other* owners when the inspected owner cannot observe the trigger.
    # Every selected hook must still come from the verified original inventory.
    rejected = []
    checked = set()
    attempts = min(5, len(groups))
    for attempt in range(attempts):
        remaining = {owner: rows for owner, rows in groups.items()
                     if owner not in checked}
        if not remaining:
            break
        if len(remaining) == 1:
            owner = next(iter(remaining))
        else:
            payload = {**context, "verified_event_groups": [
                {"event_owner": owner, "fields": [row["event_field"] for row in rows]}
                for owner, rows in sorted(remaining.items())],
                "rejected_event_owners": rejected}
            selection = _select(
                router, payload, "event_owner", sorted(remaining),
                "select_verified_fabric_event_owner" if attempt == 0
                else f"select_verified_fabric_event_owner_alternative_{attempt}")
            owner = selection.get("event_owner")
            if owner == "UNBOUND":
                # Declining a *search group* cannot prove there is no valid
                # callback inside the indexed signatures. Inspect a bounded
                # real owner deterministically; accept only a subsequently
                # selected exact event_id. This avoids falsely treating a
                # planner's first UNBOUND owner choice as a platform absence.
                def owner_rank(name):
                    rows = remaining[name]
                    general_context = (
                        "MinecraftServer", "ServerLevel", "ServerWorld",
                        "ServerPlayer", "PlayerEntity", "ServerPlayerEntity",
                    )
                    return (-sum(
                        any(str(kind).endswith(tag) for tag in general_context)
                        for row in rows for kind in row.get("parameters", ())
                    ), name)
                owner = min(remaining, key=owner_rank)
            elif owner not in remaining:
                raise HostEventBindingError(behavior, {
                    "owner_selection": selection, "rejected": rejected,
                })
        available = remaining[owner]
        checked.add(owner)
        selection = _select(
            router, {**context, "verified_listeners": available,
                     "rejected_event_owners": rejected},
            "event_binding", [row["event_id"] for row in available],
            "select_verified_fabric_event" if attempt == 0
            else f"select_verified_fabric_event_alternative_{attempt}")
        selected = selection.get("event_binding")
        if selected in {row["event_id"] for row in available}:
            return by_event[selected]
        if selected != "UNBOUND":
            raise HostEventBindingError(behavior, {
                "event_selection": selection, "rejected": rejected,
            })
        rejected.append({"event_owner": owner,
                         "reason": "No event within the selected owner can observe this trigger."})
    raise HostEventBindingError(behavior, {
        "event_binding": "UNBOUND", "checked_event_owners": rejected,
        "remaining_owner_count": len(groups) - len(checked),
    })


def _select(router, payload, field, choices, action):
    return router.generate_structured_data(
        "coder", [{"role": "system", "content": _INSTRUCTION},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
        data_name=action, parameters={"type": "object", "properties": {
            field: {"type": "string", "enum": ["UNBOUND", *choices]}},
            "required": [field], "additionalProperties": False})
