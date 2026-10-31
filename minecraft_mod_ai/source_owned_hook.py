"""One bounded decision for the *runtime observation hook* of a source feature.

The source path must not assume an unmatched event is real. Unlike rule
lowering, this decision does not claim that a state getter already exists:
the callback only supplies typed runtime receivers for subsequent observer
and effect generation. Real compilation and runtime verification still follow.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .fixed_template_generation import _validate_native_arguments
from .source_occurrence_error import SourceHookUnavailable


def _event_candidates_for_occurrence(behavior, candidates):
    """Narrow only unambiguous Minecraft occurrence classes.

    A callback being a *real* Fabric event does not make it relevant:
    AFTER_DAMAGE can never report that a player changed dimensions. The
    semantic vocabulary used here covers observable lifecycle categories,
    not arbitrary feature or theme names. Unknown occurrences deliberately
    retain the complete inspected ABI rather than guessing.
    """
    # The source-owned production writer must enforce the SAME inspected
    # occurrence semantics as the host event binder. Previously these paths
    # diverged: a model chose AFTER_KILLED_OTHER_ENTITY for an ongoing fight,
    # and later attempted to fabricate damage, health and durability writes
    # after the victim was already dead.
    from .host_behavior_event_binding import (
        _occurrence_admissible, _verified_block_mining_occurrence,
    )

    candidates = [row for row in candidates if _occurrence_admissible(behavior, row)]
    mined = _verified_block_mining_occurrence(behavior, candidates)
    if mined is not None and not (
            behavior.get("event_binding") or behavior.get("event_interface")):
        return [mined]
    trigger = str(behavior.get("trigger") or "").casefold()
    transition = bool(re.search(
        r"\b(?:reach(?:es|ed)?|arriv(?:es|ed|ing)?|enter(?:s|ed|ing)?|"
        r"travel(?:s|ed|ing)?|chang(?:es|ed|ing)?|switch(?:es|ed|ing)?)\b"
        r".{0,96}\b(?:planet|dimension|world|level|location)\b",
        trigger,
    ))
    if not transition:
        return list(candidates)
    relevant = []
    for row in candidates:
        event = str(row.get("event_id") or "").casefold().replace("_", "")
        parameters = tuple(str(p) for p in row.get("parameters", ()) or ())
        has_player = any(
            p.endswith(("ServerPlayer", "ServerPlayerEntity"))
            for p in parameters
        )
        if (has_player and ("afterplayerchangelevel" in event
                            or "afterplayerchangeworld" in event
                            or "playerchangedimension" in event)):
            relevant.append(row)
    # No matching event is evidence that we need an original occurrence
    # source, not permission to use a damage/respawn/sleep callback.
    return relevant


def choose_source_hook(
    router: Any, behavior: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    behavior_id = str(behavior.get("behavior_id") or "")
    if not behavior_id:
        raise ValueError("SOURCE_HOOK_BEHAVIOR_REQUIRED")
    by_id = {str(row["event_id"]): dict(row) for row in candidates}
    explicitly_bound = str(behavior.get("event_binding") or "")
    if explicitly_bound:
        if explicitly_bound not in by_id:
            raise ValueError("SOURCE_HOOK_NOT_IN_INSPECTED_ABI: " + explicitly_bound)
        relevant = _event_candidates_for_occurrence(behavior, [by_id[explicitly_bound]])
        if not relevant:
            raise SourceHookUnavailable(
                behavior_id, reason="explicit_hook_cannot_observe_original_occurrence",
                inspected=1,
            )
        return by_id[explicitly_bound]
    if not by_id:
        raise SourceHookUnavailable(behavior_id, reason="no_inspected_callback", inspected=0)
    relevant = _event_candidates_for_occurrence(behavior, list(by_id.values()))
    if not relevant:
        raise SourceHookUnavailable(
            behavior_id, reason="no_matching_original_occurrence_callback",
            inspected=len(by_id),
        )
    by_id = {str(row["event_id"]): dict(row) for row in relevant}
    # A model can decline inspected hooks, but it cannot invent an event owner.
    # Keep the event index compact: no 872K-symbol classpath in the 9B prompt.
    ordered = sorted(by_id.values(), key=lambda e: e["event_id"])
    options = [row["event_id"] for row in ordered]
    schema = {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "enum": [*options, "SOURCE_REQUIRED"]},
        },
        "required": ["event_id"], "additionalProperties": False,
    }
    payload = {
        "behavior": {key: behavior.get(key) for key in
                     ("behavior_id", "source_quote", "intent", "trigger", "effects",
                      "runtime_state_reads", "runtime_state_writes")},
        "inspected_fabric_callbacks": [
            {key: row.get(key) for key in
             ("event_id", "event_interface", "method", "parameters", "return_type")}
            for row in ordered],
    }
    response = router.generate_structured_data(
        "planner", [
            {"role": "system", "content": (
                "Choose exactly one REAL inspected Fabric callback that exposes "
                "the runtime receivers needed to detect the ORIGINAL occurrence. "
                "A recurring server tick is valid only if its actual input types "
                "allow the condition to be checked without inventing gameplay "
                "state. Selecting an event does not prove a missing world/UI "
                "observation exists. If none can observe the original condition, "
                "return SOURCE_REQUIRED: it must be implemented as a custom "
                "event/mixin producer in another source unit. Never replace an "
                "input with a fabricated key, command, or player join event."
            )},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], data_name="select_source_owned_observation_hook", parameters=schema,
    )
    chosen = _validate_native_arguments(response, schema)["event_id"]
    if chosen == "SOURCE_REQUIRED":
        raise SourceHookUnavailable(
            behavior_id, reason="custom_runtime_hook_needed",
            inspected=len(by_id))
    return by_id[chosen]
