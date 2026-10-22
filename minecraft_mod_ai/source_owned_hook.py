"""One bounded decision for the *runtime observation hook* of a source feature.

The source path must not assume an unmatched event is real. Unlike rule
lowering, this decision does not claim that a state getter already exists:
the callback only supplies typed runtime receivers for subsequent observer
and effect generation. Real compilation and runtime verification still follow.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .fixed_template_generation import _validate_native_arguments


class SourceHookUnavailable(ValueError):
    def __init__(self, behavior_id: str, *, reason: str, inspected: int):
        super().__init__("SOURCE_HOOK_PRODUCER_REQUIRED: " + json.dumps({
            "behavior_id": behavior_id, "reason": reason,
            "inspected_callback_count": inspected,
            "required_action": "materialize a custom event/mixin source owner "
                               "or bind a real observable Fabric callback before effects",
        }, ensure_ascii=False))


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
        return by_id[explicitly_bound]
    if not by_id:
        raise SourceHookUnavailable(behavior_id, reason="no_inspected_callback", inspected=0)
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
