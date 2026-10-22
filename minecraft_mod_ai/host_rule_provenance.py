"""Require evidence for runtime-state reads before invoking the small Java coder.

A resolved state-owner *label* is not a value producer. In particular,
world_state + host_classified_unbound_read means planning retained a real
obligation without finding a Java observer or materialized content producer.
Neither a valid Fabric event nor an unrelated world boolean discharges it.
"""
from __future__ import annotations

import json


class RuleMissingRuntimeProducer(ValueError):
    def __init__(self, behavior_id, missing):
        self.behavior_id = behavior_id
        self.missing = tuple(missing)
        details = {"behavior_id": behavior_id, "missing_runtime_producers": missing,
                   "required_action": "materialize a producing content/source module or bind an inspected runtime observer before compiling this behavior"}
        super().__init__("RULE_RUNTIME_PRODUCER_REQUIRED: " + json.dumps(
            details, ensure_ascii=False, sort_keys=True))


def validate_runtime_read_producers(behavior, runtime_bindings, symbols):
    """Reject planning-only observations; accept only checked exact source IDs.

    This is a *necessary*, not sufficient, condition for semantic coverage.
    Verified symbol existence does not prove the implementation's game effect.
    """
    missing = []
    resolved = {}
    for name in dict.fromkeys(behavior.get("runtime_state_reads", ())):
        binding = runtime_bindings.get(name)
        if not isinstance(binding, dict) or binding.get("binding") != "world_state":
            missing.append({"name": name, "reason": "missing_world_observer_binding"})
            continue
        ref = binding.get("verified_symbol_ref")
        evidence = symbols.get(ref) if isinstance(ref, str) else None
        if (not isinstance(evidence, dict)
                or evidence.get("kind") not in {"METHOD", "FIELD"}
                or evidence.get("return_type") in {None, "void"}
                or (evidence.get("kind") == "METHOD" and evidence.get("parameters"))
                or binding.get("semantic_status") != "host_verified_observer"):
            missing.append({
                "name": name,
                "reason": ("planning_only_world_state" if not ref
                           else "unverified_or_unusable_java_observer"),
                "binding_status": binding.get("semantic_status"),
            })
            continue
        resolved[name] = ref
    if missing:
        raise RuleMissingRuntimeProducer(str(behavior.get("behavior_id") or ""), missing)
    return resolved
