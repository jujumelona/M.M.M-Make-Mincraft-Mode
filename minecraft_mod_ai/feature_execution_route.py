"""Capability-first execution ownership for authored Fabric behaviors.

Choose an implementation *before* rule model calls. An authored gameplay
concept is not a runtime producer merely because a planner labeled it
world_state. This module never promotes a guessed getter or callback to proof.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def select_feature_execution_route(
    behaviors: Sequence[Mapping[str, Any]],
    runtime_bindings: Mapping[str, Any],
    symbols: Mapping[str, Any],
    callbacks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return typed_rule only for completely grounded input/event contracts.

    All other features belong to the source author, which must implement real
    event/producer Java and pass target compilation. In particular, a free-form
    world-state label is never accepted as an inspected Java producer.
    """
    if not behaviors:
        raise ValueError("FEATURE_EXECUTION_BEHAVIORS_REQUIRED")
    from .host_rule_source_plan import RuntimeSourcePlanner, bind_runtime_sources
    from .host_rule_provenance import RuleMissingRuntimeProducer
    from .source_owned_hook import _event_candidates_for_occurrence

    event_ids = {str(row.get("event_id") or "") for row in callbacks}
    planner = RuntimeSourcePlanner(symbols)
    obligations: list[dict[str, Any]] = []
    grounded: list[str] = []
    for behavior in behaviors:
        behavior_id = str(behavior.get("behavior_id") or "")
        if not behavior_id:
            raise ValueError("FEATURE_EXECUTION_BEHAVIOR_ID_REQUIRED")
        # A native runtime mutation is not a read-only observation.  Typed
        # Rule currently proves runtime inputs but does not prove that each
        # requested world/entity/item mutation was emitted. Route the whole
        # behavior to the host-owned mutation compiler instead of silently
        # accepting a rule that merely logs or reads the target.
        for state_name in dict.fromkeys(behavior.get("runtime_state_writes") or ()):
            if not isinstance(state_name, str) or not state_name.strip():
                raise ValueError("FEATURE_EXECUTION_RUNTIME_WRITE_NAME_INVALID")
            obligations.append({
                "behavior_id": behavior_id,
                "name": state_name,
                "reason": "runtime_mutation_requires_verified_effect",
                "required_action": (
                    "bind and execute a checked mutating method of the actual "
                    "Minecraft runtime owner for this state; a host scalar, "
                    "getter, local assignment or log is not a substitute"
                ),
            })
        explicit = str(behavior.get("event_binding") or "")
        if explicit and explicit not in event_ids:
            raise ValueError("FEATURE_EXECUTION_EXPLICIT_HOOK_NOT_INSPECTED: " + explicit)
        if not callbacks:
            obligations.append({"behavior_id": behavior_id,
                                "reason": "no_inspected_callback",
                                "required_action": "implement and register an actual callback/mixin hook"})
            continue
        # A real callback that exposes a player or world is not automatically
        # the occurrence of the requested action. In the captured production
        # failure, a damage event supplied a world object but never meant
        # 'player arrives at a new planet'. Apply the same original-trigger
        # admissibility to Typed Rule and source-owned Java routing.
        occurrence_callbacks = _event_candidates_for_occurrence(
            behavior, callbacks,
        )
        if explicit and explicit not in {
            str(event.get("event_id") or "") for event in occurrence_callbacks
        }:
            raise ValueError(
                "FEATURE_EXECUTION_EXPLICIT_HOOK_SEMANTIC_MISMATCH: " + explicit
            )
        if not occurrence_callbacks:
            obligations.append({
                "behavior_id": behavior_id,
                "reason": "occurrence_hook_not_owned",
                "required_action": (
                    "implement a source-owned occurrence from materialized "
                    "player interaction/world lifecycle source, not a "
                    "semantically unrelated Fabric callback"
                ),
            })
            continue
        try:
            eligible, _ = bind_runtime_sources(
                behavior, runtime_bindings, planner, occurrence_callbacks,
            )
        except RuleMissingRuntimeProducer as error:
            for missing in error.missing:
                obligations.append({"behavior_id": behavior_id, **dict(missing)})
            continue
        if explicit and not any(row.get("event_id") == explicit for row in eligible):
            obligations.append({"behavior_id": behavior_id,
                                "reason": "explicit_event_cannot_supply_runtime_reads",
                                "required_action": "implement a producer on the actual event owner"})
        elif not explicit and not str(behavior.get("trigger") or "").strip():
            # No original observable occurrence: do not manufacture a hook.
            obligations.append({"behavior_id": behavior_id,
                                "reason": "occurrence_hook_not_owned",
                                "required_action": "bind a checked event or implement an owned hook"})
        elif not eligible:
            obligations.append({"behavior_id": behavior_id,
                                "reason": "no_compatible_runtime_occurrence",
                                "required_action": "materialize a callback with real receiver inputs"})
        else:
            # This is eligibility to AUTHOR and verify a rule, not proof the
            # rule has implemented the mechanic. RuleProgram still composes a
            # typed predicate/effect using a checked event and rejects an
            # unusable semantic runtime source. No source Java fallback.
            grounded.append(behavior_id)
    return {
        "route": "source_owned_java" if obligations else "typed_rule",
        "verified_behavior_ids": grounded,
        "producer_obligations": obligations,
    }
