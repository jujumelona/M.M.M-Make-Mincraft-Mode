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

    event_ids = {str(row.get("event_id") or "") for row in callbacks}
    planner = RuntimeSourcePlanner(symbols)
    obligations: list[dict[str, Any]] = []
    grounded: list[str] = []
    for behavior in behaviors:
        behavior_id = str(behavior.get("behavior_id") or "")
        if not behavior_id:
            raise ValueError("FEATURE_EXECUTION_BEHAVIOR_ID_REQUIRED")
        explicit = str(behavior.get("event_binding") or "")
        if explicit and explicit not in event_ids:
            raise ValueError("FEATURE_EXECUTION_EXPLICIT_HOOK_NOT_INSPECTED: " + explicit)
        if not callbacks:
            obligations.append({"behavior_id": behavior_id,
                                "reason": "no_inspected_callback",
                                "required_action": "implement and register an actual callback/mixin hook"})
            continue
        try:
            eligible, _ = bind_runtime_sources(behavior, runtime_bindings, planner, callbacks)
        except RuleMissingRuntimeProducer as error:
            for missing in error.missing:
                obligations.append({"behavior_id": behavior_id, **dict(missing)})
            continue
        if explicit and not any(row.get("event_id") == explicit for row in eligible):
            obligations.append({"behavior_id": behavior_id,
                                "reason": "explicit_event_cannot_supply_runtime_reads",
                                "required_action": "implement a producer on the actual event owner"})
        elif not explicit:
            # An observed getter is not proof of which Fabric occurrence
            # owns this behavior. Avoid an unbounded native-hook pager.
            obligations.append({"behavior_id": behavior_id,
                                "reason": "occurrence_hook_not_owned",
                                "required_action": "bind a checked event or implement an owned hook"})
        else:
            grounded.append(behavior_id)
    return {
        "route": "source_owned_java" if obligations else "typed_rule",
        "verified_behavior_ids": grounded,
        "producer_obligations": obligations,
    }
