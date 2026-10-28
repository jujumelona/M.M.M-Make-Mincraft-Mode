"""Conservative, effect-preserving routing to the host-owned rule compiler.

Only an inspected, explicitly-bound callback is eligible for template-first
semantic rules. We do not infer a gameplay trigger from a nearby tick/join
event, promote an unverified getter to a producer, or claim source mutations
are implementable by read-only rule projections.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def checked_rule_template_ready(
    spec: Mapping[str, Any],
    behaviors: Sequence[Mapping[str, Any]],
    *,
    callbacks: Sequence[Mapping[str, Any]],
    retained: Mapping[str, Any] | None = None,
    owned_runtime_records: Sequence[Any] = (),
    source_ui_callbacks: Sequence[Any] = (),
) -> bool:
    """Proof gate, not heuristic classification.

    A template-first route never silently drops an original feature. Novel
    runtime mutation, implicit occurrence, or an already-owned interaction
    must be implemented by a separate source owner instead.
    """
    if not (spec.get("bindings") or {}).get("request_feature_entrypoint"):
        return False
    if not behaviors or source_ui_callbacks or owned_runtime_records:
        return False
    retained = retained or {}
    if any(retained.get(key) for key in ("functions", "event_bindings", "ui_owned_records", "source_ui_actions")):
        return False
    event_ids = {
        str(row.get("event_id") or "")
        for row in callbacks if isinstance(row, Mapping)
    }
    for behavior in behaviors:
        event_id = str(behavior.get("event_binding") or "").strip()
        if not event_id or event_id not in event_ids:
            return False
        if any(behavior.get(key) for key in (
            "runtime_state_reads", "runtime_state_writes", "state_reads", "state_writes"
        )):
            # State requires a proved storage and producer owner. This path
            # cannot infer it from a worksheet label.
            return False
        if not str(behavior.get("trigger") or "").strip():
            return False
        if not str(behavior.get("effects") or "").strip():
            return False
    return True


__all__ = ["checked_rule_template_ready"]
