"""Conservative, effect-preserving routing to the host-owned rule compiler.

Only an inspected, explicitly-bound callback is eligible for template-first
semantic rules. We do not infer a gameplay trigger from a nearby tick/join
event, promote an unverified getter to a producer, or claim source mutations
are implementable by read-only rule projections.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_RETAINED_OWNERS = ("functions", "event_bindings", "ui_owned_records", "source_ui_actions")
_STATE_OPERANDS = ("runtime_state_reads", "runtime_state_writes", "state_reads", "state_writes")


def _behavior_is_explicitly_supported(behavior: Mapping[str, Any], event_ids: set[str]) -> bool:
    event = str(behavior.get("event_binding") or "").strip()
    if not event or event not in event_ids:
        return False
    if any(behavior.get(name) for name in _STATE_OPERANDS):
        return False
    return bool(str(behavior.get("trigger") or "").strip()
                and str(behavior.get("effects") or "").strip())


def checked_rule_template_ready(
    spec: Mapping[str, Any],
    behaviors: Sequence[Mapping[str, Any]],
    *,
    callbacks: Sequence[Mapping[str, Any]],
    retained: Mapping[str, Any] | None = None,
    owned_runtime_records: Sequence[Any] = (),
    source_ui_callbacks: Sequence[Any] = (),
) -> bool:
    """Admit only behavior-preserving templates with inspected native events.

    When any behavior has an unowned runtime producer or overlapping action,
    retain the original source author. Never substitute an unrelated trigger.
    """
    if not (spec.get("bindings") or {}).get("request_feature_entrypoint") or not behaviors:
        return False
    if source_ui_callbacks or owned_runtime_records:
        return False
    if any((retained or {}).get(key) for key in _RETAINED_OWNERS):
        return False
    event_ids = {
        str(row.get("event_id") or "")
        for row in callbacks if isinstance(row, Mapping)
    }
    return all(_behavior_is_explicitly_supported(row, event_ids) for row in behaviors)


def render_checked_rule(*args: Any, **kwargs: Any) -> str:
    """Resolve the optional rule backend after both compiler modules load.

    A deferred backend boundary avoids a circular module import at import time;
    the selected rule is still validated against actual checked Java sources.
    """
    from importlib import import_module
    backend = import_module("minecraft_mod_ai.host_rule_feature")
    return backend.compose_rule_feature(*args, **kwargs)


__all__ = ["checked_rule_template_ready", "render_checked_rule"]
