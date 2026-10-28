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


def _behavior_is_explicitly_supported(behavior: Mapping[str, Any], callbacks) -> bool:
    event = str(behavior.get("event_binding") or "").strip()
    if not event or any(behavior.get(name) for name in _STATE_OPERANDS):
        return False
    matching = [row for row in callbacks if row.get("event_id") == event]
    if len(matching) != 1:
        return False
    from importlib import import_module
    occurrence = import_module("minecraft_mod_ai.source_owned_hook")
    if not occurrence._event_candidates_for_occurrence(behavior, matching):
        return False
    return bool(str(behavior.get("trigger") or "").strip()
                and str(behavior.get("effects") or "").strip())


def _rule_ownership_available(
    spec, behaviors, *, retained, owned_runtime_records, source_ui_callbacks,
) -> bool:
    if not (spec.get("bindings") or {}).get("request_feature_entrypoint") or not behaviors:
        return False
    if source_ui_callbacks or owned_runtime_records:
        return False
    if any((retained or {}).get(key) for key in _RETAINED_OWNERS):
        return False
    if any(row.get(name) for row in behaviors for name in _STATE_OPERANDS):
        return False
    return all(row.get("source_status", "grounded") == "grounded"
               and row.get("trigger") and row.get("effects") for row in behaviors)


def checked_rule_template_ready(
    spec: Mapping[str, Any],
    behaviors: Sequence[Mapping[str, Any]],
    *,
    callbacks: Sequence[Mapping[str, Any]],
    retained: Mapping[str, Any] | None = None,
    owned_runtime_records: Sequence[Any] = (),
    source_ui_callbacks: Sequence[Any] = (),
) -> bool:
    """Accept only semantic rules with real, original-occurrence listeners."""
    return _rule_ownership_available(
        spec, behaviors, retained=retained,
        owned_runtime_records=owned_runtime_records,
        source_ui_callbacks=source_ui_callbacks,
    ) and all(_behavior_is_explicitly_supported(row, callbacks) for row in behaviors)


def resolve_inspected_template_behaviors(
    router: Any, spec: Mapping[str, Any], behaviors: Sequence[Mapping[str, Any]],
    *, callbacks: Sequence[Mapping[str, Any]],
    retained: Mapping[str, Any] | None = None,
    owned_runtime_records: Sequence[Any] = (),
    source_ui_callbacks: Sequence[Any] = (),
) -> list[dict[str, Any]] | None:
    """Bind grounded actions to real Fabric callback IDs before Java authoring.

    A model chooses only from actual inspected listener identifiers, never
    invents Java code, event signatures, or a missing state producer. The
    selector may explicitly decline when no hook matches the original event.
    """
    if not behaviors or any(row.get("source_status") != "grounded" for row in behaviors):
        return None
    # Do not spend model calls on work the rule backend cannot own.
    if not _rule_ownership_available(
        spec, behaviors, retained=retained,
        owned_runtime_records=owned_runtime_records,
        source_ui_callbacks=source_ui_callbacks,
    ):
        return None
    from .source_occurrence_error import SourceHookUnavailable
    from importlib import import_module
    choose_source_hook = import_module("minecraft_mod_ai.source_owned_hook").choose_source_hook
    selected = []
    for behavior in behaviors:
        if behavior.get("event_binding"):
            selected.append(dict(behavior))
            continue
        try:
            callback = choose_source_hook(router, behavior, callbacks)
        except SourceHookUnavailable:
            return None
        selected.append({**behavior, "event_binding": callback["event_id"]})
    return selected if checked_rule_template_ready(
        spec, selected, callbacks=callbacks, retained=retained,
        owned_runtime_records=owned_runtime_records,
        source_ui_callbacks=source_ui_callbacks,
    ) else None


def render_checked_rule(*args: Any, **kwargs: Any) -> str:
    """Resolve the optional rule backend after both compiler modules load.

    A deferred backend boundary avoids a circular module import at import time;
    the selected rule is still validated against actual checked Java sources.
    """
    from importlib import import_module
    backend = import_module("minecraft_mod_ai.host_rule_feature")
    return backend.compose_rule_feature(*args, **kwargs)


__all__ = ["checked_rule_template_ready", "resolve_inspected_template_behaviors", "render_checked_rule"]
