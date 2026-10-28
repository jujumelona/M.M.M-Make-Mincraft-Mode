"""Seed exact host UI registrations and queue their source implementation first."""
from __future__ import annotations

import json
from copy import deepcopy

from .incremental_java_model import method_metadata
from .source_ui_actions import source_ui_callback


def _validated_callback(callback, *, package, parent):
    if not isinstance(callback, dict) or not isinstance(callback.get("source_ui_record"), dict):
        raise ValueError("SOURCE_UI_CALLBACK_RECORD_REQUIRED")
    record = callback["source_ui_record"]
    expected = {**source_ui_callback(record, package=package), "source_ui_record": record}
    if not parent or expected["source_requirement_id"] != parent:
        raise ValueError("SOURCE_UI_CALLBACK_SOURCE_MISMATCH")
    if "parent_requirement_id" in callback:
        expected["parent_requirement_id"] = parent
    if callback != expected:
        raise ValueError("SOURCE_UI_CALLBACK_ABI_MISMATCH")
    return deepcopy(callback)


def _callbacks(builder):
    bindings = builder.spec.get("bindings") or {}
    raw = bindings.get("source_ui_callbacks_json")
    if raw is None:
        return []
    try:
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError("expected a callback list")
        callbacks = [_validated_callback(row, package=bindings.get("source_ui_package_name", builder.program["package"]),
                                        parent=bindings.get("parent_requirement_id")) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SOURCE_UI_CALLBACK_INVALID: " + str(exc)) from exc
    if len({row["action_id"] for row in callbacks}) != len(callbacks):
        raise ValueError("SOURCE_UI_CALLBACK_DUPLICATE")
    return callbacks


def _seed_callback(builder, registrar, callback):
    parameters = [{"type": typ, "name": name} for typ, name in zip(
        callback["parameters"], callback["parameter_names"], strict=True)]
    metadata = method_metadata(callback["return_type"], parameters)
    listener = builder.new_method(owner=registrar["owner"], metadata=metadata,
        purpose="Implement this original UI action's conditions and effects from its exact input mapping: "
                + str(callback["trigger"]), behavior=deepcopy(registrar["behavior"]), reuse=False)
    listener["event"] = callback
    node = {"id": builder.allocate("n"), "op": "register", "event_owner": callback["event_owner"],
            "event_field": callback["event_field"], "handler": listener["name"]}
    registrar.setdefault("body", []).append(node)
    builder.history[node["id"]] = node
    return listener


def seed_source_ui_callbacks(builder, *, initial=True) -> None:
    """Bind source-owned actions without asking the model to choose an event."""
    if not initial:
        return
    callbacks = _callbacks(builder)
    if not callbacks:
        return
    registrars = [row for row in builder.program["members"] if row.get("registrar")]
    if not registrars:
        raise ValueError("SOURCE_UI_CALLBACK_REGISTRAR_REQUIRED")
    previous = builder.last_declaration
    listeners = [_seed_callback(builder, registrars[0], callback) for callback in callbacks]
    builder.last_declaration = previous
    for listener in reversed(listeners):
        builder.pending.remove(listener)
        builder.pending.appendleft(listener)
