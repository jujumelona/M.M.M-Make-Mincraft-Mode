"""Stage-specific Java inference views; complete source/index data stays host-side."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

from .java_authoring_api import target_interfaces


_SEMANTIC = (
    "host_java_authoring", "target_path", "output_path", "package", "class",
    "public_class", "side", "target", "requirement", "original_feature_goal",
    "implementation_requirements", "assigned_behavior", "assigned_behaviors",
    "host_entrypoint", "class_declaration", "current_unit", "current_unit_purpose",
    "all_units", "previous_declaration_metadata", "current_field", "completed_plan_fields",
    "field", "index", "count", "previous_entries", "owned_runtime_records",
    "host_owned_members",
)
_BODY = _SEMANTIC + (
    "member_role", "verified_event_callback", "handler_signature", "owner_type", "owner",
    "declaration_metadata", "host_signature", "declared_member_api", "callback_input_api",
    "runtime_state_observations", "runtime_state_bindings", "owned_runtime_records",
    "observed_source_name", "observed_source_task_id", "source_owned_work",
    "immutable_signature", "current_member", "javac_diagnostic", "diagnostic", "error",
    "previous_implementation", "previous_response", "rejected_member", "correction_obligation",
    "host_declared_neighbor_signatures",
)
_ACCESSOR = re.compile(r"\b(?:readState_|writeState_)([A-Za-z_$][\w$]*)\s*\(")


def _views(messages):
    decoded, other = [], []
    for message in messages:
        try:
            value = json.loads(message.get("content", "")) if message.get("role") == "user" else None
        except (ValueError, TypeError):
            value = None
        if isinstance(value, dict):
            decoded.append(value)
        else:
            other.append(dict(message))
    return decoded, other


def _merge(views):
    combined = {}
    for view in views:
        combined.update(view)
    return combined


def _state_names(context):
    selected = context.get("assigned_behavior")
    behaviors = [selected] if isinstance(selected, Mapping) else context.get("assigned_behaviors", ())
    return {str(name) for row in behaviors if isinstance(row, Mapping)
            for key in ("state_reads", "state_writes", "runtime_state_reads", "runtime_state_writes")
            for name in row.get(key) or ()}


def _state_interfaces(records, names):
    answer = []
    for record in records or ():
        methods = [dict(row) for row in record.get("methods", ()) if row.get("name") in {
            prefix + name for name in names for prefix in ("readState_", "writeState_")}]
        if methods:
            answer.append({"owner": record["owner"], "methods": methods})
    return answer


def _prerequisite_interfaces(records, names):
    answer = []
    for record in records or ():
        types = []
        for original in record.get("types", ()):
            members = []
            for header in original.get("members", ()):
                accessor = _ACCESSOR.search(str(header))
                if accessor is None or accessor[1] in names:
                    members.append(header)
            types.append({"owner": original.get("owner"),
                          "declaration": original.get("declaration"), "members": members})
        answer.append({key: record[key] for key in ("path", "imports") if key in record} | {"types": types})
    return answer


def _event_interface(row):
    event = {key: row[key] for key in ("event_id", "event_owner", "event_field", "event_interface", "method",
                                      "parameters", "return_type", "type_parameters", "throws")
             if key in row and (row[key] != [] or key == "parameters")}
    if row.get("event_id") == f'{row.get("event_owner")}#{row.get("event_field")}':
        event.pop("event_owner", None)
        event.pop("event_field", None)
    return event


def _callbacks(records):
    return [_event_interface(row) for row in records or ()]


def _member_callbacks(records, context, frame):
    if frame["kind"] != "field":
        return _callbacks(records)
    # Listener-valued fields still need their exact SAM. Other initializers do
    # not choose the feature's registration event: that is a method's work.
    # A constructor/factory argument can require a SAM absent from the field's
    # own type (new Handler(listener)). Keep callback interfaces referenced by
    # any already-retained callable ABI as well as by the field itself.
    references = [frame["prefix"], *(context.get(key) for key in (
        "declaration_metadata", "declared_member_api", "inspected_target_api", "inspected_api",
        "materialized_prerequisite_api", "callback_input_api"))]
    types = set(re.findall(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", json.dumps(references)))
    return _callbacks(row for row in records or ()
                      if row.get("event_interface") in types
                      or str(row.get("event_interface") or "").rsplit(".", 1)[-1] in types)


def _retained_interfaces(record):
    # These functions are already compiled upstream. Their bodies are not
    # coding work for this member, but their signatures and UI ownership are.
    if not isinstance(record, Mapping):
        return record
    fields = ("id", "covers", "name", "return_type", "parameters", "params", "type_parameters", "scope")
    answer = dict(record)
    if "functions" in record:
        answer["functions"] = [{key: row[key] for key in fields if key in row}
                               for row in record.get("functions", ())]
    return answer


def _work_scope(value, names):
    return {key: [item for item in items if item in names] if isinstance(items, list) else items
            for key, items in (value or {}).items()}


def _messages(other, task):
    return [*other, {"role": "user", "content": json.dumps(
        task, ensure_ascii=False, separators=(",", ":"), default=str)}]


def _runtime_binding_view(context, names):
    if "runtime_state_bindings" not in context:
        return {}
    return {"runtime_state_bindings": {key: value for key, value in (context.get("runtime_state_bindings") or {}).items()
            if key in names or isinstance(value, Mapping) and value.get("binding") == "typed_input"}}


def _member_instructions(other, kind, focus):
    from .direct_feature_source import member_implementation_instruction
    instruction = {"role": "system", "content": member_implementation_instruction(True, kind)}
    if focus:
        return [instruction]
    if kind == "field":
        previous = {member_implementation_instruction(True), instruction["content"]}
        return [message for message in other if message.get("content") not in previous] + [instruction]
    return other


def _body_messages(other, task, focus):
    return _messages(_member_instructions(other, task["immutable_signature"]["kind"], focus), task)


def java_data_messages(messages, action):
    """Counts/layout see intent; declaration metadata sees types, never source bodies."""
    views, other = _views(messages)
    if not any(view.get("host_java_authoring") for view in views):
        return messages
    context = _merge(views)
    task = {key: context[key] for key in _SEMANTIC if key in context}
    if "retained_typed_behavior" in context:
        task["retained_typed_behavior"] = _retained_interfaces(context["retained_typed_behavior"])
    if not action.startswith("submit_java_construction_plan"):
        names = _state_names(context)
        task["host_state_access"] = _state_interfaces(context.get("host_state_access"), names)
        task["materialized_prerequisite_api"] = _prerequisite_interfaces(
            context.get("materialized_prerequisite_api"), names)
        if "available_fabric_callbacks" in context:
            task["available_fabric_callbacks"] = _callbacks(context["available_fabric_callbacks"])
        for key in ("inspected_target_api", "inspected_api"):
            if isinstance(context.get(key), Mapping):
                task[key] = target_interfaces(context[key])
    return _messages(other, task)


def java_body_messages(messages, frame, *, focus=False):
    """Keep exact callable interfaces and current semantics without duplicate inventories."""
    views, other = _views(messages)
    if not any(view.get("host_java_authoring") for view in views):
        return messages
    context = _merge(views)
    names = _state_names(context)
    task = {key: context[key] for key in _BODY if key in context}
    task.update(_runtime_binding_view(context, names))
    task["immutable_signature"] = {key: frame[key] for key in ("kind", "prefix", "suffix")}
    task["host_state_access"] = _state_interfaces(
        context.get("host_state_access") or frame.get("host_state_access"), names)
    task["host_state_work_order_scope"] = _work_scope(context.get("host_state_work_order_scope"), names)
    task["materialized_prerequisite_api"] = _prerequisite_interfaces(
        context.get("materialized_prerequisite_api"), names)
    for key in ("inspected_target_api", "inspected_api"):
        if isinstance(context.get(key), Mapping):
            task[key] = target_interfaces(context[key])
    if "available_fabric_callbacks" in context:
        task["available_fabric_callbacks"] = _member_callbacks(context["available_fabric_callbacks"], task, frame)
    if "retained_typed_behavior" in context:
        task["retained_typed_behavior"] = _retained_interfaces(context["retained_typed_behavior"])
    return _body_messages(other, task, focus)


def java_implementation_from_response(response):
    """Keep the exact decoded body, or the original rejected transport bytes."""
    try:
        value = json.loads(response)
        previous = value.get("implementation") if isinstance(value, dict) else None
    except (TypeError, ValueError):
        previous = None
    return previous if isinstance(previous, str) else response


def java_correction_messages(messages, *, frame, response, diagnostic):
    """Correct direct authored members with the same bounded interface view."""
    views, _ = _views(messages)
    if not any(_supports_direct_correction(view) for view in views):
        return None
    task = _merge(views)
    task.update({
        "diagnostic": diagnostic, "previous_response": response,
        "previous_implementation": java_implementation_from_response(response),
        "correction_obligation": (
            "Change the invalid Java expression or statement to resolve the diagnostic. "
            "Preserve the requested behavior, state and existing UI ownership. "
            "Use the exact inspected interfaces and immutable signature. "
            "Return exactly one JSON string field named implementation."
        ),
    })
    return java_body_messages(_messages([{"role": "system", "content": (
        "Fix only this Java member body using the parser or state ABI diagnostic. "
        "The host owns its signature and delimiters. Supply a type where Java "
        "requires a type; do not put a call expression in a type position. "
        "Preserve all required effects. Do not fabricate state accessors, use "
        "unrelated state keys, or replace missing persistent state with defaults. "
        "No imports, new classes, TODOs or prose."
    )}], task), frame)


def _supports_direct_correction(view):
    return any((view.get("host_java_authoring"), view.get("host_incremental_java")))
