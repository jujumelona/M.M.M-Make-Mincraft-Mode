"""Repair authored feature members from compiler errors, without proof routing."""
from __future__ import annotations

import hashlib
import json
import re

from .java_member_diagnostics import (
    _failing_member, _first_compiler_error, _host_declared_neighbor_signatures,
)
from .first_pass_java_declarations import validate_source_frame
from .host_java_imports import host_add_verified_imports
from .java_region_parser import _parse


def _field_initializer(source, diagnostic, filename):
    line = re.search(re.escape(filename) + r":(\d+):", diagnostic)
    if line is None:
        raise ValueError("JAVA_HOST_REPAIR_NO_SOURCE_LINE")
    number = int(line[1])
    raw, root = _parse(source)
    pending = [root]
    while pending:
        node = pending.pop()
        if not node.start_point.row + 1 <= number <= node.end_point.row + 1:
            continue
        if node.type == "field_declaration":
            variables = [n for n in node.named_children if n.type == "variable_declarator"]
            value = variables[0].child_by_field_name("value") if len(variables) == 1 else None
            if value is None:
                break
            frame = {"kind": "field",
                     "prefix": raw[node.start_byte:value.start_byte].decode(),
                     "suffix": raw[value.end_byte:node.end_byte].decode()}
            return raw, node, raw[node.start_byte:node.end_byte].decode(), frame
        pending.extend(reversed(node.named_children))
    raise ValueError("JAVA_HOST_REPAIR_NOT_A_MEMBER_DIAGNOSTIC")


def _repair_payload(context, source, original, frame, diagnostic):
    implementation = context.get("implementation_context") or {}
    return {
        **{key: implementation.get(key) for key in (
            "original_feature_goal", "implementation_requirements",
            "retained_typed_behavior", "owned_runtime_records")},
        "assigned_behaviors": list(context.get("behaviors") or ()),
        "runtime_state_bindings": dict(context.get("runtime_state_bindings") or {}),
        "target_path": context["target_path"], "javac_diagnostic": diagnostic,
        "current_member": original, "response_source_frame": frame,
        "host_declared_neighbor_signatures": _host_declared_neighbor_signatures(
            source, selected_name=original.split("{", 1)[0].strip()),
        "inspected_api": context.get("api_evidence") or {},
        "materialized_prerequisite_api": list(context.get("prerequisite_api") or ()),
        "host_state_access": list(context.get("state_access") or ()),
    }


def _write_repair(router, payload, frame, original, attempt):
    instruction = (
        "Fix only this authored Java member using the concrete compiler diagnostic. "
        "Return one JSON implementation string: only the method body or field "
        "initializer expression. The host preserves its signature and every other "
        "declaration. Preserve the original assigned behaviors, existing UI action "
        "ownership, shared state APIs and all required effects. Implement missing "
        "logic against the declared members and actual target APIs; do not search "
        "for a pre-existing gameplay implementation or request proof records. "
        "No package/import directives, additional declarations or placeholders."
    )
    fingerprint = hashlib.sha256((original + payload["javac_diagnostic"]).encode()).hexdigest()[:16]
    for retry in range(2):
        task = dict(payload)
        if retry:
            task["rejected_member"] = original
            task["correction_obligation"] = "The prior response made no change; fix the displayed compiler error."
        revised = router.generate_java_source(
            "coder", [{"role": "system", "content": instruction},
                      {"role": "user", "content": json.dumps(task, ensure_ascii=False, default=str)}],
            data_name=f"repair_direct_feature_member_{attempt}_{retry}_{fingerprint}",
            source_frame=frame)
        validate_source_frame(revised, frame)
        if revised.strip() != original.strip():
            return revised
    raise ValueError("JAVA_HOST_REPAIR_NO_MEMBER_PROGRESS")


def repair_direct_feature_member(router, source, diagnostic, *, context):
    """Use the actual failed member and original feature context for source repair."""
    filename = context["filename"]
    focused = _first_compiler_error(diagnostic, filename=filename)
    try:
        raw, node, original, frame = _failing_member(source, focused, filename=filename)
    except ValueError as exc:
        if "NOT_A_MEMBER_DIAGNOSTIC" not in str(exc):
            raise
        raw, node, original, frame = _field_initializer(source, focused, filename)
    if context.get("state_access"):
        frame["host_state_access"] = list(context["state_access"])
    payload = _repair_payload(context, source, original, frame, focused)
    revised = _write_repair(router, payload, frame, original, context["attempt"])
    updated = (raw[:node.start_byte] + revised.encode() + raw[node.end_byte:]).decode()
    updated = re.sub(r"(?m)^import\s+(?:static\s+)?[\w.$*]+\s*;\s*\n", "", updated)
    return host_add_verified_imports(
        updated, symbols=context["symbols"], owner=context["owner"],
        target_path=context["target_path"])
