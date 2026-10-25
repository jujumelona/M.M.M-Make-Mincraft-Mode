"""Retain host callback ownership when a real compiler rejects a body."""
from __future__ import annotations

import hashlib
import json

from .host_callback_body import author_callback_body
from .java_region_parser import _text


def repair_callback_body(router, *, raw, node, original, frame, diagnostic,
                         target_path, symbols, behaviors, api_evidence,
                         prerequisite_api, neighbor_signatures, attempt,
                         runtime_state_bindings=(), runtime_mutation_receipts=None):
    """Return an operation-compiled replacement or None for other members."""
    name = _text(raw, node.child_by_field_name("name"))
    selected = None
    for behavior in behaviors:
        suffix = hashlib.sha256(str(behavior["behavior_id"]).encode()).hexdigest()[:20]
        if name in {"handleBehavior" + suffix, "registerBehavior" + suffix}:
            error = ValueError(
                "JAVA_HOST_OWNED_CALLBACK: compiler rejected host dispatch/registration; "
                "do not replace it with an AI-authored stub; " + diagnostic)
            error.failed_source = original
            raise error
        if name in {"matchesBehavior" + suffix, "applyBehavior" + suffix}:
            selected = behavior
            break
    if selected is None:
        return None
    predicate = name.startswith("matchesBehavior")
    frame["host_behavior_contract"] = selected
    parameters = []
    for parameter in node.child_by_field_name("parameters").named_children:
        if parameter.type == "formal_parameter":
            parameters.append(_text(raw, parameter.child_by_field_name("type")))
    from .construction_file_api_evidence import FileAPIEvidence
    api = FileAPIEvidence(symbols, target_path=target_path, owner="")
    intent = selected.get("trigger") if predicate else selected.get("effects")
    runtime_names = {*selected.get("runtime_state_reads", ()), *selected.get("runtime_state_writes", ())}
    task = {
        "target_path": target_path, "diagnostic": diagnostic,
        "previous_implementation": original, "response_source_frame": frame,
        "assigned_behavior": selected,
        "current_unit_purpose": intent,
        "member_role": "test_callback_condition" if predicate else "implement_callback_effects",
        "verified_event_callback": {
            # These are the actual parsed helper inputs. No new listener or
            # external callback ABI is inferred during compiler repair.
            "parameters": parameters,
            "return_type": _text(raw, node.child_by_field_name("type")),
        },
        "callback_input_api": api.callback_member_records(
            parameters, intent=intent or "", predicate=predicate),
        "declared_member_api": [{"signature": s} for s in neighbor_signatures],
        "inspected_target_api": api_evidence,
        "materialized_prerequisite_api": list(prerequisite_api),
        "runtime_state_observations": {k: v for k, v in dict(runtime_state_bindings).items()
                                       if k in runtime_names},
    }
    messages = [{"role": "user", "content": json.dumps(task)}]
    revised_receipts = []
    if not predicate and selected.get("runtime_state_writes"):
        from .host_runtime_mutation_catalog import bind_runtime_mutation_catalog
        frame["host_verified_runtime_mutation_candidates"] = bind_runtime_mutation_catalog(
            api, task, frame)
        frame["host_runtime_mutation_receipts"] = revised_receipts
    revised = author_callback_body(router, messages,
                                   data_name=f"repair_callback_{attempt}_{name}",
                                   frame=frame, predicate=predicate, effect_receipt=not predicate,
                                   rejected_compiler_member=original,
                                   compiler_diagnostic=diagnostic)
    if revised == original:
        raise ValueError("JAVA_HOST_REPAIR_NO_MEMBER_PROGRESS")
    if not predicate and selected.get("runtime_state_writes"):
        from .runtime_mutation_author import verify_native_runtime_mutation_coverage
        verify_native_runtime_mutation_coverage(
            "class MmmRepairedCallback {" + revised + "}", [selected], revised_receipts)
        if runtime_mutation_receipts is not None:
            identity = str(selected["behavior_id"])
            runtime_mutation_receipts[:] = [
                row for row in runtime_mutation_receipts
                if str(row.get("behavior_id") or "") != identity
            ] + revised_receipts
    return revised
