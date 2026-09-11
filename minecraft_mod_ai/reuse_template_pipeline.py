from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from typing import Any, Mapping

from .planner_operation import planner_operation
from .task_template_catalog import load_template


def _reuse_sequence() -> tuple[str, ...]:
    workflow = load_template("reuse/workflow")
    if workflow.get("execution") != "sequence":
        raise ValueError("REUSE_TEMPLATE: reuse/workflow must be a sequence")
    steps = tuple(workflow.get("steps") or ())
    if not steps or len(steps) != len(set(steps)):
        raise ValueError("REUSE_TEMPLATE: workflow steps must be non-empty and unique")
    return steps


def _reuse_binding(identifier: str, context: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps([identifier, context], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_reuse_template_sequence() -> None:
    for identifier in _reuse_sequence():
        template = load_template(identifier)
        if template.get("id") != identifier:
            raise ValueError(f"REUSE_TEMPLATE: invalid template identity {identifier}")
        if not isinstance(template.get("input"), Mapping) or not isinstance(template.get("output"), Mapping):
            raise ValueError(f"REUSE_TEMPLATE: {identifier} must declare input and output")
        proof = template.get("proof")
        if not isinstance(proof, Mapping) or not str(proof.get("predicate") or "").strip():
            raise ValueError(f"REUSE_TEMPLATE: {identifier} must declare a proof predicate")


def _evaluate_reuse_step(identifier, context, output):
    if identifier == "reuse/license_check":
        unresolved = list(output.get("unresolved_terms") or context.get("unresolved_terms") or [])
        permission = output.get("permission_state") or context.get("permission_state")
        output["unresolved_terms"] = unresolved
        if unresolved or permission in ("forbidden", "blocked", "unknown"):
            return output, False, f"License terms unresolved or not permitted: {unresolved or permission}", "BLOCKED"
        return output, True, "", "PASS"
    if identifier == "reuse/compatibility_check":
        failed = list(output.get("failed_checks") or context.get("failed_checks") or [])
        unresolved = list(output.get("unresolved_checks") or context.get("unresolved_checks") or [])
        explicit = output.get("compatible") if "compatible" in output else context.get("compatible")
        if failed or unresolved or explicit is False:
            output["compatible"] = False
            output["failed_checks"] = failed or ["Compatibility check failure"]
            output["unresolved_checks"] = unresolved
            return output, False, f"Compatibility failed: failed={output['failed_checks']}, unresolved={unresolved}", "BLOCKED"
        output["compatible"] = True
        output["failed_checks"] = []
        output["unresolved_checks"] = []
        return output, True, "", "PASS"
    if identifier == "reuse/direct_reuse":
        blocked = list(output.get("blocking_reasons") or context.get("blocking_reasons") or [])
        explicit = output.get("reusable_directly") if "reusable_directly" in output else context.get("reusable_directly")
        if blocked or explicit is False:
            output["reusable_directly"] = False
            output["blocking_reasons"] = blocked or ["Direct reuse blocked"]
            return output, False, f"Direct reuse blocked: {output['blocking_reasons']}", "BLOCKED"
        output["reusable_directly"] = True
        output["blocking_reasons"] = []
        return output, True, "", "PASS"
    for key in (
        "blocking_reasons", "blocked_reasons", "blocked_points", "failed_checks",
        "unresolved_adaptations", "unresolved_dependencies",
    ):
        value = output.get(key) or context.get(key)
        if value:
            return output, False, f"Blocked by {key}: {value}", "BLOCKED"
    return output, True, "", "PASS"


def execute_reuse_template(identifier, *, context, progress=None, checkpoint=None):
    template = load_template(identifier)
    binding = _reuse_binding(identifier, context)
    saved = (progress or {}).get(binding)
    if saved is not None:
        return deepcopy(saved)
    predicate = str(template["proof"].get("predicate") or "").strip()
    output = {}
    for key in template.get("output", {}):
        value = context.get(key)
        output[key] = deepcopy(value) if value is not None else []
    output, passed, reason, status = _evaluate_reuse_step(identifier, context, output)
    proof = {"passed": passed, "predicate": predicate}
    if not passed and reason:
        proof["reason"] = reason
    receipt = {"template_id": identifier, "status": status, "output": output, "proof": proof}
    if checkpoint is not None:
        checkpoint(binding, deepcopy(receipt))
    return receipt


def evaluate_feature_reuse(
    atomic_feature,
    *,
    platform_target=None,
    progress=None,
    checkpoint=None,
):
    validate_reuse_template_sequence()
    fid = str(atomic_feature.get("feature_id") or "feature")
    receipts = []
    accumulated_context = {
        "atomic_feature": dict(atomic_feature),
        "feature_id": fid,
        "information_need": str(atomic_feature.get("feature_description") or fid),
        "target_platform": dict(platform_target or {}),
        "queries": [],
        "query_constraints": [],
    }
    for identifier in _reuse_sequence():
        with planner_operation(identifier):
            receipt = execute_reuse_template(
                identifier,
                context=accumulated_context,
                progress=progress,
                checkpoint=checkpoint,
            )
            receipts.append(receipt)
            accumulated_context.update(receipt.get("output", {}))
    return {"feature_id": fid, "receipts": receipts, "reuse_plan": accumulated_context}


def evaluate_reuse_pipeline(
    atomic_features,
    *,
    platform_target=None,
    progress=None,
    checkpoint=None,
):
    return [
        evaluate_feature_reuse(
            feature,
            platform_target=platform_target,
            progress=progress,
            checkpoint=checkpoint,
        )
        for feature in atomic_features
    ]
