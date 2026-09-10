from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from typing import Any, Mapping

from .planner_operation import planner_operation
from .task_template_catalog import load_template

REUSE_SEQUENCE: tuple[str, ...] = (
    "reuse/query_build",
    "reuse/official_docs",
    "reuse/official_examples",
    "reuse/existing_mods",
    "reuse/repository_search",
    "reuse/file_search",
    "reuse/class_search",
    "reuse/method_search",
    "reuse/dependency_search",
    "reuse/license_check",
    "reuse/compatibility_check",
    "reuse/direct_reuse",
    "reuse/pattern_reuse",
    "reuse/adaptation",
    "reuse/integration",
)


def _reuse_binding(identifier: str, context: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps([identifier, context], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_reuse_template_sequence() -> None:
    for identifier in REUSE_SEQUENCE:
        template = load_template(identifier)
        if template.get("id") != identifier:
            raise ValueError(f"REUSE_TEMPLATE: invalid template identity {identifier}")
        if not isinstance(template.get("input"), Mapping) or not isinstance(template.get("output"), Mapping):
            raise ValueError(f"REUSE_TEMPLATE: {identifier} must declare input and output")
        proof = template.get("proof")
        if not isinstance(proof, Mapping) or not str(proof.get("predicate") or "").strip():
            raise ValueError(f"REUSE_TEMPLATE: {identifier} must declare a proof predicate")


def _evaluate_reuse_step(
    identifier: str,
    context: Mapping[str, Any],
    output: dict[str, Any],
) -> tuple[dict[str, Any], bool, str, str]:
    if identifier == "reuse/license_check":
        unres = list(output.get("unresolved_terms") or context.get("unresolved_terms") or [])
        perm = output.get("permission_state") or context.get("permission_state")
        output["unresolved_terms"] = unres
        if unres or perm in ("forbidden", "blocked", "unknown"):
            return output, False, f"License terms unresolved or not permitted: {unres or perm}", "BLOCKED"
        return output, True, "", "PASS"

    if identifier == "reuse/compatibility_check":
        failed = list(output.get("failed_checks") or context.get("failed_checks") or [])
        unres = list(output.get("unresolved_checks") or context.get("unresolved_checks") or [])
        explicit_compat = output.get("compatible") if "compatible" in output else context.get("compatible")
        if failed or unres or explicit_compat is False:
            output["compatible"] = False
            output["failed_checks"] = failed or ["Compatibility check failure"]
            output["unresolved_checks"] = unres
            return output, False, f"Compatibility failed: failed={output['failed_checks']}, unresolved={unres}", "BLOCKED"
        output["compatible"] = True
        output["failed_checks"] = []
        output["unresolved_checks"] = []
        return output, True, "", "PASS"

    if identifier == "reuse/direct_reuse":
        blocked = list(output.get("blocking_reasons") or context.get("blocking_reasons") or [])
        explicit_direct = output.get("reusable_directly") if "reusable_directly" in output else context.get("reusable_directly")
        if blocked or explicit_direct is False:
            output["reusable_directly"] = False
            output["blocking_reasons"] = blocked or ["Direct reuse blocked"]
            return output, False, f"Direct reuse blocked: {output['blocking_reasons']}", "BLOCKED"
        output["reusable_directly"] = True
        output["blocking_reasons"] = []
        return output, True, "", "PASS"

    # Generic check for blockers in reuse outputs
    for key in ("blocking_reasons", "blocked_reasons", "blocked_points", "failed_checks", "unresolved_adaptations", "unresolved_dependencies"):
        val = output.get(key) or context.get(key)
        if val:
            return output, False, f"Blocked by {key}: {val}", "BLOCKED"
    return output, True, "", "PASS"


def execute_reuse_template(
    identifier: str,
    *,
    context: Mapping[str, Any],
    progress: dict[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, Any]:
    template = load_template(identifier)
    binding = _reuse_binding(identifier, context)
    saved = (progress or {}).get(binding)
    if saved is not None:
        return deepcopy(saved)

    proof = template["proof"]
    predicate = str(proof.get("predicate") or "").strip()

    output_keys = list(template.get("output", {}).keys())
    output: dict[str, Any] = {}
    for key in output_keys:
        val = context.get(key)
        output[key] = deepcopy(val) if val is not None else []

    output, passed, reason, status = _evaluate_reuse_step(identifier, context, output)

    proof_payload: dict[str, Any] = {"passed": passed, "predicate": predicate}
    if not passed and reason:
        proof_payload["reason"] = reason

    receipt = {
        "template_id": identifier,
        "status": status,
        "output": output,
        "proof": proof_payload,
    }

    if checkpoint is not None:
        checkpoint(binding, deepcopy(receipt))
    return receipt


def evaluate_feature_reuse(
    atomic_feature: Mapping[str, Any],
    *,
    platform_target: Mapping[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, Any]:
    validate_reuse_template_sequence()
    fid = str(atomic_feature.get("feature_id") or "feature")

    receipts = []
    accumulated_context: dict[str, Any] = {
        "atomic_feature": dict(atomic_feature),
        "feature_id": fid,
        "information_need": str(atomic_feature.get("feature_description") or fid),
        "target_platform": dict(platform_target or {}),
        "queries": [],
        "query_constraints": [],
    }

    for identifier in REUSE_SEQUENCE:
        with planner_operation(identifier):
            receipt = execute_reuse_template(
                identifier,
                context=accumulated_context,
                progress=progress,
                checkpoint=checkpoint,
            )
            receipts.append(receipt)
            accumulated_context.update(receipt.get("output", {}))

    return {
        "feature_id": fid,
        "receipts": receipts,
        "reuse_plan": accumulated_context,
    }


def evaluate_reuse_pipeline(
    atomic_features: list[Mapping[str, Any]],
    *,
    platform_target: Mapping[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    checkpoint=None,
) -> list[dict[str, Any]]:
    return [
        evaluate_feature_reuse(
            feature,
            platform_target=platform_target,
            progress=progress,
            checkpoint=checkpoint,
        )
        for feature in atomic_features
    ]
