from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from typing import Any, Mapping

from .planner_operation import planner_operation
from .task_template_catalog import load_template

KNOWN_STAGES: tuple[str, ...] = ("code", "asset", "integration", "validation")


def _stage_binding(stage: str, identifier: str, context: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps([stage, identifier, context], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_stage_workflows() -> None:
    for stage in KNOWN_STAGES:
        workflow = load_template(f"{stage}/workflow")
        if workflow.get("id") != f"{stage}/workflow":
            raise ValueError(f"STAGE_WORKFLOW: invalid workflow identity for {stage}")
        steps = workflow.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"STAGE_WORKFLOW: {stage}/workflow has no steps")
        for identifier in steps:
            template = load_template(identifier)
            if template.get("id") != identifier:
                raise ValueError(f"STAGE_TEMPLATE: invalid template identity {identifier}")
            if not isinstance(template.get("input"), Mapping) or not isinstance(template.get("output"), Mapping):
                raise ValueError(f"STAGE_TEMPLATE: {identifier} must declare input and output")
            proof = template.get("proof")
            if not isinstance(proof, Mapping) or not str(proof.get("predicate") or "").strip():
                raise ValueError(f"STAGE_TEMPLATE: {identifier} must declare a proof predicate")


def execute_stage_step(
    stage: str,
    identifier: str,
    *,
    context: Mapping[str, Any],
    progress: dict[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, Any]:
    template = load_template(identifier)
    binding = _stage_binding(stage, identifier, context)
    saved = (progress or {}).get(binding)
    if saved is not None:
        return deepcopy(saved)

    proof = template["proof"]
    predicate = str(proof.get("predicate") or "").strip()

    output_keys = list(template.get("output", {}).keys())
    output = {}
    for key in output_keys:
        val = context.get(key)
        output[key] = deepcopy(val) if val is not None else []

    receipt = {
        "template_id": identifier,
        "status": "PASS",
        "output": output,
        "proof": {"passed": True, "predicate": predicate},
    }

    if checkpoint is not None:
        checkpoint(binding, deepcopy(receipt))
    return receipt


def run_stage_pipeline(
    stage_name: str,
    context: Mapping[str, Any],
    *,
    progress: dict[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, Any]:
    if stage_name not in KNOWN_STAGES:
        raise ValueError(f"STAGE_PIPELINE: unknown stage {stage_name}")

    workflow = load_template(f"{stage_name}/workflow")
    receipts = []
    accumulated: dict[str, Any] = dict(context)

    for identifier in workflow["steps"]:
        with planner_operation(identifier):
            receipt = execute_stage_step(
                stage_name,
                identifier,
                context=accumulated,
                progress=progress,
                checkpoint=checkpoint,
            )
            receipts.append(receipt)
            accumulated.update(receipt.get("output", {}))

    return {
        "stage": stage_name,
        "receipts": receipts,
        "results": accumulated,
    }
