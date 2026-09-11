from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from typing import Any, Mapping

from .planner_operation import planner_operation
from .task_template_catalog import load_template


def _research_sequence() -> tuple[str, ...]:
    workflow = load_template("research/workflow")
    if workflow.get("execution") != "sequence":
        raise ValueError("RESEARCH_TEMPLATE: research/workflow must be a sequence")
    steps = tuple(workflow.get("steps") or ())
    if not steps or len(steps) != len(set(steps)):
        raise ValueError("RESEARCH_TEMPLATE: workflow steps must be non-empty and unique")
    return steps


def _research_binding(identifier: str, context: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps([identifier, context], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_research_template_sequence() -> None:
    for identifier in _research_sequence():
        template = load_template(identifier)
        if template.get("id") != identifier:
            raise ValueError(f"RESEARCH_TEMPLATE: invalid template identity {identifier}")
        if not isinstance(template.get("input"), Mapping) or not isinstance(template.get("output"), Mapping):
            raise ValueError(f"RESEARCH_TEMPLATE: {identifier} must declare input and output")
        proof = template.get("proof")
        if not isinstance(proof, Mapping) or not str(proof.get("predicate") or "").strip():
            raise ValueError(f"RESEARCH_TEMPLATE: {identifier} must declare a proof predicate")


def _evaluate_research_step(identifier, context, output):
    if identifier == "research/reference_identity":
        unres = list(output.get("unresolved_identities") or context.get("unresolved_identities") or [])
        output["unresolved_identities"] = unres
        if unres:
            return output, False, f"Unresolved identities: {unres}", "BLOCKED"
        return output, True, "", "PASS"
    if identifier == "research/evidence_check":
        failed = list(output.get("failed_checks") or context.get("failed_checks") or [])
        conflicts = list(output.get("unresolved_conflicts") or context.get("unresolved_conflicts") or [])
        explicit = output.get("supported") if "supported" in output else context.get("supported")
        if failed or conflicts or explicit is False:
            output["supported"] = False
            output["failed_checks"] = failed or ["Evidence check failure"]
            output["unresolved_conflicts"] = conflicts
            return output, False, f"Evidence check failed: failed={output['failed_checks']}, conflicts={conflicts}", "BLOCKED"
        output["supported"] = True
        output["failed_checks"] = []
        output["unresolved_conflicts"] = []
        return output, True, "", "PASS"
    for key in ("unanswered_questions", "failed_checks", "unresolved_conflicts", "blocked_reason"):
        val = output.get(key) or context.get(key)
        if val:
            return output, False, f"Blocked by {key}: {val}", "BLOCKED"
    return output, True, "", "PASS"


def execute_research_template(identifier, *, context, progress=None, checkpoint=None):
    template = load_template(identifier)
    binding = _research_binding(identifier, context)
    saved = (progress or {}).get(binding)
    if saved is not None:
        return deepcopy(saved)
    proof = template["proof"]
    predicate = str(proof.get("predicate") or "").strip()
    output = {}
    for key in template.get("output", {}):
        val = context.get(key)
        output[key] = deepcopy(val) if val is not None else []
    output, passed, reason, status = _evaluate_research_step(identifier, context, output)
    proof_payload = {"passed": passed, "predicate": predicate}
    if not passed and reason:
        proof_payload["reason"] = reason
    receipt = {"template_id": identifier, "status": status, "output": output, "proof": proof_payload}
    if checkpoint is not None:
        checkpoint(binding, deepcopy(receipt))
    return receipt


def run_research_pipeline(state, *, evidence_items=None, progress=None, checkpoint=None):
    validate_research_template_sequence()
    evidence_items = evidence_items or []
    references = state.get("references") or []
    receipts = []
    accumulated_context = {
        "reference_entities": references,
        "available_evidence": evidence_items,
        "evidence_items": evidence_items,
        "audio_evidence": [e for e in evidence_items if e.get("kind") == "audio"],
        "visual_evidence": [e for e in evidence_items if e.get("kind") == "visual"],
    }
    for identifier in _research_sequence():
        with planner_operation(identifier):
            receipt = execute_research_template(
                identifier, context=accumulated_context, progress=progress, checkpoint=checkpoint
            )
            receipts.append(receipt)
            accumulated_context.update(receipt.get("output", {}))
    return {
        "receipts": receipts,
        "verified_identities": accumulated_context.get("verified_identities", []),
        "systems": accumulated_context.get("systems", []),
        "gameplay_loops": accumulated_context.get("gameplay_loops", []),
        "progression_records": accumulated_context.get("progression_records", []),
        "content_records": accumulated_context.get("content_records", []),
        "visual_records": accumulated_context.get("visual_records", []),
        "audio_records": accumulated_context.get("audio_records", []),
    }
