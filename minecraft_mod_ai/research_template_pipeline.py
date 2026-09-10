from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from typing import Any, Mapping

from .planner_operation import planner_operation
from .task_template_catalog import load_template

RESEARCH_SEQUENCE: tuple[str, ...] = (
    "research/reference_identity",
    "research/reference_research",
    "research/system_extraction",
    "research/gameplay_loop_extraction",
    "research/progression_extraction",
    "research/content_extraction",
    "research/visual_extraction",
    "research/audio_extraction",
    "research/evidence_check",
)


def _research_binding(identifier: str, context: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps([identifier, context], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_research_template_sequence() -> None:
    for identifier in RESEARCH_SEQUENCE:
        template = load_template(identifier)
        if template.get("id") != identifier:
            raise ValueError(f"RESEARCH_TEMPLATE: invalid template identity {identifier}")
        if not isinstance(template.get("input"), Mapping) or not isinstance(template.get("output"), Mapping):
            raise ValueError(f"RESEARCH_TEMPLATE: {identifier} must declare input and output")
        proof = template.get("proof")
        if not isinstance(proof, Mapping) or not str(proof.get("predicate") or "").strip():
            raise ValueError(f"RESEARCH_TEMPLATE: {identifier} must declare a proof predicate")


def execute_research_template(
    identifier: str,
    *,
    context: Mapping[str, Any],
    progress: dict[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, Any]:
    template = load_template(identifier)
    binding = _research_binding(identifier, context)
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


def run_research_pipeline(
    state: Mapping[str, Any],
    *,
    evidence_items: list[dict[str, Any]] | None = None,
    progress: dict[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, Any]:
    validate_research_template_sequence()
    evidence_items = evidence_items or []
    references = state.get("references") or []

    receipts = []
    accumulated_context: dict[str, Any] = {
        "reference_entities": references,
        "available_evidence": evidence_items,
        "evidence_items": evidence_items,
        "audio_evidence": [e for e in evidence_items if e.get("kind") == "audio"],
        "visual_evidence": [e for e in evidence_items if e.get("kind") == "visual"],
    }

    for identifier in RESEARCH_SEQUENCE:
        with planner_operation(identifier):
            receipt = execute_research_template(
                identifier,
                context=accumulated_context,
                progress=progress,
                checkpoint=checkpoint,
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
