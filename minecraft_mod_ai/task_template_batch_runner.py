"""Bounded multi-template execution for real fixed-tool model transports.

Batching is transport-only. Every concern produces its complete record set exactly
once; the host owns completion and validation. No synchronized record/done waves exist.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .bounded_record_template import (
    normalize_bounded_record_response,
    record_batch_response_schema,
    run_bounded_record_template,
)
from .model_output_atomicity_contract import MAX_MODEL_FIELDS
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context
from . import task_template_runner as runner


def supports_record_batching(model_router: Any) -> bool:
    if model_router is None:
        return False
    from .model_router import ModelRouter
    return isinstance(model_router, ModelRouter)


def record_template_batches(identifiers: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    width = max(1, int(MAX_MODEL_FIELDS))
    values = tuple(str(identifier) for identifier in identifiers)
    return tuple(values[start : start + width] for start in range(0, len(values), width))


def run_record_template_batch(
    model_router: Any,
    identifiers: Sequence[str],
    *,
    context: Mapping[str, Any],
    allowed_refs: set[str],
    progress: Mapping[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, dict[str, Any]]:
    """Resolve 1..MAX_MODEL_FIELDS independent concerns in one bounded transport call."""
    identifiers = tuple(str(identifier) for identifier in identifiers)
    if not identifiers or len(identifiers) > MAX_MODEL_FIELDS:
        raise ValueError(
            f"TEMPLATE_RECORD_BATCH: expected 1..{MAX_MODEL_FIELDS} identifiers"
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("TEMPLATE_RECORD_BATCH: duplicate identifiers")
    if not supports_record_batching(model_router):
        return {
            identifier: run_bounded_record_template(
                model_router,
                identifier,
                context=dict(context),
                allowed_refs=allowed_refs,
                progress=progress,
                checkpoint=checkpoint,
            )
            for identifier in identifiers
        }

    allowed_refs = {str(ref) for ref in allowed_refs}
    states: dict[str, dict[str, Any]] = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    payload: dict[str, Any] = {}
    instructions = [
        "Resolve each independent Minecraft engineering concern completely in this one call.",
        "For every slot return records, blocked_reason, and evidence_refs only.",
        "Return an empty records array when that concern has no applicable record.",
        "Do not emit continuation, done, applicability, retry, or loop-control decisions.",
    ]

    for index, identifier in enumerate(identifiers):
        template = load_record_template(identifier)
        normalized = task_context(template, context)
        slot = f"c{index}"
        binding = "bounded-record-v2:" + task_binding(template, normalized, allowed_refs)
        states[identifier] = {
            "slot": slot,
            "template": template,
            "context": normalized,
            "binding": binding,
        }
        properties[slot] = record_batch_response_schema(template)
        required.append(slot)
        payload[slot] = {
            "template_id": identifier,
            **normalized,
            "allowed_evidence_refs": sorted(allowed_refs),
        }
        local_rules = "\n".join(str(rule) for rule in template.get("rules", ()))
        instructions.append(
            f"[{slot}] {identifier}: {template['task']}"
            + ("\n" + local_rules if local_rules else "")
        )

    saved_values: dict[str, Any] = {}
    missing: list[str] = []
    for identifier, state in states.items():
        saved = (progress or {}).get(state["binding"])
        if saved is None:
            missing.append(identifier)
        elif not isinstance(saved, Mapping):
            raise ValueError(f"TEMPLATE_PROGRESS: expected object for {identifier}")
        else:
            saved_values[state["slot"]] = deepcopy(dict(saved))

    generated: dict[str, Any] = {}
    if missing:
        missing_slots = {states[identifier]["slot"] for identifier in missing}
        request_schema = {
            "type": "object",
            "properties": {slot: properties[slot] for slot in missing_slots},
            "required": sorted(missing_slots),
            "additionalProperties": False,
        }
        request_payload = {
            slot: value for slot, value in payload.items() if slot in missing_slots
        }
        request_instructions = [instructions[0], instructions[1], instructions[2], instructions[3]]
        for identifier in missing:
            slot = states[identifier]["slot"]
            template = states[identifier]["template"]
            local_rules = "\n".join(str(rule) for rule in template.get("rules", ()))
            request_instructions.append(
                f"[{slot}] {identifier}: {template['task']}"
                + ("\n" + local_rules if local_rules else "")
            )
        generated = runner.generate_fixed_template_value(
            model_router,
            "planner",
            [
                {"role": "system", "content": "\n\n".join(request_instructions)},
                {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
            ],
            response_schema=request_schema,
            enable_tools=False,
            tool_name="submit_detail_concern_batch",
        )
        Draft202012Validator(request_schema).validate(generated)

    combined = {**saved_values, **generated}
    results: dict[str, dict[str, Any]] = {}
    for identifier, state in states.items():
        slot = state["slot"]
        if slot not in combined:
            raise ValueError(f"TEMPLATE_RECORD_BATCH: missing response for {identifier}")
        raw = deepcopy(combined[slot])
        result = normalize_bounded_record_response(
            identifier, state["template"], raw, allowed_refs
        )
        results[identifier] = result
        if identifier in missing and checkpoint is not None:
            checkpoint(state["binding"], deepcopy(raw))
    return results


__all__ = [
    "record_template_batches",
    "run_record_template_batch",
    "supports_record_batching",
]
