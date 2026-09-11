"""Bounded multi-template execution for real fixed-tool model transports.

The host owns batching and checkpoint identity. Each child concern keeps the same template,
context binding, validation rules, and response history as ``run_record_template``; only the
transport combines up to the global small-model field limit into one model decision.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .model_output_atomicity_contract import MAX_MODEL_FIELDS
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context
from . import task_template_runner as runner


def supports_record_batching(model_router: Any) -> bool:
    """Batch only transports with native fixed-tool arguments.

    Lightweight fixture routers intentionally keep the legacy one-concern path so tests and
    deterministic mocks do not need to emulate the nested native-tool envelope.
    """

    return model_router is not None and callable(
        getattr(model_router, "generate_tool_decision", None)
    )


def record_template_batches(identifiers: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    width = max(1, int(MAX_MODEL_FIELDS))
    values = tuple(str(identifier) for identifier in identifiers)
    return tuple(values[start : start + width] for start in range(0, len(values), width))


def _state(
    identifier: str,
    index: int,
    *,
    context: Mapping[str, Any],
    allowed_refs: set[str],
    progress: Mapping[str, Any] | None,
) -> dict[str, Any]:
    template = load_record_template(identifier)
    normalized = task_context(template, context)
    binding = task_binding(template, normalized, allowed_refs)
    saved = deepcopy((progress or {}).get(binding, []))
    if not isinstance(saved, list):
        raise ValueError("TEMPLATE_PROGRESS: expected response array")
    return {
        "slot": f"c{index}",
        "template": template,
        "context": normalized,
        "binding": binding,
        "saved": saved,
        "accepted": [],
        "records": [],
        "refs": [],
        "seen": set(),
        "done": False,
        "reason": "",
    }


def _generate_wave(
    model_router: Any,
    identifiers: Sequence[str],
    states: Mapping[str, dict[str, Any]],
    *,
    allowed_refs: set[str],
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    payload: dict[str, Any] = {}
    required: list[str] = []
    instructions = [
        "Advance each independent Minecraft engineering concern by exactly one fixed-template transition.",
        "For every slot emit exactly one of record/done/not_applicable/blocked and obey only that slot's task and rules.",
        "Do not merge concerns, invent neighboring work, or repeat an already accepted record.",
    ]
    for identifier in identifiers:
        state = states[identifier]
        slot = state["slot"]
        template = state["template"]
        properties[slot] = runner.record_response_schema(template)
        required.append(slot)
        payload[slot] = {
            "template_id": identifier,
            **state["context"],
            "accepted_records": deepcopy(state["records"]),
            "allowed_evidence_refs": sorted(allowed_refs),
        }
        instructions.append(
            f"[{slot}] {identifier}: {template['task']}\n"
            + "\n".join(str(rule) for rule in template.get("rules", ()))
        )

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    value = runner.generate_fixed_template_value(
        model_router,
        "planner",
        [
            {"role": "system", "content": "\n\n".join(instructions)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_schema=schema,
        enable_tools=False,
        tool_name="submit_detail_concern_batch",
    )
    Draft202012Validator(schema).validate(value)
    return dict(value)


def _accept_response(
    identifier: str,
    state: dict[str, Any],
    raw_value: Mapping[str, Any],
    *,
    allowed_refs: set[str],
) -> None:
    value = deepcopy(dict(raw_value))
    evidence_from_value = value.pop("evidence_refs", None)
    Draft202012Validator(runner.record_response_schema(state["template"])).validate(value)

    if evidence_from_value is not None:
        value["evidence_refs"] = list(evidence_from_value)
    else:
        value["evidence_refs"] = runner._grounded_evidence_refs(
            state["context"], allowed_refs
        )
    if any(ref not in allowed_refs for ref in value["evidence_refs"]):
        raise ValueError(f"TEMPLATE_EVIDENCE: unknown evidence in {identifier}")

    status = value["status"]
    record = value["record"]
    reason = value["reason"].strip()
    if status == "record":
        if record is None or runner._contains_blank_string(
            record, state["template"]["record_schema"]
        ):
            raise ValueError(f"TEMPLATE_RECORD: empty record in {identifier}")
        key = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if key in state["seen"]:
            raise runner.TemplateBlocked(
                f"TEMPLATE_NO_PROGRESS: repeated record in {identifier}"
            )
        state["seen"].add(key)
        state["records"].append(record)
        state["refs"].extend(
            ref for ref in value["evidence_refs"] if ref not in state["refs"]
        )
    elif record is not None:
        raise ValueError(f"TEMPLATE_STATUS: {status} cannot carry a record")
    elif status == "blocked":
        raise runner.TemplateBlocked(
            f"TEMPLATE_BLOCKED: {identifier}: {reason or 'missing blocking reason'}"
        )
    elif status == "done" and state["records"]:
        state["refs"].extend(
            ref for ref in value["evidence_refs"] if ref not in state["refs"]
        )
        state["done"] = True
    elif status == "not_applicable" and not state["records"] and reason:
        state["refs"] = list(value["evidence_refs"])
        state["reason"] = reason
        state["done"] = True
    else:
        raise ValueError(f"TEMPLATE_STATUS: invalid {status} transition in {identifier}")

    state["accepted"].append(deepcopy(value))
    if state["done"] and len(state["accepted"]) < len(state["saved"]):
        raise ValueError("TEMPLATE_PROGRESS: responses after completion")


def run_record_template_batch(
    model_router: Any,
    identifiers: Sequence[str],
    *,
    context: Mapping[str, Any],
    allowed_refs: set[str],
    progress: Mapping[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, dict[str, Any]]:
    """Run 1..MAX_MODEL_FIELDS independent record templates as synchronized waves."""

    identifiers = tuple(str(identifier) for identifier in identifiers)
    if not identifiers or len(identifiers) > MAX_MODEL_FIELDS:
        raise ValueError(
            f"TEMPLATE_RECORD_BATCH: expected 1..{MAX_MODEL_FIELDS} identifiers"
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("TEMPLATE_RECORD_BATCH: duplicate identifiers")
    if not supports_record_batching(model_router):
        return {
            identifier: runner.run_record_template(
                model_router,
                identifier,
                context=context,
                allowed_refs=allowed_refs,
                progress=progress,
                checkpoint=checkpoint,
            )
            for identifier in identifiers
        }

    states = {
        identifier: _state(
            identifier,
            index,
            context=context,
            allowed_refs=allowed_refs,
            progress=progress,
        )
        for index, identifier in enumerate(identifiers)
    }

    while not all(bool(state["done"]) for state in states.values()):
        generated: dict[str, Any] = {}
        missing: list[str] = []
        replay_flags: dict[str, bool] = {}
        for identifier, state in states.items():
            if state["done"]:
                continue
            replaying = len(state["accepted"]) < len(state["saved"])
            replay_flags[identifier] = replaying
            if replaying:
                generated[state["slot"]] = deepcopy(
                    state["saved"][len(state["accepted"])]
                )
            else:
                missing.append(identifier)

        if missing:
            generated.update(
                _generate_wave(
                    model_router,
                    missing,
                    states,
                    allowed_refs=allowed_refs,
                )
            )

        for identifier, state in states.items():
            if state["done"]:
                continue
            slot = state["slot"]
            if slot not in generated:
                raise ValueError(f"TEMPLATE_RECORD_BATCH: missing response for {identifier}")
            replaying = replay_flags.get(identifier, False)
            _accept_response(
                identifier,
                state,
                generated[slot],
                allowed_refs=allowed_refs,
            )
            if not replaying and checkpoint is not None:
                checkpoint(state["binding"], deepcopy(state["accepted"]))

    return {
        identifier: {
            "records": deepcopy(state["records"]),
            "reason": state["reason"],
            "evidence_refs": list(state["refs"]),
        }
        for identifier, state in states.items()
    }


__all__ = [
    "record_template_batches",
    "run_record_template_batch",
    "supports_record_batching",
]
