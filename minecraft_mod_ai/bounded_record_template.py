"""Generate record sets from accepted semantic records, never a model-owned count."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .design_generation_schema import context_bound_record_schema
from .fixed_template_generation import generate_fixed_template_value
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context

_EMPTY_REASON = "No applicable records in the supplied context."
_MAX_RECORD_DISCOVERY_ROUNDS = 64


def record_cardinality_response_schema(
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility surface for the active next-record contract.

    Cardinality is no longer requested as a separate integer. The model contributes at
    most one semantic record per call, or null when no additional record applies.
    """
    record_schema = deepcopy((template or {}).get("record_schema", {}))
    return {
        "type": "object",
        "properties": {
            "record": {
                "anyOf": [
                    record_schema,
                    {"type": "null"},
                ],
            },
        },
        "required": ["record"],
        "additionalProperties": False,
    }


# Compatibility name retained for callers that inspect the active record-control schema.
record_batch_response_schema = record_cardinality_response_schema


def _host_evidence_refs(context: dict[str, Any], allowed_refs: set[str]) -> list[str]:
    """Admit only evidence identifiers already present in host-supplied context."""
    refs: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value in allowed_refs and value not in refs:
            refs.append(value)

    for key in ("source_evidence_id", "evidence_ref", "shard_id"):
        add(context.get(key))
    evidence = context.get("evidence")
    if isinstance(evidence, str):
        add(evidence)
    elif isinstance(evidence, (list, tuple, set)):
        for item in evidence:
            if isinstance(item, str):
                add(item)
            elif isinstance(item, dict):
                for key in ("evidence_id", "source_evidence_id", "ref", "id", "shard_id"):
                    add(item.get(key))
    return refs


def _next_record_schema(
    identifier: str,
    template: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    record_schema = context_bound_record_schema(
        identifier,
        template["record_schema"],
        context,
    )
    return {
        "type": "object",
        "properties": {
            "record": {
                "anyOf": [
                    record_schema,
                    {"type": "null"},
                ],
            },
        },
        "required": ["record"],
        "additionalProperties": False,
    }


def run_bounded_record_template(
    router,
    identifier: str,
    *,
    context: dict[str, Any],
    allowed_refs=(),
    progress=None,
    checkpoint=None,
):
    """Accumulate distinct records until semantic convergence.

    The host owns accepted-record state. There is no count pre-pass, ordinal contract,
    or fatal duplicate-cardinality gate. A null next record or an exact duplicate means
    the semantic frontier has converged. The finite round cap is only a runtime safety
    budget against a non-converging model, not a claimed semantic cardinality.
    """
    template = load_record_template(identifier)
    normalized_context = task_context(template, context)
    admitted_refs = {str(ref) for ref in allowed_refs}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    rules = "\n".join(str(rule) for rule in template.get("rules", ()))
    system_prompt = (
        str(template.get("task") or "Produce the requested record.")
        + ("\n" + rules if rules else "")
        + "\nThe host supplies accepted_records as immutable semantic progress. "
        "Return at most one additional distinct applicable record in the record field. "
        "Return record=null only when no additional distinct record is supported by the "
        "authoritative context. Do not estimate, declare, or preserve a total count or "
        "ordinal. Do not repeat an accepted record."
    )

    for _round in range(_MAX_RECORD_DISCOVERY_ROUNDS):
        iteration_context = {
            **normalized_context,
            "accepted_records": deepcopy(records),
        }
        schema = _next_record_schema(identifier, template, iteration_context)
        validator = Draft202012Validator(schema)
        binding = "record-next-v1:" + task_binding(
            template,
            iteration_context,
            admitted_refs,
        )
        saved = (progress or {}).get(binding)

        if isinstance(saved, dict) and "record" in saved:
            value = deepcopy(saved)
        else:
            value = generate_fixed_template_value(
                router,
                "planner",
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(iteration_context, ensure_ascii=False),
                    },
                ],
                response_schema=schema,
                enable_tools=False,
                tool_name="submit_next_" + identifier.replace("/", "_"),
            )
            validator.validate(value)
            if checkpoint is not None:
                checkpoint(binding, deepcopy(value))

        validator.validate(value)
        record = value.get("record")
        if record is None:
            break
        if not isinstance(record, dict):
            raise ValueError(f"TEMPLATE_RECORD_INVALID: {identifier} produced a non-object record")

        key = json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if key in seen:
            # Repetition is convergence/no-progress, not a correctness failure.
            break
        seen.add(key)
        records.append(deepcopy(record))
    else:
        raise RuntimeError(
            "TEMPLATE_RESOURCE_LIMIT: next-record generation did not converge within "
            f"{_MAX_RECORD_DISCOVERY_ROUNDS} host iterations for {identifier}"
        )

    return {
        "records": records,
        "reason": "" if records else _EMPTY_REASON,
        "evidence_refs": _host_evidence_refs(normalized_context, admitted_refs),
    }


run_record_template = run_bounded_record_template

__all__ = [
    "record_batch_response_schema",
    "record_cardinality_response_schema",
    "run_bounded_record_template",
    "run_record_template",
]
