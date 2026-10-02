"""Generate complete semantic record sets without model-owned loop control."""
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
_MAX_RECORD_SET_ITEMS = 16


def record_cardinality_response_schema(
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility name for the semantic record-set contract.

    There is deliberately no count/done/continuation field. The model returns only
    authored records; the host validates and de-duplicates the resulting set.
    """
    record_schema = deepcopy((template or {}).get("record_schema", {}))
    return {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "maxItems": _MAX_RECORD_SET_ITEMS,
                "items": record_schema,
            },
        },
        "required": ["records"],
        "additionalProperties": False,
    }


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


def _record_set_schema(
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
            "records": {
                "type": "array",
                "maxItems": _MAX_RECORD_SET_ITEMS,
                "items": record_schema,
            },
        },
        "required": ["records"],
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
    """Generate semantic records in one host-owned set operation.

    No count pre-pass, ordinal contract, done flag, null sentinel, retry protocol or
    model-owned continuation exists here. The model authors only records. The host
    validates the complete data shape and projects exact duplicates away.
    """
    template = load_record_template(identifier)
    normalized_context = task_context(template, context)
    admitted_refs = {str(ref) for ref in allowed_refs}
    schema = _record_set_schema(identifier, template, normalized_context)
    validator = Draft202012Validator(schema)
    binding = "record-set-v1:" + task_binding(
        template,
        normalized_context,
        admitted_refs,
    )
    saved = (progress or {}).get(binding)

    if isinstance(saved, dict) and isinstance(saved.get("records"), list):
        value = deepcopy(saved)
    else:
        rules = "\n".join(str(rule) for rule in template.get("rules", ()))
        system_prompt = (
            "Author the complete set of distinct records supported by the supplied "
            "authoritative context. The template task/rules below describe one record's "
            "semantics; the outer records array is host-owned transport for the complete "
            "set. Return an empty records array when none apply. Do not emit count, done, "
            "continuation, retry, ordinal, blocked, or other loop-control metadata.\n"
            + str(template.get("task") or "Produce the requested records.")
            + ("\n" + rules if rules else "")
        )
        value = generate_fixed_template_value(
            router,
            "planner",
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(normalized_context, ensure_ascii=False),
                },
            ],
            response_schema=schema,
            enable_tools=False,
            tool_name="submit_records_" + identifier.replace("/", "_"),
        )
        validator.validate(value)
        if checkpoint is not None:
            checkpoint(binding, deepcopy(value))

    validator.validate(value)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in value["records"]:
        key = json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(deepcopy(record))

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
