"""Bounded record-template execution for small models.

One model call owns one narrow extraction decision. The host owns completion,
empty-result applicability, evidence admission and validation. The model never
drives record/done/applicable/retry loop control.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .fixed_template_generation import generate_fixed_template_value
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context
from .task_template_runner import TemplateBlocked

_EMPTY_REASON = "No applicable records in the supplied context."


def _contains_blank_string(value: Any, schema: dict[str, Any] | None = None) -> bool:
    schema = schema or {}
    if isinstance(value, str):
        return not value.strip() and schema.get("minLength", 1) > 0
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return any(_contains_blank_string(item, properties.get(key)) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_blank_string(item, schema.get("items")) for item in value)
    return False


def record_batch_response_schema(template: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": deepcopy(template["record_schema"]),
            },
            "blocked_reason": {"type": "string", "maxLength": 512},
            "evidence_refs": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
        },
        "required": ["records", "blocked_reason", "evidence_refs"],
        "additionalProperties": False,
    }


def normalize_bounded_record_response(identifier, template, value, allowed_refs):
    schema = record_batch_response_schema(template)
    Draft202012Validator(schema).validate(value)
    records = list(value["records"])
    blocked_reason = value["blocked_reason"].strip()
    evidence_refs = list(value["evidence_refs"])

    unknown_refs = [ref for ref in evidence_refs if ref not in allowed_refs]
    if unknown_refs:
        raise ValueError(f"TEMPLATE_EVIDENCE: unknown evidence in {identifier}: {unknown_refs}")
    if blocked_reason:
        if records:
            raise ValueError(f"TEMPLATE_BLOCKED: {identifier} cannot carry records and a blocker")
        raise TemplateBlocked(f"TEMPLATE_BLOCKED: {identifier}: {blocked_reason}")

    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    for record in records:
        if _contains_blank_string(record, template["record_schema"]):
            raise ValueError(f"TEMPLATE_RECORD: empty record in {identifier}")
        key = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if key in seen:
            raise TemplateBlocked(f"TEMPLATE_NO_PROGRESS: duplicate record in {identifier}")
        seen.add(key)
        accepted.append(record)
    return {
        "records": accepted,
        "reason": "" if accepted else _EMPTY_REASON,
        "evidence_refs": evidence_refs,
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
    """Return the complete record set for one concern in exactly one model call."""
    template = load_record_template(identifier)
    normalized_context = task_context(template, context)
    allowed_refs = {str(ref) for ref in allowed_refs}
    schema = record_batch_response_schema(template)
    binding = "bounded-record-v2:" + task_binding(template, normalized_context, allowed_refs)
    saved = (progress or {}).get(binding)

    if saved is None:
        rules = "\n".join(str(rule) for rule in template.get("rules", ()))
        system_prompt = (
            str(template.get("task") or "Produce the requested records.")
            + ("\n" + rules if rules else "")
            + "\nReturn the complete record set supported by this narrowed context in this single call. "
              "Return an empty records array when the concern has no authored/applicable record. "
              "Do not emit continuation, done, applicability, retry, or loop-control decisions. "
              "Set blocked_reason only when a missing fact makes a correct result impossible."
        )
        value = generate_fixed_template_value(
            router,
            "planner",
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {**normalized_context, "allowed_evidence_refs": sorted(allowed_refs)},
                        ensure_ascii=False,
                    ),
                },
            ],
            response_schema=schema,
            enable_tools=False,
            tool_name="submit_" + identifier.replace("/", "_") + "_records",
        )
        if checkpoint is not None:
            checkpoint(binding, deepcopy(value))
    else:
        if not isinstance(saved, dict):
            raise ValueError(f"TEMPLATE_PROGRESS: expected object for {identifier}")
        value = deepcopy(saved)

    return normalize_bounded_record_response(
        identifier, template, value, allowed_refs
    )


run_record_template = run_bounded_record_template

__all__ = [
    "normalize_bounded_record_response",
    "record_batch_response_schema",
    "run_bounded_record_template",
    "run_record_template",
]
