"""Generate complete semantic record sets without model-owned loop control."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .design_generation_schema import context_bound_record_schema
from .execution_contract_policy import PLANNER_RECORD_COUNT_OUTPUT_TOKEN_CEILING
from .fixed_template_generation import generate_fixed_template_value
from .parallel_model_tasks import deterministic_model_map, serialized_callback
from .single_record_template import run_single_record_template
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context

_EMPTY_REASON = "No applicable records in the supplied context."
_MAX_RECORD_SET_ITEMS = 16


def _validated_minimum_count(minimum_count: int) -> int:
    if type(minimum_count) is not int or not 0 <= minimum_count <= _MAX_RECORD_SET_ITEMS:
        raise ValueError(
            f"TEMPLATE_RECORD_SET_MINIMUM: {minimum_count!r} is outside "
            f"0..{_MAX_RECORD_SET_ITEMS}"
        )
    return minimum_count


def record_cardinality_response_schema(
    template: dict[str, Any] | None = None,
    *,
    minimum_count: int = 0,
) -> dict[str, Any]:
    """Tiny semantic cardinality decision; the host owns all record iteration."""

    del template
    minimum = _validated_minimum_count(minimum_count)
    return {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "minimum": minimum,
                "maximum": _MAX_RECORD_SET_ITEMS,
                "enum": list(range(minimum, _MAX_RECORD_SET_ITEMS + 1)),
            },
        },
        "required": ["count"],
        "additionalProperties": False,
    }



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


def _record_count_messages(
    identifier: str,
    template: dict[str, Any],
    context: dict[str, Any],
    *,
    minimum_count: int = 0,
) -> list[dict[str, str]]:
    rules = "\n".join(str(rule) for rule in template.get("rules", ()))
    minimum_rule = (
        f"Return a count of at least {minimum_count}; the host has already proven "
        "that this many records are required. "
        if minimum_count > 0
        else "Return count 0 when no records apply. "
    )
    return [
        {
            "role": "system",
            "content": (
                "Choose only the number of distinct authored/applicable records required "
                "by the authoritative context. "
                + minimum_rule
                + "Do not author record content and do not emit completion, continuation, "
                "retry, ordinal, blocked, or other loop-control metadata.\n"
                + str(template.get("task") or "Determine required record cardinality.")
                + ("\n" + rules if rules else "")
            ),
        },
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False),
        },
    ]


def _record_key(record: dict[str, Any]) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _validate_record_semantics(
    record: dict[str, Any],
    *,
    identifier: str,
    schema: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    """Reject empty semantic fields even when JSON Schema minLength accepts spaces.

    This must run on both newly authored records and cached checkpoints, before
    acknowledging a completed record set. No malformed record may acquire a
    validated checkpoint merely because its string contains whitespace.
    """
    validator.validate(record)
    properties = schema.get("properties", {})
    for field in schema.get("required", ()):
        field_schema = properties.get(field, {})
        value = record.get(field)
        if (
            isinstance(field_schema, dict)
            and field_schema.get("type") == "string"
            and isinstance(value, str)
            and not value.strip()
        ):
            raise ValueError(
                f"TEMPLATE_RECORD_EMPTY_SEMANTIC_FIELD: {identifier}.{field}"
            )


def run_bounded_record_template(
    router,
    identifier: str,
    *,
    context: dict[str, Any],
    allowed_refs=(),
    progress=None,
    checkpoint=None,
    minimum_count: int = 0,
):
    """Generate a finite record set as tiny count + host-owned ordinal record jobs."""

    template = load_record_template(identifier)
    normalized_context = task_context(template, context)
    admitted_refs = {str(ref) for ref in allowed_refs}
    record_schema = context_bound_record_schema(
        identifier,
        template["record_schema"],
        normalized_context,
    )
    record_validator = Draft202012Validator(record_schema)
    minimum = _validated_minimum_count(minimum_count)
    binding = f"record-set-v3:min={minimum}:" + task_binding(
        template,
        normalized_context,
        admitted_refs,
    )
    saved = (progress or {}).get(binding)
    if saved is not None and not isinstance(saved, dict):
        raise ValueError(
            f"TEMPLATE_RECORD_SET_SAVED_SHAPE: {identifier} checkpoint must be an object"
        )
    if (
        isinstance(saved, dict)
        and "records" in saved
        and not isinstance(saved.get("records"), list)
    ):
        raise ValueError(
            f"TEMPLATE_RECORD_SET_SAVED_SHAPE: {identifier} records must be a list"
        )
    if (
        isinstance(saved, dict)
        and "count" in saved
        and type(saved.get("count")) is not int
    ):
        raise ValueError(
            f"TEMPLATE_RECORD_SET_SAVED_COUNT: {identifier} checkpoint count "
            "must be an integer"
        )
    if isinstance(saved, dict) and set(saved) not in (
        {"count"},
        {"count", "records"},
    ):
        raise ValueError(
            f"TEMPLATE_RECORD_SET_SAVED_SHAPE: {identifier} checkpoint fields "
            "must be exactly count or count+records"
        )

    if isinstance(saved, dict) and isinstance(saved.get("records"), list):
        raw_records = saved["records"]
        if any(not isinstance(item, dict) for item in raw_records):
            raise ValueError(
                f"TEMPLATE_RECORD_SET_SAVED_SHAPE: {identifier} contains a non-object record"
            )
        records = [deepcopy(item) for item in raw_records]
        if len(records) > _MAX_RECORD_SET_ITEMS:
            raise ValueError(
                f"TEMPLATE_RECORD_SET_COUNT: {identifier} saved {len(records)} records "
                f"exceeds {_MAX_RECORD_SET_ITEMS}"
            )
        if len(records) < minimum:
            raise ValueError(
                f"TEMPLATE_RECORD_SET_CARDINALITY_DRIFT: {identifier} requires at least "
                f"{minimum} records, saved {len(records)}"
            )
        expected = saved.get("count")
        if type(expected) is not int:
            raise ValueError(
                f"TEMPLATE_RECORD_SET_SAVED_COUNT: {identifier} checkpoint must "
                "contain an integer count with its records"
            )
        if expected != len(records):
            raise ValueError(
                f"TEMPLATE_RECORD_SET_CARDINALITY_DRIFT: expected {expected}, "
                f"saved {len(records)} records"
            )
        for record in records:
            _validate_record_semantics(
                record,
                identifier=identifier,
                schema=record_schema,
                validator=record_validator,
            )
    else:
        if isinstance(saved, dict) and type(saved.get("count")) is int:
            count = int(saved["count"])
        else:
            count_value = generate_fixed_template_value(
                router,
                "planner",
                _record_count_messages(
                    identifier,
                    template,
                    normalized_context,
                    minimum_count=minimum,
                ),
                response_schema=record_cardinality_response_schema(
                    template,
                    minimum_count=minimum,
                ),
                enable_tools=False,
                tool_name="submit_record_count_" + identifier.replace("/", "_"),
                output_token_ceiling=PLANNER_RECORD_COUNT_OUTPUT_TOKEN_CEILING,
            )
            count = int(count_value["count"])
            if checkpoint is not None:
                checkpoint(binding, {"count": count})

        if count < minimum or count > _MAX_RECORD_SET_ITEMS:
            raise ValueError(
                f"TEMPLATE_RECORD_SET_COUNT: {identifier} count {count} is outside "
                f"{minimum}..{_MAX_RECORD_SET_ITEMS}"
            )

        safe_checkpoint = serialized_callback(checkpoint)
        ordinals = tuple(range(count))

        def author_record(index: int) -> dict[str, Any]:
            return run_single_record_template(
                router,
                identifier,
                context={
                    **normalized_context,
                    "record_index": index,
                    "record_ordinal": index + 1,
                    "record_count": count,
                    # Ordinal identity makes sibling calls independent. The host checks
                    # distinctness after collection instead of growing later prompts.
                    "accepted_records": [],
                },
                progress=progress,
                checkpoint=safe_checkpoint,
            )

        records = deterministic_model_map(
            router,
            ordinals,
            author_record,
            role="planner",
            thread_name_prefix="bounded-record",
        )

        if len(records) != count:
            raise RuntimeError(
                f"TEMPLATE_RECORD_SET_CARDINALITY_DRIFT: expected {count}, "
                f"received {len(records)}"
            )
        for record in records:
            _validate_record_semantics(
                record,
                identifier=identifier,
                schema=record_schema,
                validator=record_validator,
            )
        if checkpoint is not None:
            checkpoint(
                binding,
                {
                    "count": count,
                    "records": deepcopy(records),
                },
            )

    keys = [_record_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(
            f"TEMPLATE_RECORD_SET_DUPLICATE: {identifier} returned duplicate "
            "ordinal records"
        )

    return {
        "records": records,
        "reason": "" if records else _EMPTY_REASON,
        "evidence_refs": _host_evidence_refs(normalized_context, admitted_refs),
    }


__all__ = [
    "record_cardinality_response_schema",
    "run_bounded_record_template",
]
