"""Host-owned cardinality for record templates.

The model never returns an arbitrarily capped record array and never controls
continuation. It first determines the authored cardinality as one integer; the
host then performs exactly that many single-record calls in stable ordinal order.
Independent ordinals may occupy measured native llama slots concurrently.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
from threading import RLock
from typing import Any

from jsonschema import Draft202012Validator

from .fixed_template_generation import generate_fixed_template_value
from .model_concurrency import router_native_model_parallelism
from .model_output_atomicity_contract import MAX_MODEL_STRING_CHARS
from .single_record_template import run_single_record_template
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context
from .template_errors import TemplateBlocked

_EMPTY_REASON = "No applicable records in the supplied context."


def _cardinality_blocking_enabled(template: dict[str, Any] | None) -> bool:
    """Default to the historical blocking contract unless the template opts out."""
    return not (template and template.get("cardinality_blocking") is False)


def record_cardinality_response_schema(template: dict[str, Any] | None = None) -> dict[str, Any]:
    """Small-model-safe schema with no arbitrary semantic cardinality ceiling."""
    properties: dict[str, Any] = {
        "count": {"type": "integer", "minimum": 0},
    }
    required = ["count"]
    if _cardinality_blocking_enabled(template):
        properties["blocked_reason"] = {
            "type": "string",
            "maxLength": MAX_MODEL_STRING_CHARS,
        }
        required.append("blocked_reason")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# Compatibility name for callers that inspect the active record-control schema.
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


def _load_cardinality(
    router: Any,
    identifier: str,
    template: dict[str, Any],
    context: dict[str, Any],
    *,
    allowed_refs: set[str],
    progress: dict[str, Any] | None,
    checkpoint: Any,
) -> int:
    schema = record_cardinality_response_schema(template)
    blocking_enabled = _cardinality_blocking_enabled(template)
    binding = "record-cardinality-v1:" + task_binding(template, context, allowed_refs)
    saved = (progress or {}).get(binding)
    if saved is None:
        rules = "\n".join(str(rule) for rule in template.get("rules", ()))
        blocking_instruction = (
            "Set blocked_reason only when a missing fact makes the cardinality impossible "
            "to determine correctly."
            if blocking_enabled
            else (
                "The supplied context is authoritative and sufficient to determine this "
                "cardinality. Do not make a blocked/missing-fact decision; return count 0 "
                "when no records apply."
            )
        )
        system_prompt = (
            str(template.get("task") or "Produce the requested records.")
            + ("\n" + rules if rules else "")
            + "\nDetermine only the exact number of distinct authored/applicable records "
            "supported by this narrowed context. Return count 0 when none apply. Do not "
            "clamp the count to an implementation limit and do not make a continuation, "
            "done, retry, or loop-control decision. "
            + blocking_instruction
        )
        value = generate_fixed_template_value(
            router,
            "planner",
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            response_schema=schema,
            enable_tools=False,
            tool_name="submit_" + identifier.replace("/", "_") + "_count",
        )
        Draft202012Validator(schema).validate(value)
        if checkpoint is not None:
            checkpoint(binding, deepcopy(value))
    else:
        if not isinstance(saved, dict):
            raise ValueError(f"TEMPLATE_PROGRESS: expected cardinality object for {identifier}")
        value = deepcopy(saved)
        Draft202012Validator(schema).validate(value)

    blocked_reason = str(value.get("blocked_reason", "")).strip()
    if blocked_reason:
        raise TemplateBlocked(f"TEMPLATE_BLOCKED: {identifier}: {blocked_reason}")
    return int(value["count"])


def run_bounded_record_template(
    router,
    identifier: str,
    *,
    context: dict[str, Any],
    allowed_refs=(),
    progress=None,
    checkpoint=None,
):
    """Resolve one concern with host-owned exact cardinality and stable ordinals."""
    template = load_record_template(identifier)
    normalized_context = task_context(template, context)
    admitted_refs = {str(ref) for ref in allowed_refs}
    checkpoint_lock = RLock()

    def serialized_checkpoint(*args, **kwargs):
        if checkpoint is None:
            return None
        with checkpoint_lock:
            return checkpoint(*args, **kwargs)

    effective_checkpoint = serialized_checkpoint if checkpoint is not None else None
    count = _load_cardinality(
        router,
        identifier,
        template,
        normalized_context,
        allowed_refs=admitted_refs,
        progress=progress,
        checkpoint=effective_checkpoint,
    )

    def generate_record(index: int) -> dict[str, Any]:
        return run_single_record_template(
            router,
            identifier,
            context={
                **normalized_context,
                "record_index": index,
                "record_ordinal": index + 1,
                "record_count": count,
            },
            progress=progress,
            checkpoint=effective_checkpoint,
            generator=generate_fixed_template_value,
        )

    workers = max(1, min(count, router_native_model_parallelism(router))) if count else 1
    if count <= 1 or workers == 1:
        records = [generate_record(index) for index in range(count)]
    else:
        contexts = [copy_context() for _ in range(count)]
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="planning-template-record-ordinal",
        ) as pool:
            futures = [
                pool.submit(contexts[index].run, generate_record, index)
                for index in range(count)
            ]
            try:
                records = [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    seen: set[str] = set()
    for index, record in enumerate(records):
        key = json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if key in seen:
            raise TemplateBlocked(
                f"TEMPLATE_NO_PROGRESS: duplicate record in {identifier} at ordinal {index + 1}"
            )
        seen.add(key)

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
