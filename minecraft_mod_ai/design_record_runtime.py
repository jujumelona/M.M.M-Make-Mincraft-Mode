"""Host-owned design record execution with no model-controlled continuation loops."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .bounded_record_template import run_bounded_record_template
from .parallel_model_tasks import deterministic_model_map, serialized_callback
from .single_record_template import run_single_record_template
from .task_template_catalog import load_record_template
from .task_template_input import task_context
from .template_errors import TemplateBlocked

_EXACT_SINGLE = frozenset({"design/content_capability"})


def _evidence_refs(context: dict[str, Any], allowed_refs) -> list[str]:
    allowed = {str(ref) for ref in allowed_refs}
    refs: list[str] = []
    for key in ("source_evidence_id", "evidence_ref", "shard_id"):
        value = context.get(key)
        if isinstance(value, str) and value in allowed and value not in refs:
            refs.append(value)
    evidence = context.get("evidence")
    values = [evidence] if isinstance(evidence, str) else evidence
    if isinstance(values, (list, tuple, set)):
        for value in values:
            if isinstance(value, str) and value in allowed and value not in refs:
                refs.append(value)
    return refs


def _record_key(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _run_properties(router, identifier, context, progress, checkpoint):
    template = load_record_template(identifier)
    normalized = task_context(template, context)
    allowed = normalized.get("allowed_properties")
    required = normalized.get("required_properties")
    allowed_set = {str(name) for name in allowed} if isinstance(allowed, list) else set()
    required_properties = (
        list(dict.fromkeys(str(name) for name in required))
        if isinstance(required, list)
        else []
    )
    unknown = [name for name in required_properties if name not in allowed_set]
    if unknown:
        raise TemplateBlocked(f"TEMPLATE_REQUIRED_PROPERTY_UNKNOWN: {unknown}")

    # Required properties are host-selected independent slots. Feeding previously
    # generated property records into later calls created an artificial dependency,
    # grew every subsequent prompt, and forced otherwise independent model work into
    # a serial chain. Keep one immutable base context for every slot and let the
    # native model concurrency policy schedule only the model calls.
    safe_checkpoint = serialized_callback(checkpoint)
    jobs = tuple(enumerate(required_properties))

    def run_property(job):
        index, requested_property = job
        record = run_single_record_template(
            router,
            identifier,
            context={
                **normalized,
                "allowed_properties": [requested_property],
                "requested_property": requested_property,
                "record_index": index,
                "record_count": len(required_properties),
                "accepted_records": [],
            },
            progress=progress,
            checkpoint=safe_checkpoint,
        )
        if record.get("property") != requested_property:
            raise TemplateBlocked(
                f"TEMPLATE_REQUIRED_PROPERTY_MISMATCH: expected {requested_property}, "
                f"received {record.get('property')}"
            )
        return record

    records = deterministic_model_map(
        router,
        jobs,
        run_property,
        role="planner",
        thread_name_prefix="design-property",
    )
    return records, normalized


def _run_entities(router, identifier, context, progress, checkpoint):
    template = load_record_template(identifier)
    normalized = task_context(template, context)
    cardinality = run_single_record_template(
        router,
        "design/content_entity_count",
        context={**normalized, "accepted_records": []},
        progress=progress,
        checkpoint=checkpoint,
    )
    target_count = int(cardinality["count"])
    if target_count < 1:
        raise TemplateBlocked(f"TEMPLATE_ENTITY_CARDINALITY_INVALID: {target_count}")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(target_count):
        record = run_single_record_template(
            router,
            identifier,
            context={
                **normalized,
                "entity_ordinal": index + 1,
                "entity_count": target_count,
                "accepted_records": deepcopy(records),
            },
            progress=progress,
            checkpoint=checkpoint,
        )
        key = _record_key(record)
        if key in seen:
            raise TemplateBlocked(f"TEMPLATE_NO_PROGRESS: repeated record in {identifier}")
        seen.add(key)
        records.append(record)
    return records, normalized


def _relation_vocabulary() -> tuple[str, ...]:
    template = load_record_template("design/content_relation")
    try:
        values = template["record_schema"]["properties"]["relation_type"]["enum"]
    except (KeyError, TypeError) as exc:
        raise TemplateBlocked("TEMPLATE_RELATION_SCHEMA_INVALID") from exc
    if not isinstance(values, list) or not values or any(
        not isinstance(value, str) for value in values
    ):
        raise TemplateBlocked("TEMPLATE_RELATION_SCHEMA_INVALID")
    if len(values) != len(set(values)):
        raise TemplateBlocked("TEMPLATE_RELATION_SCHEMA_DUPLICATE")
    return tuple(values)


def _max_pair_relation_count(relation_types: tuple[str, ...]) -> int:
    ordinary = [value for value in relation_types if not value.startswith("key_")]
    key_values = [value for value in relation_types if value.startswith("key_")]
    if not ordinary or not key_values:
        raise TemplateBlocked("TEMPLATE_RELATION_SCHEMA_INVALID")
    return len(ordinary) + 1


def _run_relations(router, identifier, context, progress, checkpoint):
    template = load_record_template(identifier)
    normalized = task_context(template, context)
    entity_ids = normalized.get("entity_ids")
    if not isinstance(entity_ids, list):
        raise TemplateBlocked("TEMPLATE_RELATION_ENTITY_IDS_REQUIRED")
    if len(entity_ids) != len(set(entity_ids)):
        raise TemplateBlocked("TEMPLATE_RELATION_ENTITY_IDS_DUPLICATE")

    relation_types = _relation_vocabulary()
    max_pair_relations = _max_pair_relation_count(relation_types)
    allowed_relation_types = list(relation_types)
    pairs = tuple(
        (source_id, target_id)
        for source_id in entity_ids
        for target_id in entity_ids
        if source_id != target_id
    )
    safe_checkpoint = serialized_callback(checkpoint)

    def run_pair(pair):
        source_id, target_id = pair
        pair_context = {
            **normalized,
            "source_id": source_id,
            "target_id": target_id,
            "allowed_relation_types": allowed_relation_types,
        }
        cardinality = run_single_record_template(
            router,
            "design/content_relation_count",
            context={**pair_context, "accepted_records": []},
            progress=progress,
            checkpoint=safe_checkpoint,
        )
        target_count = int(cardinality["count"])
        if target_count < 0 or target_count > max_pair_relations:
            raise TemplateBlocked(
                "TEMPLATE_RELATION_CARDINALITY_INVALID: "
                f"{source_id}->{target_id}: {target_count} exceeds semantic maximum "
                f"{max_pair_relations}"
            )

        pair_records: list[dict[str, Any]] = []
        pair_relation_types: set[str] = set()
        key_relation_seen = False
        result: list[dict[str, Any]] = []
        for index in range(target_count):
            decision = run_single_record_template(
                router,
                identifier,
                context={
                    **pair_context,
                    "record_index": index,
                    "record_count": target_count,
                    "accepted_records": deepcopy(pair_records),
                },
                progress=progress,
                checkpoint=safe_checkpoint,
            )
            relation_type = decision.get("relation_type")
            if relation_type not in relation_types:
                raise TemplateBlocked(
                    f"TEMPLATE_RELATION_UNSUPPORTED: {source_id}->{target_id}: "
                    f"{relation_type}"
                )
            if relation_type in pair_relation_types:
                raise TemplateBlocked(
                    f"TEMPLATE_RELATION_DUPLICATE: {source_id}->{target_id}: "
                    f"{relation_type}"
                )
            if relation_type.startswith("key_"):
                if key_relation_seen:
                    raise TemplateBlocked(
                        f"TEMPLATE_RELATION_KEY_DUPLICATE: {source_id}->{target_id}"
                    )
                key_relation_seen = True

            pair_relation_types.add(relation_type)
            pair_records.append({"relation_type": relation_type})
            result.append(
                {
                    "relation_type": relation_type,
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                }
            )
        return result

    pair_results = deterministic_model_map(
        router,
        pairs,
        run_pair,
        role="planner",
        thread_name_prefix="design-relation-pair",
    )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair_records in pair_results:
        for record in pair_records:
            key = _record_key(record)
            if key in seen:
                raise TemplateBlocked(f"TEMPLATE_RELATION_DUPLICATE: {key}")
            seen.add(key)
            records.append(record)
    return records, normalized


def run_record_template(
    router,
    identifier: str,
    *,
    context: dict[str, Any],
    allowed_refs=(),
    progress=None,
    checkpoint=None,
):
    """Resolve one declared concern without model-owned continuation or termination."""
    template = load_record_template(identifier)
    normalized = task_context(template, context)

    if identifier in _EXACT_SINGLE:
        record = run_single_record_template(
            router,
            identifier,
            context=normalized,
            progress=progress,
            checkpoint=checkpoint,
        )
        return {
            "records": [record],
            "reason": "",
            "evidence_refs": _evidence_refs(normalized, allowed_refs),
        }

    if identifier == "design/content_property":
        records, normalized = _run_properties(
            router, identifier, normalized, progress, checkpoint
        )
    elif identifier == "design/content_entity":
        records, normalized = _run_entities(
            router, identifier, normalized, progress, checkpoint
        )
    elif identifier == "design/content_relation":
        records, normalized = _run_relations(
            router, identifier, normalized, progress, checkpoint
        )
    else:
        return run_bounded_record_template(
            router,
            identifier,
            context=normalized,
            allowed_refs=allowed_refs,
            progress=progress,
            checkpoint=checkpoint,
        )

    return {
        "records": records,
        "reason": "" if records else "No applicable records in the supplied context.",
        "evidence_refs": _evidence_refs(normalized, allowed_refs),
    }


__all__ = ["TemplateBlocked", "run_record_template"]