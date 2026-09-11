"""Host-owned design record execution with no model-controlled continuation loops."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .bounded_record_template import run_bounded_record_template
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

    records: list[dict[str, Any]] = []
    for index, requested_property in enumerate(required_properties):
        record = run_single_record_template(
            router,
            identifier,
            context={
                **normalized,
                "allowed_properties": [requested_property],
                "requested_property": requested_property,
                "record_index": index,
                "record_count": len(required_properties),
                "accepted_records": deepcopy(records),
            },
            progress=progress,
            checkpoint=checkpoint,
        )
        if record.get("property") != requested_property:
            raise TemplateBlocked(
                f"TEMPLATE_REQUIRED_PROPERTY_MISMATCH: expected {requested_property}, "
                f"received {record.get('property')}"
            )
        records.append(record)
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
    template = load_record_template("design/relation_set")
    try:
        values = template["record_schema"]["properties"]["relations"]["items"]["enum"]
    except (KeyError, TypeError) as exc:
        raise TemplateBlocked("TEMPLATE_RELATION_SCHEMA_INVALID") from exc
    if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
        raise TemplateBlocked("TEMPLATE_RELATION_SCHEMA_INVALID")
    return tuple(values)


def _run_relations(router, identifier, context, progress, checkpoint):
    template = load_record_template(identifier)
    normalized = task_context(template, context)
    entity_ids = normalized.get("entity_ids")
    if not isinstance(entity_ids, list):
        raise TemplateBlocked("TEMPLATE_RELATION_ENTITY_IDS_REQUIRED")
    relation_types = _relation_vocabulary()

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_id in entity_ids:
        for target_id in entity_ids:
            if source_id == target_id:
                continue
            decision = run_single_record_template(
                router,
                "design/relation_set",
                context={
                    **normalized,
                    "source_id": source_id,
                    "target_id": target_id,
                    "allowed_relation_types": list(relation_types),
                },
                progress=progress,
                checkpoint=checkpoint,
            )
            selected = decision.get("relations", [])
            if not isinstance(selected, list) or any(item not in relation_types for item in selected):
                raise TemplateBlocked(
                    f"TEMPLATE_RELATION_UNSUPPORTED: {source_id}->{target_id}: {selected}"
                )
            if len(selected) != len(set(selected)):
                raise TemplateBlocked(
                    f"TEMPLATE_RELATION_DUPLICATE: {source_id}->{target_id}: {selected}"
                )
            for relation_type in selected:
                record = {
                    "relation_type": relation_type,
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                }
                key = _record_key(record)
                if key in seen:
                    raise TemplateBlocked(f"TEMPLATE_RELATION_DUPLICATE: {key}")
                seen.add(key)
                records.append(record)

            key_code = int(decision.get("key_code", 0))
            if key_code:
                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                if not 1 <= key_code <= len(alphabet):
                    raise TemplateBlocked(
                        f"TEMPLATE_RELATION_KEY_INVALID: {source_id}->{target_id}: {key_code}"
                    )
                record = {
                    "relation_type": "key_" + alphabet[key_code - 1],
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                }
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
