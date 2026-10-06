"""Host-owned design record execution with no model-controlled continuation loops."""
from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator

from .bounded_record_template import run_bounded_record_template
from .content_design_contract import relation_type_supported_for_content_pair
from .fixed_template_generation import generate_fixed_template_value
from .model_output_atomicity_contract import structured_output_token_ceiling
from .parallel_model_tasks import deterministic_model_map, serialized_callback
from .single_record_template import run_single_record_template
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context
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


_MAX_DECISION_SLOTS = 16


def _decision_slot_selection_schema(allowed_slots: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "slots": {
                "type": "array",
                "minItems": 0,
                "maxItems": min(len(allowed_slots), _MAX_DECISION_SLOTS),
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": list(allowed_slots),
                },
            },
        },
        "required": ["slots"],
        "additionalProperties": False,
    }


def _run_decisions(router, identifier, context, progress, checkpoint):
    """Select semantic decision slots once, then let the host own slot identity."""

    template = load_record_template(identifier)
    normalized = task_context(template, context)
    raw_allowed = normalized.get("allowed_slots")
    if not isinstance(raw_allowed, list):
        raise TemplateBlocked("TEMPLATE_DECISION_ALLOWED_SLOTS_REQUIRED")
    allowed_slots = list(
        dict.fromkeys(
            value
            for value in raw_allowed
            if isinstance(value, str) and value.strip()
        )
    )
    if not allowed_slots:
        return [], normalized

    selector_schema = _decision_slot_selection_schema(allowed_slots)
    binding = "decision-slots-v1:" + task_binding(template, normalized, ())
    saved = (progress or {}).get(binding)
    if isinstance(saved, dict) and isinstance(saved.get("slots"), list):
        selected_slots = list(saved["slots"])
    else:
        selected = generate_fixed_template_value(
            router,
            "planner",
            (
                {
                    "role": "system",
                    "content": (
                        "Select the distinct design slots that require an authored decision "
                        "for this requirement. Choose only from allowed_slots. Return an empty "
                        "list when no listed slot requires a decision. Do not author values, "
                        "ordinals, counts, retries, or continuation metadata."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(normalized, ensure_ascii=False),
                },
            ),
            response_schema=selector_schema,
            enable_tools=False,
            description="Select applicable design decision slots.",
            output_token_ceiling=structured_output_token_ceiling(selector_schema),
        )
        selected_slots = list(selected["slots"])
        if checkpoint is not None:
            checkpoint(binding, {"slots": list(selected_slots)})

    if (
        len(selected_slots) != len(set(selected_slots))
        or len(selected_slots) > _MAX_DECISION_SLOTS
        or any(slot not in allowed_slots for slot in selected_slots)
    ):
        raise TemplateBlocked("TEMPLATE_DECISION_SLOT_SELECTION_INVALID")

    safe_checkpoint = serialized_callback(checkpoint)
    jobs = tuple(enumerate(selected_slots))

    def run_slot(job):
        index, slot_id = job
        record = run_single_record_template(
            router,
            identifier,
            context={
                **normalized,
                "allowed_slots": [slot_id],
                "record_index": index,
                "record_ordinal": index + 1,
                "record_count": len(selected_slots),
                "accepted_records": [],
            },
            progress=progress,
            checkpoint=safe_checkpoint,
        )
        if record.get("slot_id") != slot_id:
            raise TemplateBlocked(
                f"TEMPLATE_DECISION_SLOT_MISMATCH: expected {slot_id}, "
                f"received {record.get('slot_id')}"
            )
        return record

    records = deterministic_model_map(
        router,
        jobs,
        run_slot,
        role="planner",
        thread_name_prefix="design-decision-slot",
    )
    return records, normalized


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
    from .content_design_contract import CONTENT_KINDS

    template = load_record_template(identifier)
    normalized = task_context(template, context)
    # Cardinality is selected before the entity schema is used. Give that call
    # the same concrete vocabulary; its count-only response schema cannot carry
    # the kind enum that the later record-authoring calls receive.
    normalized.setdefault("allowed_content_kinds", list(CONTENT_KINDS))
    minimum = normalized.get("minimum_entity_count", 0)
    if type(minimum) is not int or minimum not in {0, 1}:
        raise TemplateBlocked("CONTENT_MINIMUM_ENTITY_COUNT_INVALID")
    result = run_bounded_record_template(
        router,
        identifier,
        context=normalized,
        progress=progress,
        checkpoint=checkpoint,
        minimum_count=minimum,
    )
    return list(result["records"]), normalized


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


def _relation_selector(
    router,
    template,
    normalized,
    *,
    selector_id: str,
    selector_context: dict[str, Any],
    response_schema: dict[str, Any],
    progress,
    checkpoint,
    instruction: str,
) -> dict[str, Any]:
    """Run one finite semantic selector whose cardinality is owned by the host."""

    validator = Draft202012Validator(response_schema)
    binding_context = {
        "selector_id": selector_id,
        "selector_context": selector_context,
    }
    binding = "relation-selector-v1:" + task_binding(template, binding_context, ())
    saved = (progress or {}).get(binding)
    if saved is not None:
        validator.validate(saved)
        return dict(saved)

    value = generate_fixed_template_value(
        router,
        "planner",
        (
            {
                "role": "system",
                "content": (
                    instruction
                    + " The host owns cardinality, endpoints, ordering, and termination. "
                    "Choose only identifiers present in the supplied candidate lists."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(selector_context, ensure_ascii=False),
            },
        ),
        response_schema=response_schema,
        enable_tools=False,
        description="Resolve one host-bounded content resource relation plan.",
        output_token_ceiling=structured_output_token_ceiling(response_schema),
    )
    validator.validate(value)
    if checkpoint is not None:
        checkpoint(binding, dict(value))
    return dict(value)


def _resource_source_relations(
    router,
    template,
    normalized,
    *,
    entity_kind_by_id: dict[str, str],
    entity_by_id: dict[str, dict[str, Any]],
    progress,
    checkpoint,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Resolve recipe/tag source graphs with source-level cardinality invariants."""

    item_ids = [
        entity_id
        for entity_id, kind in entity_kind_by_id.items()
        if kind == "item"
    ]
    resource_kinds = {"crafting_recipe", "smelting_recipe", "registry_tag"}
    resource_sources = {
        entity_id
        for entity_id, kind in entity_kind_by_id.items()
        if kind in resource_kinds
    }
    records: list[dict[str, Any]] = []

    base_context = {
        "requirements": normalized.get("requirements", []),
        "design_contexts": normalized.get("design_contexts", []),
    }

    for source_id in normalized.get("entity_ids", []):
        source_kind = entity_kind_by_id[source_id]
        if source_kind not in resource_kinds:
            continue
        source_entity = entity_by_id[source_id]
        selector_base = {
            **base_context,
            "source_entity": source_entity,
        }

        if source_kind == "crafting_recipe":
            if not item_ids:
                raise TemplateBlocked(
                    f"TEMPLATE_RESOURCE_RELATION_TARGET_REQUIRED: {source_id}: item"
                )
            schema = {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["shaped", "shapeless"],
                    },
                    "output_id": {
                        "type": "string",
                        "enum": list(item_ids),
                    },
                    "input_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": min(9, len(item_ids)),
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": list(item_ids),
                        },
                    },
                },
                "required": ["mode", "output_id", "input_ids"],
                "additionalProperties": False,
            }
            value = _relation_selector(
                router,
                template,
                normalized,
                selector_id=f"crafting:{source_id}",
                selector_context={
                    **selector_base,
                    "candidate_item_ids": list(item_ids),
                },
                response_schema=schema,
                progress=progress,
                checkpoint=checkpoint,
                instruction=(
                    "Resolve the crafting recipe semantics for this concrete recipe. "
                    "Choose exactly one output item, one to nine distinct ingredient "
                    "entities, and whether the authored recipe is shaped or shapeless."
                ),
            )
            records.append(
                {
                    "relation_type": "produces",
                    "source_id": source_id,
                    "target_id": value["output_id"],
                }
            )
            if value["mode"] == "shapeless":
                for target_id in value["input_ids"]:
                    records.append(
                        {
                            "relation_type": "consumes",
                            "source_id": source_id,
                            "target_id": target_id,
                        }
                    )
            else:
                for index, target_id in enumerate(value["input_ids"]):
                    records.append(
                        {
                            "relation_type": f"key_{chr(ord('A') + index)}",
                            "source_id": source_id,
                            "target_id": target_id,
                        }
                    )

        elif source_kind == "smelting_recipe":
            if not item_ids:
                raise TemplateBlocked(
                    f"TEMPLATE_RESOURCE_RELATION_TARGET_REQUIRED: {source_id}: item"
                )
            schema = {
                "type": "object",
                "properties": {
                    "output_id": {
                        "type": "string",
                        "enum": list(item_ids),
                    },
                    "ingredient_id": {
                        "type": "string",
                        "enum": list(item_ids),
                    },
                },
                "required": ["output_id", "ingredient_id"],
                "additionalProperties": False,
            }
            value = _relation_selector(
                router,
                template,
                normalized,
                selector_id=f"smelting:{source_id}",
                selector_context={
                    **selector_base,
                    "candidate_item_ids": list(item_ids),
                },
                response_schema=schema,
                progress=progress,
                checkpoint=checkpoint,
                instruction=(
                    "Resolve this cooking recipe with exactly one input item and "
                    "exactly one output item."
                ),
            )
            records.extend(
                (
                    {
                        "relation_type": "consumes",
                        "source_id": source_id,
                        "target_id": value["ingredient_id"],
                    },
                    {
                        "relation_type": "produces",
                        "source_id": source_id,
                        "target_id": value["output_id"],
                    },
                )
            )

        else:
            candidates_by_kind = {
                kind: [
                    entity_id
                    for entity_id, candidate_kind in entity_kind_by_id.items()
                    if candidate_kind == kind
                ]
                for kind in ("item", "block", "entity")
            }
            candidates_by_kind = {
                kind: ids for kind, ids in candidates_by_kind.items() if ids
            }
            if not candidates_by_kind:
                raise TemplateBlocked(
                    f"TEMPLATE_RESOURCE_RELATION_TARGET_REQUIRED: {source_id}: tag member"
                )
            kind_schema = {
                "type": "object",
                "properties": {
                    "member_kind": {
                        "type": "string",
                        "enum": list(candidates_by_kind),
                    },
                },
                "required": ["member_kind"],
                "additionalProperties": False,
            }
            kind_value = _relation_selector(
                router,
                template,
                normalized,
                selector_id=f"tag-kind:{source_id}",
                selector_context={
                    **selector_base,
                    "candidate_member_kinds": list(candidates_by_kind),
                },
                response_schema=kind_schema,
                progress=progress,
                checkpoint=checkpoint,
                instruction=(
                    "Choose the single Minecraft registry member kind grouped by "
                    "this tag."
                ),
            )
            member_kind = kind_value["member_kind"]
            candidate_ids = candidates_by_kind[member_kind]
            members_schema = {
                "type": "object",
                "properties": {
                    "member_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": len(candidate_ids),
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": list(candidate_ids),
                        },
                    },
                },
                "required": ["member_ids"],
                "additionalProperties": False,
            }
            members_value = _relation_selector(
                router,
                template,
                normalized,
                selector_id=f"tag-members:{source_id}:{member_kind}",
                selector_context={
                    **selector_base,
                    "member_kind": member_kind,
                    "candidate_member_ids": list(candidate_ids),
                },
                response_schema=members_schema,
                progress=progress,
                checkpoint=checkpoint,
                instruction=(
                    "Choose one or more concrete members of this registry tag from "
                    "the fixed member-kind candidate set."
                ),
            )
            for target_id in members_value["member_ids"]:
                records.append(
                    {
                        "relation_type": "contains",
                        "source_id": source_id,
                        "target_id": target_id,
                    }
                )

    return records, resource_sources


def _run_relations(router, identifier, context, progress, checkpoint):
    template = load_record_template(identifier)
    normalized = task_context(template, context)
    entity_ids = normalized.get("entity_ids")
    if not isinstance(entity_ids, list):
        raise TemplateBlocked("TEMPLATE_RELATION_ENTITY_IDS_REQUIRED")
    if len(entity_ids) != len(set(entity_ids)):
        raise TemplateBlocked("TEMPLATE_RELATION_ENTITY_IDS_DUPLICATE")

    relation_types = _relation_vocabulary()
    raw_entities = normalized.get("entities")
    if not isinstance(raw_entities, list):
        raise TemplateBlocked("TEMPLATE_RELATION_ENTITIES_REQUIRED")
    entity_kind_by_id: dict[str, str] = {}
    entity_by_id: dict[str, dict[str, Any]] = {}
    for entity in raw_entities:
        if not isinstance(entity, dict):
            raise TemplateBlocked("TEMPLATE_RELATION_ENTITY_INVALID")
        entity_id = entity.get("entity_id")
        kind = entity.get("kind")
        if not isinstance(entity_id, str) or not isinstance(kind, str):
            raise TemplateBlocked("TEMPLATE_RELATION_ENTITY_INVALID")
        if entity_id in entity_kind_by_id and entity_kind_by_id[entity_id] != kind:
            raise TemplateBlocked(f"TEMPLATE_RELATION_ENTITY_KIND_CONFLICT: {entity_id}")
        entity_kind_by_id[entity_id] = kind
        entity_by_id[entity_id] = dict(entity)
    if set(entity_ids) != set(entity_kind_by_id):
        raise TemplateBlocked("TEMPLATE_RELATION_ENTITY_CONTEXT_MISMATCH")

    safe_checkpoint = serialized_callback(checkpoint)
    resource_records, resource_sources = _resource_source_relations(
        router,
        template,
        normalized,
        entity_kind_by_id=entity_kind_by_id,
        entity_by_id=entity_by_id,
        progress=progress,
        checkpoint=safe_checkpoint,
    )
    pairs = tuple(
        (source_id, target_id)
        for source_id in entity_ids
        for target_id in entity_ids
        if source_id != target_id and source_id not in resource_sources
    )

    def run_pair(pair):
        source_id, target_id = pair
        allowed_relation_types = [
            relation_type
            for relation_type in relation_types
            if relation_type_supported_for_content_pair(
                relation_type,
                entity_kind_by_id[source_id],
                entity_kind_by_id[target_id],
            )
        ]
        # An impossible pair is a host-known empty relation set. Do not ask the
        # model to choose cardinality or expose the unbound full relation schema.
        if not allowed_relation_types:
            return []

        pair_context = {
            **normalized,
            "source_id": source_id,
            "target_id": target_id,
            "allowed_relation_types": allowed_relation_types,
        }
        batch = run_bounded_record_template(
            router,
            identifier,
            context=pair_context,
            progress=progress,
            checkpoint=safe_checkpoint,
        )

        pair_relation_types: set[str] = set()
        key_relation_seen = False
        result: list[dict[str, Any]] = []
        for decision in batch["records"]:
            relation_type = decision.get("relation_type")
            if relation_type not in allowed_relation_types:
                raise TemplateBlocked(
                    f"TEMPLATE_RELATION_UNSUPPORTED: {source_id}->{target_id}: "
                    f"{relation_type}"
                )
            if relation_type in pair_relation_types:
                continue
            if relation_type.startswith("key_"):
                if key_relation_seen:
                    raise TemplateBlocked(
                        f"TEMPLATE_RELATION_KEY_DUPLICATE: {source_id}->{target_id}"
                    )
                key_relation_seen = True

            pair_relation_types.add(relation_type)
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
    for record in resource_records:
        key = _record_key(record)
        if key not in seen:
            seen.add(key)
            records.append(record)
    for pair_records in pair_results:
        for record in pair_records:
            key = _record_key(record)
            if key in seen:
                continue
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

    if identifier == "design/decision":
        records, normalized = _run_decisions(
            router, identifier, normalized, progress, checkpoint
        )
    elif identifier == "design/content_property":
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
