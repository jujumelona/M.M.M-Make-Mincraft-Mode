from __future__ import annotations

"""Worksheet concern chunking with host-owned semantic merge.

A concern is never split into independent field pages, so record index/count/order is
not used as cross-call identity. The host merges complete concern records and validates
the canonical worksheet section afterward.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any

from .model_output_atomicity_contract import _assert_closed_object_schemas
from .planning_detail_slots import DETAIL_RECORDS, record_field_schema
from .planning_detail_template import (
    _PLACEHOLDERS,
    _normalize_section_name,
    _section_description,
    validate_worksheet_section,
)

_CANONICAL_FIELD_DEFAULTS: dict[str, str] = {
    "authority": "server",
    "unit": "count",
    "range": "any",
    "default": "standard",
    "source": "environment",
    "visibility": "public",
    "side_effect": "state update",
    "rejection": "action denied without state change",
    "preserved_state": "all state preserved",
    "cooldown": "immediate (0 ticks)",
    "frequency": "on demand",
    "order": "sequential",
    "non_goal": "out of current requirement scope",
    "limit": "bounded by system memory and tick rate",
    "enforcement": "strict runtime assertion",
    "rollback": "restore prior snapshot",
    "commit": "atomic state commit",
    "time_bound": "under 1 tick (50ms)",
    "memory_bound": "bounded collection",
    "determinism": "deterministic calculation",
    "reentrancy_rule": "thread-safe / non-reentrant",
    "trust_boundary": "client-server boundary validation",
    "dirty_rule": "mark dirty on mutation",
}


class WorksheetConcernChunk(tuple):
    """Tuple-compatible concern selection carrying host-owned field projections."""

    def __new__(
        cls,
        concerns: Sequence[str],
        field_projection: Mapping[str, Sequence[str]],
    ) -> "WorksheetConcernChunk":
        obj = super().__new__(cls, tuple(concerns))
        obj.field_projection = {
            str(concern): tuple(str(field) for field in fields)
            for concern, fields in field_projection.items()
        }
        return obj


def _meaningful_text(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.casefold() not in _PLACEHOLDERS


def _chunk_projection(
    section: str,
    concerns: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    records = DETAIL_RECORDS[section]
    explicit = getattr(concerns, "field_projection", None)
    projection: dict[str, tuple[str, ...]] = {}
    for concern in concerns:
        if concern not in records:
            raise ValueError(f"Unknown concern {concern!r} for section {section!r}")
        all_fields = tuple(records[concern].split())
        selected = tuple(explicit.get(concern, all_fields)) if isinstance(explicit, Mapping) else all_fields
        if not selected or any(field not in all_fields for field in selected):
            raise ValueError(
                f"Invalid field projection for concern {concern!r} in section {section!r}"
            )
        projection[concern] = selected
    return projection


def pack_section_concerns(
    section: str,
    *,
    max_chunk_size: int | None = None,
) -> list[tuple[str, ...]]:
    """Pack complete concerns without splitting one record across model calls."""
    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    if max_chunk_size is not None and max_chunk_size < 1:
        raise ValueError("max_chunk_size must be positive when supplied")

    chunk_size = max_chunk_size or 1
    items = list(records.items())
    chunks: list[tuple[str, ...]] = []
    for start in range(0, len(items), chunk_size):
        selected = items[start : start + chunk_size]
        chunk = WorksheetConcernChunk(
            [concern for concern, _ in selected],
            {
                concern: tuple(columns.split())
                for concern, columns in selected
            },
        )
        schema = worksheet_chunk_schema(
            key,
            chunk,
            include_evidence=not chunks,
        )
        _assert_closed_object_schemas(
            schema,
            path=f"worksheet chunk {key}.{start // chunk_size + 1}",
        )
        chunks.append(chunk)
    return chunks


def worksheet_chunk_schema(
    section: str,
    concerns: Sequence[str],
    *,
    include_evidence: bool = False,
    record_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return one complete-concern schema.

    record_counts is accepted only for call-site compatibility and is intentionally
    ignored: a previous model response never becomes a cardinality contract.
    """
    del record_counts
    key = _normalize_section_name(section)
    active = tuple(concerns)
    if not active:
        raise ValueError(f"worksheet chunk for {key!r} cannot be empty")
    projection = _chunk_projection(key, concerns)

    properties: dict[str, Any] = {}
    authored_signal: list[dict[str, Any]] = []
    for concern in active:
        fields = projection[concern]
        field_schemas = {
            field: deepcopy(record_field_schema(key, concern, field))
            for field in fields
        }
        item_schema: dict[str, Any] = {
            "type": "object",
            "properties": field_schemas,
            "required": [],
            "minProperties": 1,
            "additionalProperties": False,
        }
        properties[concern] = {
            "type": "array",
            "items": item_schema,
        }
        authored_signal.append({"required": [concern]})

    properties["inapplicable_concerns"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "concern": {"type": "string", "enum": list(active)},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["concern", "reason"],
            "additionalProperties": False,
        },
    }
    authored_signal.append(
        {
            "required": ["inapplicable_concerns"],
            "properties": {"inapplicable_concerns": {"minItems": 1}},
        }
    )

    if include_evidence:
        properties["constraint_evidence_refs"] = {
            "type": "array",
            "uniqueItems": True,
            "description": (
                "Evidence references supplied by the host that constrain this authored design section. "
                "Use an empty array when the section is a design decision rather than an external fact."
            ),
            "items": {"type": "string", "minLength": 1},
        }

    return {
        "type": "object",
        "description": f"Complete concern chunk for {key}: {', '.join(active)}",
        "properties": properties,
        "required": [],
        "anyOf": authored_signal,
        "additionalProperties": False,
    }


def validate_worksheet_chunk_signal(
    section: str,
    concerns: Sequence[str],
    chunk: Mapping[str, Any],
) -> dict[str, Any]:
    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    active = tuple(concerns)
    data = dict(chunk)
    nested = data.get("specification")
    if isinstance(nested, Mapping):
        data = dict(nested)

    for concern in active:
        if concern not in records:
            raise ValueError(f"Unknown concern {concern!r} for section {key!r}")
        if concern in data:
            # Presence of an active concern key is sufficient transport signal.
            # The deterministic merge owns shape salvage, semantic cleanup and
            # applicability; malformed scalar values are normalized to no records.
            return dict(chunk)

    raw_inapplicable = data.get("inapplicable_concerns")
    candidates = (
        raw_inapplicable
        if isinstance(raw_inapplicable, list)
        else [raw_inapplicable]
        if isinstance(raw_inapplicable, Mapping)
        else []
    )
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        concern = str(item.get("concern") or "").strip()
        if concern in active and _meaningful_text(item.get("reason")):
            return dict(chunk)

    raise ValueError(
        "DETAILED_PLAN_WORKSHEET_CHUNK: "
        f"{key} concerns {', '.join(active)} contain no meaningful authored content"
    )


def worksheet_chunk_prompt(
    section: str,
    chunk_index: int,
    chunk_count: int,
    concerns: Sequence[str],
    *,
    include_evidence: bool = False,
    record_counts: Mapping[str, int] | None = None,
) -> str:
    from .planning_contract_ssot import schema_skeleton_template

    del record_counts
    key = _normalize_section_name(section)
    schema = worksheet_chunk_schema(
        key,
        concerns,
        include_evidence=include_evidence,
    )
    skeleton = schema_skeleton_template(schema)
    projection = _chunk_projection(key, concerns)
    field_text = "; ".join(
        f"{concern}=[{', '.join(fields)}]" for concern, fields in projection.items()
    )
    evidence_instruction = (
        " Also supply constraint_evidence_refs as an array of host-supplied evidence IDs (or empty array)."
        if include_evidence
        else ""
    )
    return "\n".join(
        (
            f"ENGINEERING WORKSHEET — concern chunk {chunk_index}/{chunk_count}:",
            f"Section: {key}",
            f"Active Concerns: {', '.join(concerns)}",
            f"Active Record Fields: {field_text}",
            f"Purpose: {_section_description(key)}",
            f"Fill the complete shown fields for these concern arrays.{evidence_instruction}",
            "Each concern is generated as one semantic unit; do not depend on record indices from another call.",
            "Prefer complete values for the shown fields, but do not invent external facts; the host normalizes harmless omissions.",
            "Return each active concern key. Use an empty array when no record applies; the host owns applicability reconciliation and does not require a model-authored reason.",
            "Never use N/A, none, TODO, TBD, unknown, same-as-above, or another placeholder as the authored content.",
            "DO NOT output JSON Schema keywords (never output 'type', 'properties', 'required', or 'additionalProperties').",
            "Return only a JSON object following this data template skeleton:",
            json.dumps(skeleton, ensure_ascii=False, indent=2),
        )
    )


def merge_worksheet_section_chunks(
    section: str,
    chunks: Sequence[Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[str, Any]:
    from .planning_contract_ssot import is_schema_definition_echo

    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    expected_chunks = pack_section_concerns(key)
    if len(chunks) != len(expected_chunks):
        raise ValueError(
            f"DETAILED_PLAN_WORKSHEET: {key} expected {len(expected_chunks)} chunks, got {len(chunks)}"
        )

    merged_specification: dict[str, list[dict[str, Any]]] = {
        concern: [] for concern in records
    }
    combined_inapplicable: list[dict[str, Any]] = []
    evidence_refs: list[str] = []

    for chunk, chunk_spec in zip(chunks, expected_chunks, strict=True):
        if not isinstance(chunk, Mapping):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: chunk for {key} must be a JSON object"
            )
        if is_schema_definition_echo(chunk):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: chunk for {key} echoed JSON Schema definition instead of data records"
            )
        validate_worksheet_chunk_signal(key, chunk_spec, chunk)
        projection = _chunk_projection(key, chunk_spec)
        chunk_dict = dict(chunk)
        if "specification" in chunk_dict and isinstance(chunk_dict["specification"], Mapping):
            spec = chunk_dict.pop("specification")
            chunk_dict.update(spec)

        for field, value in chunk_dict.items():
            if field == "constraint_evidence_refs":
                if isinstance(value, list):
                    for ref in value:
                        if isinstance(ref, str) and ref.strip() and ref.strip() not in evidence_refs:
                            evidence_refs.append(ref.strip())
                continue
            if field == "inapplicable_concerns":
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, Mapping):
                            concern = str(item.get("concern") or "").strip()
                            reason = str(item.get("reason") or "").strip()
                            if (
                                concern in records
                                and _meaningful_text(reason)
                                and not any(
                                    existing.get("concern") == concern
                                    for existing in combined_inapplicable
                                )
                            ):
                                combined_inapplicable.append(
                                    {"concern": concern, "reason": reason}
                                )
                continue
            if field not in records or field not in projection:
                continue

            if isinstance(value, Mapping):
                candidate_items = [dict(value)]
            elif isinstance(value, list):
                candidate_items = [dict(item) for item in value if isinstance(item, Mapping)]
            else:
                candidate_items = []

            count = len(candidate_items)
            merged_rows = merged_specification[field]
            while len(merged_rows) < count:
                merged_rows.append({})
            selected_fields = projection[field]
            for index, item in enumerate(candidate_items):
                destination = merged_rows[index]
                for field_name in selected_fields:
                    if field_name not in item:
                        continue
                    value_text = item[field_name]
                    if field_name in destination and destination[field_name] != value_text:
                        raise ValueError(
                            "DETAILED_PLAN_WORKSHEET_FIELD_PAGE_CONFLICT: "
                            f"{key}.{field}[{index}].{field_name} changed across pages"
                        )
                    destination[field_name] = value_text

    for concern, columns in records.items():
        expected_fields = tuple(columns.split())
        cleaned_records: list[dict[str, Any]] = []
        for item in merged_specification[concern]:
            if not isinstance(item, Mapping):
                continue
            has_meaningful = any(
                _meaningful_text(item.get(field_name))
                for field_name in expected_fields
            )
            if not has_meaningful:
                continue
            clean_item: dict[str, Any] = {}
            for field_name in expected_fields:
                raw_value = item.get(field_name)
                if (
                    key == "integration"
                    and concern == "initialization_order"
                    and field_name == "prerequisite"
                    and raw_value is None
                ):
                    clean_item[field_name] = "no prerequisite"
                    continue
                field_schema = record_field_schema(key, concern, field_name)
                raw_type = field_schema.get("type")
                array_capable = raw_type == "array" or (
                    isinstance(raw_type, list) and "array" in raw_type
                )
                if array_capable and isinstance(raw_value, list):
                    clean_item[field_name] = deepcopy(raw_value)
                    continue
                if raw_value is None and isinstance(raw_type, list) and "null" in raw_type:
                    clean_item[field_name] = None
                    continue
                val = str(raw_value or "").strip()
                if not val or val.casefold() in _PLACEHOLDERS:
                    if array_capable:
                        clean_item[field_name] = []
                    elif (
                        key == "integration"
                        and concern == "initialization_order"
                        and field_name == "prerequisite"
                    ):
                        clean_item[field_name] = "no prerequisite"
                    else:
                        clean_item[field_name] = _CANONICAL_FIELD_DEFAULTS.get(
                            field_name,
                            f"standard {field_name}",
                        )
                    continue
                clean_item[field_name] = val
            cleaned_records.append(clean_item)
        merged_specification[concern] = cleaned_records

    empty_concerns = {
        concern for concern in records if not merged_specification.get(concern)
    }
    inapplicable_by_concern: dict[str, str] = {}
    for item in combined_inapplicable:
        concern = str(item.get("concern") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if concern in empty_concerns and _meaningful_text(reason):
            inapplicable_by_concern[concern] = reason

    if not any(merged_specification[concern] for concern in records) and not inapplicable_by_concern:
        raise ValueError(
            f"DETAILED_PLAN_WORKSHEET: {key} contains no meaningful authored content"
        )

    final_inapplicable: list[dict[str, str]] = []
    for concern in sorted(empty_concerns):
        reason = inapplicable_by_concern.get(concern)
        if not reason:
            reason = f"No {concern.replace('_', ' ')} required for this {key.replace('_', ' ')}."
        final_inapplicable.append({"concern": concern, "reason": reason})

    valid_evidence_refs = [ref for ref in evidence_refs if ref in allowed_refs]
    assembled_specification: dict[str, Any] = dict(merged_specification)
    assembled_specification["inapplicable_concerns"] = final_inapplicable
    assembled_section = {
        "specification": assembled_specification,
        "constraint_evidence_refs": valid_evidence_refs,
    }
    return validate_worksheet_section(assembled_section, allowed_refs, key)


__all__ = [
    "WorksheetConcernChunk",
    "merge_worksheet_section_chunks",
    "pack_section_concerns",
    "validate_worksheet_chunk_signal",
    "worksheet_chunk_prompt",
    "worksheet_chunk_schema",
]
