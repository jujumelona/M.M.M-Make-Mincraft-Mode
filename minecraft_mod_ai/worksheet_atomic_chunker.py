from __future__ import annotations

"""Dedicated atomic worksheet chunker and host deterministic merger.

The host owns both concern paging and oversized record-field paging. Model-facing
schemas always satisfy the global atomicity contract; the host deterministically
reassembles field fragments and validates the canonical worksheet section afterward.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any

from .model_output_atomicity_contract import (
    MAX_MODEL_FIELDS,
    _assert_closed_object_schemas,
    is_atomic_model_schema,
)
from .planning_detail_slots import DETAIL_RECORDS
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


def _field_pages(fields: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    width = max(1, int(MAX_MODEL_FIELDS))
    return tuple(
        tuple(fields[index : index + width])
        for index in range(0, len(fields), width)
    )


def pack_section_concerns(
    section: str,
    *,
    max_chunk_size: int | None = None,
) -> list[tuple[str, ...]]:
    """Pack concerns and oversized record fields into atomic model-facing chunks.

    Returned objects remain tuple-compatible for existing callers while carrying the
    exact host-selected record fields for each concern page.
    """
    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    if max_chunk_size is not None and max_chunk_size < 1:
        raise ValueError("max_chunk_size must be positive when supplied")

    pages: list[tuple[str, tuple[str, ...]]] = []
    for concern, columns in records.items():
        for fields in _field_pages(tuple(columns.split())):
            pages.append((concern, fields))

    chunks: list[tuple[str, ...]] = []
    position = 0
    while position < len(pages):
        outer_capacity = max(1, MAX_MODEL_FIELDS - (2 if not chunks else 1))
        if max_chunk_size is not None:
            outer_capacity = min(outer_capacity, max_chunk_size)

        selected: list[tuple[str, tuple[str, ...]]] = []
        used_concerns: set[str] = set()
        scan = position
        while scan < len(pages) and len(selected) < outer_capacity:
            concern, fields = pages[scan]
            if concern in used_concerns:
                break
            selected.append((concern, fields))
            used_concerns.add(concern)
            scan += 1

        if not selected:
            selected = [pages[position]]
            scan = position + 1

        chunk = WorksheetConcernChunk(
            [concern for concern, _ in selected],
            {concern: fields for concern, fields in selected},
        )
        schema = worksheet_chunk_schema(key, chunk, include_evidence=not chunks)
        if not is_atomic_model_schema(schema):
            if len(selected) != 1:
                concern, fields = selected[0]
                chunk = WorksheetConcernChunk((concern,), {concern: fields})
                schema = worksheet_chunk_schema(key, chunk, include_evidence=not chunks)
                scan = position + 1
            if not is_atomic_model_schema(schema):
                _assert_closed_object_schemas(
                    schema,
                    path=f"worksheet chunk {key}.{chunk[0]}",
                )
                raise ValueError(
                    f"DETAILED_PLAN_WORKSHEET_ATOMICITY: unable to atomize {key}.{chunk[0]}"
                )

        chunks.append(chunk)
        position = scan

    return chunks


def worksheet_chunk_schema(
    section: str,
    concerns: Sequence[str],
    *,
    include_evidence: bool = False,
) -> dict[str, Any]:
    """Return one bounded partial-record schema for a host-selected field page."""
    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    active = tuple(concerns)
    if not active:
        raise ValueError(f"worksheet chunk for {key!r} cannot be empty")
    projection = _chunk_projection(key, concerns)

    properties: dict[str, Any] = {}
    authored_signal: list[dict[str, Any]] = []
    for concern in active:
        fields = projection[concern]
        properties[concern] = {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    field: {"type": "string", "minLength": 1, "maxLength": 256}
                    for field in fields
                },
                "required": [],
                "minProperties": 1,
                "additionalProperties": False,
            },
        }
        authored_signal.append(
            {"required": [concern], "properties": {concern: {"minItems": 1}}}
        )

    properties["inapplicable_concerns"] = {
        "type": "array",
        "maxItems": 4,
        "items": {
            "type": "object",
            "properties": {
                "concern": {"type": "string", "enum": list(active)},
                "reason": {"type": "string", "minLength": 1, "maxLength": 256},
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
            "maxItems": 4,
            "uniqueItems": True,
            "description": (
                "Evidence references supplied by the host that constrain this authored design section. "
                "Use an empty array when the section is a design decision rather than an external fact."
            ),
            "items": {"type": "string", "minLength": 1, "maxLength": 256},
        }

    return {
        "type": "object",
        "description": f"Atomic concern chunk for {key}: {', '.join(active)}",
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
    projection = _chunk_projection(key, concerns)
    data = dict(chunk)
    nested = data.get("specification")
    if isinstance(nested, Mapping):
        data = dict(nested)

    for concern in active:
        if concern not in records:
            raise ValueError(f"Unknown concern {concern!r} for section {key!r}")
        value = data.get(concern)
        if isinstance(value, Mapping):
            candidates = [value]
        elif isinstance(value, list):
            candidates = value
        else:
            candidates = []
        for item in candidates:
            if isinstance(item, Mapping) and any(
                _meaningful_text(item.get(field)) for field in projection[concern]
            ):
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
) -> str:
    from .planning_contract_ssot import schema_skeleton_template

    key = _normalize_section_name(section)
    schema = worksheet_chunk_schema(key, concerns, include_evidence=include_evidence)
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
            f"ENGINEERING WORKSHEET — atomic concern chunk {chunk_index}/{chunk_count}:",
            f"Section: {key}",
            f"Active Concerns: {', '.join(concerns)}",
            f"Active Record Fields: {field_text}",
            f"Purpose: {_section_description(key)}",
            f"Fill only the shown fields for these concern arrays.{evidence_instruction}",
            "If a concern appears in another chunk, preserve record count and record order so the host can merge field pages deterministically.",
            "Prefer complete values for the shown fields, but do not invent external facts; the host normalizes harmless omissions.",
            "The chunk must contain at least one concrete concern record or one concrete inapplicable reason; evidence refs alone are not an answer.",
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
    expected_record_counts: dict[str, int] = {}
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
            prior_count = expected_record_counts.get(field)
            if prior_count is None:
                expected_record_counts[field] = count
            elif prior_count != count:
                raise ValueError(
                    "DETAILED_PLAN_WORKSHEET_FIELD_PAGE_COUNT: "
                    f"{key}.{field} changed record count from {prior_count} to {count}"
                )

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
        cleaned_records: list[dict[str, str]] = []
        for item in merged_specification[concern]:
            if not isinstance(item, Mapping):
                continue
            has_meaningful = any(
                _meaningful_text(item.get(field_name))
                for field_name in expected_fields
            )
            if not has_meaningful:
                continue
            clean_item: dict[str, str] = {}
            for field_name in expected_fields:
                val = str(item.get(field_name) or "").strip()
                if not val or val.casefold() in _PLACEHOLDERS:
                    val = _CANONICAL_FIELD_DEFAULTS.get(
                        field_name,
                        f"standard {field_name}",
                    )
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
