from __future__ import annotations

"""Dedicated atomic worksheet chunker and host deterministic merger.

Small models cannot reliably emit the complete engineering worksheet in one structured
response. The host therefore packs as many consecutive concerns as the global model
atomicity contract permits, while preserving a loose model-facing schema and a strict
canonical host validation boundary.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any

from .model_output_atomicity_contract import assert_atomic_model_schema, is_atomic_model_schema
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


def _meaningful_text(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.casefold() not in _PLACEHOLDERS


def pack_section_concerns(
    section: str,
    *,
    max_chunk_size: int | None = None,
) -> list[tuple[str, ...]]:
    """Pack the largest deterministic concern groups allowed by model atomicity.

    ``max_chunk_size`` is only an optional compatibility/test upper bound. Production
    packing has no arbitrary concern-count target: the global schema atomicity contract
    decides where each chunk must end.
    """
    key = _normalize_section_name(section)
    concerns = tuple(DETAIL_RECORDS[key].keys())
    if max_chunk_size is not None and max_chunk_size < 1:
        raise ValueError("max_chunk_size must be positive when supplied")

    chunks: list[tuple[str, ...]] = []
    position = 0
    while position < len(concerns):
        remaining = len(concerns) - position
        width_limit = remaining if max_chunk_size is None else min(remaining, max_chunk_size)
        chosen: tuple[str, ...] | None = None
        is_first = not chunks

        for width in range(width_limit, 0, -1):
            candidate = concerns[position : position + width]
            schema = worksheet_chunk_schema(
                key,
                candidate,
                include_evidence=is_first,
            )
            if is_atomic_model_schema(schema):
                chosen = candidate
                break

        if chosen is None:
            single = (concerns[position],)
            single_schema = worksheet_chunk_schema(
                key,
                single,
                include_evidence=is_first,
            )
            assert_atomic_model_schema(
                single_schema,
                surface=f"worksheet chunk {key}.{single[0]}",
            )
            chosen = single

        chunks.append(chosen)
        position += len(chosen)

    return chunks


def worksheet_chunk_schema(
    section: str,
    concerns: Sequence[str],
    *,
    include_evidence: bool = False,
) -> dict[str, Any]:
    """Return a loose partial-record schema with a minimum authored-content signal.

    Individual record fields remain optional so mostly-correct small-model output is not
    discarded. A chunk must nevertheless contain at least one non-empty concern record
    or one explicit inapplicable record; evidence refs alone cannot satisfy the chunk.
    """
    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    active = tuple(concerns)
    if not active:
        raise ValueError(f"worksheet chunk for {key!r} cannot be empty")

    properties: dict[str, Any] = {}
    authored_signal: list[dict[str, Any]] = []
    for concern in active:
        if concern not in records:
            raise ValueError(f"Unknown concern {concern!r} for section {key!r}")
        fields = records[concern].split()
        properties[concern] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    field: {"type": "string", "minLength": 1}
                    for field in fields
                },
                "required": [],
                "minProperties": 1,
                "additionalProperties": False,
            },
        }
        authored_signal.append(
            {
                "required": [concern],
                "properties": {concern: {"minItems": 1}},
            }
        )

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
            "items": {"type": "string"},
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
    """Reject only chunks that contain no meaningful authored design signal."""
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
        value = data.get(concern)
        if isinstance(value, Mapping):
            candidates = [value]
        elif isinstance(value, list):
            candidates = value
        else:
            candidates = []
        expected_fields = records[concern].split()
        for item in candidates:
            if isinstance(item, Mapping) and any(
                _meaningful_text(item.get(field)) for field in expected_fields
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
    """Return compact, single-chunk instructions with a concrete data skeleton."""
    from .planning_contract_ssot import schema_skeleton_template

    key = _normalize_section_name(section)
    schema = worksheet_chunk_schema(key, concerns, include_evidence=include_evidence)
    skeleton = schema_skeleton_template(schema)
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
            f"Purpose: {_section_description(key)}",
            f"Fill these concern arrays when applicable: {', '.join(concerns)}.{evidence_instruction}",
            "Prefer complete records, but do not invent facts just to fill a field; the host normalizes harmless omissions.",
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
    """Merge partial chunks permissively, then enforce the canonical host contract."""
    from .planning_contract_ssot import is_schema_definition_echo

    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    expected_chunks = pack_section_concerns(key)
    if len(chunks) != len(expected_chunks):
        raise ValueError(
            f"DETAILED_PLAN_WORKSHEET: {key} expected {len(expected_chunks)} chunks, got {len(chunks)}"
        )

    merged_specification: dict[str, Any] = {}
    combined_inapplicable: list[dict[str, Any]] = []
    evidence_refs: list[str] = []

    for chunk, active_concerns in zip(chunks, expected_chunks, strict=True):
        if not isinstance(chunk, Mapping):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: chunk for {key} must be a JSON object"
            )
        if is_schema_definition_echo(chunk):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: chunk for {key} echoed JSON Schema definition instead of data records"
            )
        validate_worksheet_chunk_signal(key, active_concerns, chunk)
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
            elif field == "inapplicable_concerns":
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
            elif field in records:
                if isinstance(value, Mapping):
                    candidate_items = [dict(value)]
                elif isinstance(value, list):
                    candidate_items = deepcopy(value)
                else:
                    candidate_items = []
                existing = merged_specification.setdefault(field, [])
                if isinstance(existing, list):
                    existing.extend(candidate_items)
            else:
                continue

    for field in records:
        merged_specification.setdefault(field, [])

    for field in records:
        raw_items = merged_specification.get(field)
        if not isinstance(raw_items, list):
            merged_specification[field] = []
            continue
        expected_fields = records[field].split()
        cleaned_records = []
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            item_dict = dict(item)
            has_meaningful = any(
                _meaningful_text(item_dict.get(field_name))
                for field_name in expected_fields
            )
            if not has_meaningful:
                continue
            clean_item = {}
            for field_name in expected_fields:
                val = str(item_dict.get(field_name) or "").strip()
                if not val or val.casefold() in _PLACEHOLDERS:
                    val = _CANONICAL_FIELD_DEFAULTS.get(
                        field_name,
                        f"standard {field_name}",
                    )
                clean_item[field_name] = val
            cleaned_records.append(clean_item)
        merged_specification[field] = cleaned_records

    empty_concerns = {c for c in records if not merged_specification.get(c)}
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
    merged_specification["inapplicable_concerns"] = final_inapplicable
    assembled_section = {
        "specification": merged_specification,
        "constraint_evidence_refs": valid_evidence_refs,
    }
    return validate_worksheet_section(assembled_section, allowed_refs, key)


__all__ = [
    "merge_worksheet_section_chunks",
    "pack_section_concerns",
    "validate_worksheet_chunk_signal",
    "worksheet_chunk_prompt",
    "worksheet_chunk_schema",
]
