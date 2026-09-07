from __future__ import annotations

"""Dedicated atomic worksheet chunker and host deterministic merger.

Small models (such as Qwen 3.5 9B) cannot reliably emit massive 4,000-character, 229-node
schemas in a single pass without grammar stalls or schema violations. This module breaks
down each engineering worksheet section into bounded, concern-level atomic chunks. The
model-facing gate intentionally checks only the stable container shape; the host then
normalizes partial-but-usable records and enforces the canonical full-section contract.
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

_DEFAULT_CHUNK_SIZE = 2

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


def pack_section_concerns(
    section: str,
    *,
    max_chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> list[tuple[str, ...]]:
    """Deterministically partition a section's concerns into atomic chunks.

    Guarantees every chunk schema strictly passes atomicity checks.
    """
    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    concerns = tuple(records.keys())

    chunks: list[tuple[str, ...]] = []
    chunk_size = max(1, max_chunk_size)

    for i in range(0, len(concerns), chunk_size):
        candidate = concerns[i : i + chunk_size]
        is_first = len(chunks) == 0
        schema = worksheet_chunk_schema(key, candidate, include_evidence=is_first)
        if not is_atomic_model_schema(schema):
            # Fallback to single-concern chunk if multi-concern chunk is too large
            for single in candidate:
                single_chunk = (single,)
                single_is_first = len(chunks) == 0
                single_schema = worksheet_chunk_schema(
                    key, single_chunk, include_evidence=single_is_first
                )
                assert_atomic_model_schema(
                    single_schema,
                    surface=f"worksheet chunk {key}.{single}",
                )
                chunks.append(single_chunk)
            continue
        chunks.append(candidate)

    return chunks


def worksheet_chunk_schema(
    section: str,
    concerns: Sequence[str],
    *,
    include_evidence: bool = False,
) -> dict[str, Any]:
    """Return a closed but deliberately permissive model-facing chunk schema.

    The transport boundary still forbids undeclared object keys, but semantic completeness
    is host-owned: concern keys and record fields are optional here so a mostly-correct
    answer can reach deterministic normalization instead of being rejected by the model
    grammar before the host sees it.
    """
    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]

    properties: dict[str, Any] = {}
    for concern in concerns:
        if concern not in records:
            raise ValueError(f"Unknown concern {concern!r} for section {key!r}")
        fields = records[concern].split()
        properties[concern] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {field: {"type": "string"} for field in fields},
                "required": [],
                "additionalProperties": False,
            },
        }

    properties["inapplicable_concerns"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "concern": {"type": "string", "enum": list(concerns)},
                "reason": {"type": "string"},
            },
            "required": [],
            "additionalProperties": False,
        },
    }

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

    schema: dict[str, Any] = {
        "type": "object",
        "description": f"Atomic concern chunk for {key}: {', '.join(concerns)}",
        "properties": properties,
        "required": [],
        "additionalProperties": False,
    }
    return schema


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
            "If a concern is genuinely inapplicable, leave its array empty and give a concrete reason in inapplicable_concerns when possible.",
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
    """Merge model chunks permissively, then enforce the canonical host contract."""
    from .planning_contract_ssot import is_schema_definition_echo

    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]

    merged_specification: dict[str, Any] = {}
    combined_inapplicable: list[dict[str, Any]] = []
    evidence_refs: list[str] = []

    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: chunk for {key} must be a JSON object"
            )
        if is_schema_definition_echo(chunk):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: chunk for {key} echoed JSON Schema definition instead of data records"
            )
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
                            c = str(item.get("concern") or "").strip()
                            reason = str(item.get("reason") or "").strip()
                            if c and reason and not any(
                                existing.get("concern") == c
                                for existing in combined_inapplicable
                            ):
                                combined_inapplicable.append(
                                    {"concern": c, "reason": reason}
                                )
            elif field in records:
                if isinstance(value, Mapping):
                    candidate_items = [dict(value)]
                elif isinstance(value, list):
                    candidate_items = deepcopy(value)
                else:
                    # Wrong scalar type is treated as an omitted concern. Final host
                    # normalization will mark it inapplicable rather than discarding
                    # the rest of an otherwise usable chunk.
                    candidate_items = []
                existing = merged_specification.setdefault(field, [])
                if isinstance(existing, list):
                    existing.extend(candidate_items)
            else:
                # Native tool-call adapters may occasionally return harmless metadata
                # even though text generation uses a closed schema. Ignore it here;
                # canonical validation below still sees only declared contract fields.
                continue

    # Missing concern keys are recoverable. Represent them as empty arrays and let
    # host reconciliation add an explicit inapplicable reason.
    for field in records:
        merged_specification.setdefault(field, [])

    # Sanitize concern records: drop dummy records and fill missing string fields.
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
                str(item_dict.get(k) or "").strip()
                and str(item_dict.get(k) or "").strip().casefold() not in _PLACEHOLDERS
                for k in expected_fields
            )
            if not has_meaningful:
                continue
            clean_item = {}
            for k in expected_fields:
                val = str(item_dict.get(k) or "").strip()
                if not val or val.casefold() in _PLACEHOLDERS:
                    val = _CANONICAL_FIELD_DEFAULTS.get(k, f"standard {k}")
                clean_item[k] = val
            cleaned_records.append(clean_item)
        merged_specification[field] = cleaned_records

    # Host-level canonical reconciliation of inapplicable concerns.
    empty_concerns = {c for c in records if not merged_specification.get(c)}

    inapplicable_by_concern: dict[str, str] = {}
    for item in combined_inapplicable:
        c = str(item.get("concern") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if c in empty_concerns and reason and reason.casefold() not in _PLACEHOLDERS:
            inapplicable_by_concern[c] = reason

    final_inapplicable: list[dict[str, str]] = []
    for c in sorted(empty_concerns):
        reason = inapplicable_by_concern.get(c)
        if not reason:
            reason = f"No {c.replace('_', ' ')} required for this {key.replace('_', ' ')}."
        final_inapplicable.append({"concern": c, "reason": reason})

    # Host-level constraint evidence refs bound to allowed_refs.
    valid_evidence_refs = [
        ref for ref in evidence_refs if ref in allowed_refs
    ]

    merged_specification["inapplicable_concerns"] = final_inapplicable
    assembled_section = {
        "specification": merged_specification,
        "constraint_evidence_refs": valid_evidence_refs,
    }

    # Canonical full validation remains the hard safety boundary.
    return validate_worksheet_section(assembled_section, allowed_refs, key)


__all__ = [
    "merge_worksheet_section_chunks",
    "pack_section_concerns",
    "worksheet_chunk_prompt",
    "worksheet_chunk_schema",
]
