from __future__ import annotations

"""Dedicated atomic worksheet chunker and host deterministic merger.

Small models (such as Qwen 3.5 9B) cannot reliably emit massive 4,000-character, 229-node
schemas in a single pass without grammar stalls or schema violations. This module breaks
down each engineering worksheet section into bounded, concern-level atomic chunks that
strictly satisfy model output atomicity limits. The host deterministically merges the
chunk outputs and enforces the canonical full-section specification.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any

from .model_output_atomicity_contract import assert_atomic_model_schema, is_atomic_model_schema
from .planning_detail_slots import DETAIL_RECORDS
from .planning_detail_template import (
    _normalize_section_name,
    _section_description,
    validate_worksheet_section,
)

_DEFAULT_CHUNK_SIZE = 2


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
    """Return a strictly closed, atomic JSON schema for a subset of concerns."""
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
                "properties": {
                    field: {"type": "string", "minLength": 1} for field in fields
                },
                "required": fields,
                "additionalProperties": False,
            },
        }

    properties["inapplicable_concerns"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "concern": {"type": "string", "enum": list(concerns)},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["concern", "reason"],
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
            "items": {"type": "string", "minLength": 1},
        }

    required = list(concerns) + ["inapplicable_concerns"]
    if include_evidence:
        required.append("constraint_evidence_refs")

    schema: dict[str, Any] = {
        "type": "object",
        "description": f"Atomic concern chunk for {key}: {', '.join(concerns)}",
        "properties": properties,
        "required": required,
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
    """Return compact, single-chunk instructions."""
    key = _normalize_section_name(section)
    schema = worksheet_chunk_schema(key, concerns, include_evidence=include_evidence)
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
            f"Fill exactly these concern arrays: {', '.join(concerns)}.{evidence_instruction}",
            "Every concern array must contain records with non-empty strings for all required fields.",
            "If a concern is genuinely inapplicable, leave its array empty and provide exactly one concrete reason in inapplicable_concerns.",
            "Do not output markdown, reasoning prose, or extra keys.",
            "Exact response schema: "
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        )
    )


def merge_worksheet_section_chunks(
    section: str,
    chunks: Sequence[Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[str, Any]:
    """Deterministically merge atomic chunk outputs into a canonical worksheet section."""
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
        chunk_dict = dict(chunk)
        if "specification" in chunk_dict and isinstance(chunk_dict["specification"], Mapping):
            spec = chunk_dict.pop("specification")
            chunk_dict.update(spec)
        for field, value in chunk_dict.items():
            if field == "constraint_evidence_refs":
                if isinstance(value, list):
                    for ref in value:
                        if isinstance(ref, str) and ref.strip() and ref not in evidence_refs:
                            evidence_refs.append(ref.strip())
            elif field == "inapplicable_concerns":
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, Mapping):
                            c = str(item.get("concern") or "")
                            reason = str(item.get("reason") or "")
                            if c and reason and not any(
                                existing.get("concern") == c
                                for existing in combined_inapplicable
                            ):
                                combined_inapplicable.append(
                                    {"concern": c, "reason": reason}
                                )
            elif field in records:
                if field in merged_specification:
                    raise ValueError(
                        f"DETAILED_PLAN_WORKSHEET: duplicate concern {field!r} across chunks in {key}"
                    )
                if not isinstance(value, list):
                    raise ValueError(
                        f"DETAILED_PLAN_WORKSHEET: concern {field!r} must be an array of records"
                    )
                merged_specification[field] = deepcopy(value)
            else:
                raise ValueError(
                    f"DETAILED_PLAN_WORKSHEET: undeclared field {field!r} in chunk for {key}"
                )

    # Ensure all declared concerns are present
    missing_concerns = set(records) - set(merged_specification)
    if missing_concerns:
        raise ValueError(
            f"DETAILED_PLAN_WORKSHEET: missing concerns in merged {key}: "
            + ", ".join(sorted(missing_concerns))
        )

    merged_specification["inapplicable_concerns"] = combined_inapplicable
    assembled_section = {
        "specification": merged_specification,
        "constraint_evidence_refs": evidence_refs,
    }

    # Canonical full validation
    return validate_worksheet_section(assembled_section, allowed_refs, key)


__all__ = [
    "merge_worksheet_section_chunks",
    "pack_section_concerns",
    "worksheet_chunk_prompt",
    "worksheet_chunk_schema",
]
