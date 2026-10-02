from __future__ import annotations

"""Canonical structured authored-design contract.

The model authors fixed concern records. Markdown is a deterministic projection for
human review and provenance only; executable production consumes the records.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .planning_detail_slots import DETAIL_RECORDS
from .planning_detail_template import WORKSHEET_SECTIONS, worksheet_section_schema


def normalize_structured_sections(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("AUTHORED_STRUCTURED_DESIGN: sections must be an object")
    unknown = set(raw) - set(WORKSHEET_SECTIONS)
    if unknown:
        raise ValueError(
            "AUTHORED_STRUCTURED_DESIGN: unknown sections: "
            + ", ".join(sorted(str(item) for item in unknown))
        )

    normalized: dict[str, Any] = {}
    for section in WORKSHEET_SECTIONS:
        if section not in raw:
            continue
        value = raw[section]
        if not isinstance(value, Mapping):
            raise ValueError(
                f"AUTHORED_STRUCTURED_DESIGN: {section} must be an object"
            )
        error = next(
            Draft202012Validator(worksheet_section_schema(section)).iter_errors(value),
            None,
        )
        if error is not None:
            path = ".".join(str(part) for part in error.absolute_path)
            raise ValueError(
                f"AUTHORED_STRUCTURED_DESIGN: {section}.{path}: {error.message}"
            )
        normalized[section] = deepcopy(dict(value))
    return normalized


def structured_sections_sha256(
    sections: Mapping[str, Any] | None,
) -> str:
    """Hash the canonical structured authored-design authority.

    Producers, consumers, and tests must use this helper so normalization and
    serialization cannot drift independently.
    """
    normalized = normalize_structured_sections(sections)
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def active_concern_records(
    sections: Mapping[str, Any] | None,
    section: str,
) -> dict[str, list[dict[str, Any]]]:
    normalized = normalize_structured_sections(sections)
    row = normalized.get(section)
    if not isinstance(row, Mapping):
        return {}
    specification = row.get("specification")
    if not isinstance(specification, Mapping):
        return {}

    result: dict[str, list[dict[str, Any]]] = {}
    for concern in DETAIL_RECORDS.get(section, {}):
        value = specification.get(concern)
        if not isinstance(value, list) or not value:
            continue
        records = [
            deepcopy(dict(item))
            for item in value
            if isinstance(item, Mapping)
        ]
        if records:
            result[concern] = records
    return result


def _projection_text(value: Any) -> str:
    return " ".join(
        str(value or "").replace("\r", " ").replace("\n", " ").split()
    )


def _projection_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten canonical nested records only for the human Markdown projection."""
    result: dict[str, Any] = {}

    def visit(value: Mapping[str, Any]) -> None:
        for key, item in value.items():
            if isinstance(item, Mapping):
                visit(item)
                continue
            if key in result:
                raise ValueError(
                    f"AUTHORED_STRUCTURED_DESIGN: duplicate leaf field {key!r}"
                )
            result[str(key)] = item

    visit(record)
    return result


def render_structured_sections(sections: Mapping[str, Any]) -> str:
    normalized = normalize_structured_sections(sections)
    lines: list[str] = []
    for section in WORKSHEET_SECTIONS:
        records_by_concern = active_concern_records(normalized, section)
        if not records_by_concern:
            continue
        lines.append(f"## {section}")
        for concern, columns in DETAIL_RECORDS[section].items():
            records = records_by_concern.get(concern)
            if not records:
                continue
            fields = tuple(columns.split())
            lines.append(f"- {concern}: {' '.join(fields)}")
            for index, record in enumerate(records, start=1):
                projected = _projection_record(record)
                parts = [
                    f"{field}={_projection_text(projected.get(field, ''))}"
                    for field in fields
                ]
                lines.append(f"  - record_{index}: " + "; ".join(parts))
        lines.append("")
    return "\n".join(lines).rstrip()


def _authored_chunk_messages(
    prompt: str,
    *,
    section: str,
    chunk_index: int,
    chunk_count: int,
    concerns: Sequence[str],
    completed: Mapping[str, Mapping[str, Any]],
    include_evidence: bool,
    record_counts: Mapping[str, int],
) -> tuple[dict[str, str], ...]:
    from .planning_state_implementation import SECTION_DEPENDENCIES
    from .worksheet_atomic_chunker import worksheet_chunk_prompt

    dependencies = {
        dependency: deepcopy(completed[dependency])
        for dependency in SECTION_DEPENDENCIES.get(section, ())
        if dependency in completed
    }
    prerequisite_text = (
        json.dumps(
            dependencies,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if dependencies
        else "{}"
    )
    return (
        {
            "role": "system",
            "content": (
                "Author one bounded piece of the game design directly into the host-owned "
                "canonical record template. Make concrete design choices from the user's "
                "request; do not summarize a separate Markdown draft and do not defer semantic "
                "work to production. Preserve prerequisite records exactly where they constrain "
                "this chunk. Do not invent external API symbols, repository facts, versions, "
                "or evidence identifiers. Return only the fixed structured payload requested "
                "by the host."
            ),
        },
        {
            "role": "user",
            "content": (
                "User request:\n"
                + prompt
                + "\n\nCanonical prerequisite sections:\n"
                + prerequisite_text
                + "\n\n"
                + worksheet_chunk_prompt(
                    section,
                    chunk_index,
                    chunk_count,
                    concerns,
                    include_evidence=include_evidence,
                    record_counts=record_counts,
                )
            ),
        },
    )


def _generate_authored_chunk(
    router: Any,
    prompt: str,
    *,
    section: str,
    chunk_index: int,
    chunk_count: int,
    concerns: Sequence[str],
    completed: Mapping[str, Mapping[str, Any]],
    include_evidence: bool,
    record_counts: Mapping[str, int],
    media_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Generate one canonical chunk; schema failure becomes narrower work, not prose recovery."""

    from .fixed_template_generation import generate_fixed_template_value
    from .worksheet_atomic_chunker import WorksheetConcernChunk, worksheet_chunk_schema

    def generate(selected: Sequence[str], *, evidence: bool) -> dict[str, Any]:
        schema = worksheet_chunk_schema(
            section,
            selected,
            include_evidence=evidence,
            record_counts=record_counts,
        )
        value = generate_fixed_template_value(
            router,
            "planner",
            _authored_chunk_messages(
                prompt,
                section=section,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                concerns=selected,
                completed=completed,
                include_evidence=evidence,
                record_counts=record_counts,
            ),
            response_schema=schema,
            media_paths=media_paths,
            enable_tools=False,
            tool_name=f"author_{section}_{chunk_index}",
            description=f"Author canonical {section} concern records.",
        )
        if not isinstance(value, Mapping):
            raise ValueError(
                f"AUTHORED_STRUCTURED_DESIGN: {section} chunk {chunk_index} must be an object"
            )
        return dict(value)

    try:
        return generate(concerns, evidence=include_evidence)
    except (ValueError, RuntimeError, TypeError):
        if len(concerns) <= 1:
            raise

    explicit_projection = getattr(concerns, "field_projection", {})
    merged: dict[str, Any] = {}
    merged_inapplicable: list[dict[str, Any]] = []
    merged_refs: list[str] = []
    for position, concern in enumerate(concerns):
        fields = (
            explicit_projection.get(concern)
            if isinstance(explicit_projection, Mapping)
            else None
        )
        isolated = WorksheetConcernChunk(
            (concern,),
            {
                concern: tuple(fields)
                if fields
                else tuple(DETAIL_RECORDS[section][concern].split())
            },
        )
        value = generate(
            isolated,
            evidence=bool(include_evidence and position == 0),
        )
        if concern in value:
            merged[concern] = deepcopy(value[concern])
        for item in value.get("inapplicable_concerns", []):
            if isinstance(item, Mapping) and item not in merged_inapplicable:
                merged_inapplicable.append(deepcopy(dict(item)))
        for ref in value.get("constraint_evidence_refs", []):
            if isinstance(ref, str) and ref not in merged_refs:
                merged_refs.append(ref)

    if merged_inapplicable:
        merged["inapplicable_concerns"] = merged_inapplicable
    if include_evidence:
        merged["constraint_evidence_refs"] = merged_refs
    return merged


def author_structured_sections(
    router: Any,
    prompt: str,
    *,
    media_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Author the canonical design directly, one small concern chunk at a time.

    The returned records are the semantic source of truth. Markdown is rendered from
    these records afterward and is never required to be re-parsed to recover semantics.
    """

    from .worksheet_atomic_chunker import (
        merge_worksheet_section_chunks,
        pack_section_concerns,
    )

    completed: dict[str, Any] = {}
    for section in WORKSHEET_SECTIONS:
        chunks = pack_section_concerns(section)
        chunk_results: list[dict[str, Any]] = []
        record_counts: dict[str, int] = {}
        for index, concerns in enumerate(chunks, start=1):
            value = _generate_authored_chunk(
                router,
                prompt,
                section=section,
                chunk_index=index,
                chunk_count=len(chunks),
                concerns=concerns,
                completed=completed,
                include_evidence=index == 1,
                record_counts=record_counts,
                media_paths=media_paths,
            )
            inapplicable = {
                str(item.get("concern") or "")
                for item in value.get("inapplicable_concerns", [])
                if isinstance(item, Mapping)
            }
            for concern in concerns:
                rows = value.get(concern)
                if isinstance(rows, list):
                    record_counts.setdefault(concern, len(rows))
                elif concern in inapplicable:
                    record_counts.setdefault(concern, 0)
            chunk_results.append(value)

        completed[section] = merge_worksheet_section_chunks(
            section,
            chunk_results,
            set(),
        )

    return normalize_structured_sections(completed)


__all__ = [
    "active_concern_records",
    "author_structured_sections",
    "normalize_structured_sections",
    "render_structured_sections",
]
