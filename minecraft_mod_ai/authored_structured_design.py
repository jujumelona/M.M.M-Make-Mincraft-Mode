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
        if section == "state_model":
            from .structured_state_runtime import validate_structured_state_section

            try:
                validate_structured_state_section(value)
            except ValueError as exc:
                raise ValueError(
                    "AUTHORED_STRUCTURED_DESIGN: state_model is outside the "
                    f"host-compiled state DSL: {exc}"
                ) from exc
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
    state_symbols: Any = None,
    section_context: Mapping[str, Any] | None = None,
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

    user_parts = [
        "User request:\n" + prompt,
        "Canonical prerequisite sections:\n" + prerequisite_text,
    ]
    if state_symbols:
        symbols_text = (
            state_symbols.prompt_text()
            if hasattr(state_symbols, "prompt_text")
            else str(state_symbols)
        )
        if symbols_text:
            user_parts.append(symbols_text)
    if section_context:
        context_text = json.dumps(
            section_context,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        user_parts.append(f"Canonical {section} records authored so far:\n{context_text}")
    user_parts.append(
        worksheet_chunk_prompt(
            section,
            chunk_index,
            chunk_count,
            concerns,
            include_evidence=include_evidence,
        )
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
            "content": "\n\n".join(user_parts),
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
    media_paths: Sequence[str | Path],
    state_symbols: Any = None,
    section_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one canonical chunk; schema failure becomes narrower work, not prose recovery."""

    from .fixed_template_generation import generate_fixed_template_value
    from .worksheet_atomic_chunker import WorksheetConcernChunk, worksheet_chunk_schema

    def generate(selected: Sequence[str], *, evidence: bool) -> dict[str, Any]:
        schema = worksheet_chunk_schema(
            section,
            selected,
            include_evidence=evidence,
            state_symbols=state_symbols,
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
                state_symbols=state_symbols,
                section_context=section_context,
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
        result = dict(value)
        if section == "state_model":
            from .structured_state_runtime import validate_state_concern

            for concern in selected:
                rows = result.get(concern, [])
                if rows:
                    validate_state_concern(
                        concern,
                        rows,
                        symbols=state_symbols,
                    )
        return result

    try:
        return generate(concerns, evidence=include_evidence)
    except (ValueError, RuntimeError, TypeError) as initial_error:
        if len(concerns) == 1:
            concern = str(concerns[0])
            explicit_projection = getattr(concerns, "field_projection", {})
            projected_fields = (
                tuple(explicit_projection.get(concern, ()))
                if isinstance(explicit_projection, Mapping)
                else ()
            )
            fields = projected_fields or tuple(DETAIL_RECORDS[section][concern].split())
            if len(fields) <= 1:
                raise

            # Last-resort small-model recovery: one field per model call.  Each page
            # carries only one bounded string/array field, so a pathological model
            # cannot consume the full concern budget by repeating sibling records.
            record: dict[str, Any] = {}
            merged_inapplicable: list[dict[str, Any]] = []
            merged_refs: list[str] = []
            for position, field in enumerate(fields):
                isolated = WorksheetConcernChunk(
                    (concern,),
                    {concern: (field,)},
                )
                value = generate(
                    isolated,
                    evidence=bool(include_evidence and position == 0),
                )
                rows = value.get(concern)
                if isinstance(rows, list):
                    first = next(
                        (
                            item
                            for item in rows
                            if isinstance(item, Mapping) and field in item
                        ),
                        None,
                    )
                    if first is not None:
                        record[field] = deepcopy(first[field])
                for item in value.get("inapplicable_concerns", []):
                    if isinstance(item, Mapping) and item not in merged_inapplicable:
                        merged_inapplicable.append(deepcopy(dict(item)))
                for ref in value.get("constraint_evidence_refs", []):
                    if isinstance(ref, str) and ref not in merged_refs:
                        merged_refs.append(ref)

            if not record:
                raise initial_error
            merged: dict[str, Any] = {concern: [record]}
            if merged_inapplicable:
                merged["inapplicable_concerns"] = merged_inapplicable
            if include_evidence:
                merged["constraint_evidence_refs"] = merged_refs
            return merged

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
        if section == "state_model":
            from .structured_state_runtime import StateSymbolTable, validate_state_concern

            chunks = pack_section_concerns("state_model")
            var_idx = next(
                (i for i, c in enumerate(chunks) if "variables" in c),
                None,
            )
            if var_idx is not None:
                ordered_chunks: list[Sequence[str]] = [chunks[var_idx]] + [
                    c for i, c in enumerate(chunks) if i != var_idx
                ]
            else:
                ordered_chunks = list(chunks)

            chunk_results: list[dict[str, Any]] = []
            state_symbols: StateSymbolTable | None = None
            accumulated_state_records: dict[str, Any] = {}

            for index, concerns in enumerate(ordered_chunks, start=1):
                value = _generate_authored_chunk(
                    router,
                    prompt,
                    section="state_model",
                    chunk_index=index,
                    chunk_count=len(ordered_chunks),
                    concerns=concerns,
                    completed=completed,
                    include_evidence=index == 1,
                    media_paths=media_paths,
                    state_symbols=state_symbols,
                    section_context=accumulated_state_records if accumulated_state_records else None,
                )
                for concern in concerns:
                    rows = value.get(concern, [])
                    validate_state_concern(concern, rows, symbols=state_symbols)
                    if rows:
                        accumulated_state_records[concern] = deepcopy(rows)
                    if concern == "variables":
                        state_symbols = StateSymbolTable(rows)

                chunk_results.append(value)

            completed["state_model"] = merge_worksheet_section_chunks(
                "state_model",
                chunk_results,
                set(),
            )
            continue

        chunks = pack_section_concerns(section)
        chunk_results = []
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
                media_paths=media_paths,
            )
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
