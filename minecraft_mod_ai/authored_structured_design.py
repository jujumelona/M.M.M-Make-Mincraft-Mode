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
    from .planning_section_dependencies import SECTION_DEPENDENCIES
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
    except (ValueError, RuntimeError, TypeError):
        if len(concerns) == 1:
            # One complete concern is already the minimum semantic work unit.
            # Do not explode it into one model call per field: that recreates the
            # latency/runaway failure mode and lets partial fields drift apart.
            raise

    explicit_projection = getattr(concerns, "field_projection", {})

    def generate_isolated(
        item: tuple[int, str],
    ) -> tuple[int, str, dict[str, Any]]:
        position, concern = item
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
        return (
            position,
            concern,
            generate(
                isolated,
                evidence=bool(include_evidence and position == 0),
            ),
        )

    indexed_concerns = [
        (position, str(concern))
        for position, concern in enumerate(concerns)
    ]
    from .model_concurrency import router_native_model_parallelism

    fallback_workers = min(
        len(indexed_concerns),
        max(1, router_native_model_parallelism(router, role="planner")),
    )
    if fallback_workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=fallback_workers,
            thread_name_prefix=f"mmm-plan-recover-{section}",
        ) as executor:
            isolated_results = list(
                executor.map(generate_isolated, indexed_concerns)
            )
    else:
        isolated_results = [
            generate_isolated(item)
            for item in indexed_concerns
        ]

    isolated_results.sort(key=lambda item: item[0])
    merged: dict[str, Any] = {}
    merged_inapplicable: list[dict[str, Any]] = []
    merged_refs: list[str] = []
    for _position, concern, value in isolated_results:
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
    """Author the canonical design as a finite host-owned dependency DAG."""

    from .model_concurrency import router_native_model_parallelism
    from .planning_section_dependencies import SECTION_DEPENDENCIES
    from .worksheet_atomic_chunker import (
        merge_worksheet_section_chunks,
        pack_section_concerns,
    )

    slots = max(1, router_native_model_parallelism(router, role="planner"))

    def author_section(
        section: str,
        completed_snapshot: Mapping[str, Mapping[str, Any]],
        *,
        allow_chunk_parallel: bool,
    ) -> dict[str, Any]:
        if section == "state_model":
            from .structured_state_runtime import (
                StateSymbolTable,
                validate_state_concern,
            )

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
                    completed=completed_snapshot,
                    include_evidence=index == 1,
                    media_paths=media_paths,
                    state_symbols=state_symbols,
                    section_context=(
                        accumulated_state_records
                        if accumulated_state_records
                        else None
                    ),
                )
                for concern in concerns:
                    rows = value.get(concern, [])
                    validate_state_concern(
                        concern,
                        rows,
                        symbols=state_symbols,
                    )
                    if rows:
                        accumulated_state_records[concern] = deepcopy(rows)
                    if concern == "variables":
                        state_symbols = StateSymbolTable(rows)
                chunk_results.append(value)

            return merge_worksheet_section_chunks(
                "state_model",
                chunk_results,
                set(),
            )

        chunks = pack_section_concerns(section)

        def generate_chunk(
            item: tuple[int, Sequence[str]],
        ) -> tuple[int, dict[str, Any]]:
            index, concerns = item
            return index, _generate_authored_chunk(
                router,
                prompt,
                section=section,
                chunk_index=index,
                chunk_count=len(chunks),
                concerns=concerns,
                completed=completed_snapshot,
                include_evidence=index == 1,
                media_paths=media_paths,
            )

        indexed = list(enumerate(chunks, start=1))
        if not indexed:
            chunk_results: list[dict[str, Any]] = []
        elif allow_chunk_parallel and slots > 1 and len(indexed) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(
                max_workers=min(slots, len(indexed)),
                thread_name_prefix=f"mmm-plan-{section}",
            ) as executor:
                ordered = list(executor.map(generate_chunk, indexed))
            ordered.sort(key=lambda item: item[0])
            chunk_results = [value for _, value in ordered]
        else:
            chunk_results = [
                generate_chunk(item)[1]
                for item in indexed
            ]

        return merge_worksheet_section_chunks(
            section,
            chunk_results,
            set(),
        )

    completed: dict[str, Any] = {}
    pending = set(WORKSHEET_SECTIONS)
    while pending:
        ready = [
            section
            for section in WORKSHEET_SECTIONS
            if section in pending
            and set(SECTION_DEPENDENCIES.get(section, ())).issubset(completed)
        ]
        if not ready:
            unresolved = {
                section: tuple(
                    dependency
                    for dependency in SECTION_DEPENDENCIES.get(section, ())
                    if dependency not in completed
                )
                for section in WORKSHEET_SECTIONS
                if section in pending
            }
            raise ValueError(
                "AUTHORED_STRUCTURED_DESIGN_DEPENDENCY_CYCLE: "
                + repr(unresolved)
            )

        snapshot = deepcopy(completed)
        if slots > 1 and len(ready) > 1:
            from concurrent.futures import ThreadPoolExecutor

            def run_section(section: str) -> tuple[str, dict[str, Any]]:
                return section, author_section(
                    section,
                    snapshot,
                    allow_chunk_parallel=False,
                )

            with ThreadPoolExecutor(
                max_workers=min(slots, len(ready)),
                thread_name_prefix="mmm-plan-wave",
            ) as executor:
                authored = list(executor.map(run_section, ready))
        else:
            authored = [
                (
                    section,
                    author_section(
                        section,
                        snapshot,
                        allow_chunk_parallel=True,
                    ),
                )
                for section in ready
            ]

        authored_by_section = dict(authored)
        for section in WORKSHEET_SECTIONS:
            if section in authored_by_section:
                completed[section] = authored_by_section[section]
                pending.remove(section)

    return normalize_structured_sections(completed)


__all__ = [
    "active_concern_records",
    "author_structured_sections",
    "normalize_structured_sections",
    "render_structured_sections",
]
