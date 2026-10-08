from __future__ import annotations

"""Canonical structured authored-design contract.

The model authors fixed concern records. Markdown is a deterministic projection for
human review and provenance only; executable production consumes the records.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .execution_contract_policy import PLANNER_CONCERN_MAX_RECORDS
from .model_output_atomicity_contract import structured_output_token_ceiling
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
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return " ".join(
        str(value).replace("\r", " ").replace("\n", " ").split()
    )


def _projection_record(
    record: Mapping[str, Any],
    *,
    fields: Sequence[str],
) -> dict[str, Any]:
    """Project declared worksheet fields without flattening their values.

    Executable fields such as guard, condition and mutation contain nested typed
    IR where keys such as kind legitimately repeat. The human Markdown projection
    therefore treats each declared worksheet field as one opaque canonical value.
    Legacy wrapper mappings are traversed only to locate a declared field.
    """

    wanted = {str(field) for field in fields}
    result: dict[str, Any] = {}

    def visit(value: Mapping[str, Any]) -> None:
        for raw_key, item in value.items():
            key = str(raw_key)
            if key in wanted:
                if key in result:
                    raise ValueError(
                        "AUTHORED_STRUCTURED_DESIGN: duplicate projected field "
                        f"{key!r}"
                    )
                result[key] = deepcopy(item)
                continue
            if isinstance(item, Mapping):
                visit(item)

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
                projected = _projection_record(record, fields=fields)
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
    state_symbols: Any = None,
    section_context: Mapping[str, Any] | None = None,
    record_counts: Mapping[str, int | None] | None = None,
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
            "When the same active concern already appears above, preserve its existing "
            "record count and row order exactly while filling the newly projected fields."
        )
    user_parts.append(
        worksheet_chunk_prompt(
            section,
            chunk_index,
            chunk_count,
            concerns,
            record_counts=record_counts,
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



def _authored_cardinality_messages(
    prompt: str,
    *,
    section: str,
    concern: str,
    completed: Mapping[str, Mapping[str, Any]],
    state_symbols: Any = None,
    section_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], ...]:
    from .planning_section_dependencies import SECTION_DEPENDENCIES
    from .worksheet_atomic_chunker import worksheet_concern_cardinality_prompt

    dependencies = {
        dependency: deepcopy(completed[dependency])
        for dependency in SECTION_DEPENDENCIES.get(section, ())
        if dependency in completed
    }
    user_parts = [
        "User request:\n" + prompt,
        "Canonical prerequisite sections:\n"
        + json.dumps(
            dependencies,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
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
        user_parts.append(
            f"Canonical {section} records authored so far:\n"
            + json.dumps(
                section_context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    user_parts.append(
        worksheet_concern_cardinality_prompt(section, concern)
    )
    return (
        {
            "role": "system",
            "content": (
                "Make exactly one bounded semantic decision: how many records the declared "
                "concern requires. Do not author record fields, prose, loop control, or tool "
                "calls. The host owns iteration and all subsequent field pages."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(user_parts),
        },
    )


def _generate_concern_record_count(
    router: Any,
    prompt: str,
    *,
    section: str,
    concern: str,
    completed: Mapping[str, Mapping[str, Any]],
    state_symbols: Any = None,
    section_context: Mapping[str, Any] | None = None,
) -> int:
    from .fixed_template_generation import generate_fixed_template_value
    from .worksheet_atomic_chunker import worksheet_concern_cardinality_schema

    schema = worksheet_concern_cardinality_schema(section, concern)
    value = generate_fixed_template_value(
        router,
        "planner",
        _authored_cardinality_messages(
            prompt,
            section=section,
            concern=concern,
            completed=completed,
            state_symbols=state_symbols,
            section_context=section_context,
        ),
        response_schema=schema,
        media_paths=(),
        enable_tools=False,
        description=f"Choose bounded record cardinality for {section}.{concern}.",
        output_token_ceiling=structured_output_token_ceiling(schema),
    )
    if not isinstance(value, Mapping):
        raise ValueError(
            f"AUTHORED_STRUCTURED_DESIGN: cardinality for {section}.{concern} must be an object"
        )
    try:
        result = int(value["record_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"AUTHORED_STRUCTURED_DESIGN: invalid cardinality for {section}.{concern}"
        ) from exc
    if result < 0 or result > PLANNER_CONCERN_MAX_RECORDS:
        raise ValueError(
            f"AUTHORED_STRUCTURED_DESIGN: cardinality for {section}.{concern} "
            f"is outside 0..{PLANNER_CONCERN_MAX_RECORDS}: {result}"
        )
    return result

@dataclass(frozen=True)
class _PlannerPageRequest:
    section: str
    chunk_index: int
    chunk_count: int
    concerns: Sequence[str]
    completed: Mapping[str, Mapping[str, Any]]
    media_paths: Sequence[str | Path]


@dataclass(frozen=True)
class _PlannerSectionRequest:
    router: Any
    prompt: str
    section: str
    completed: Mapping[str, Mapping[str, Any]]
    media_paths: Sequence[str | Path]
    chunks: Sequence[Sequence[str]]
    slots: int
    budget: Any = None


def _generate_authored_chunk(
    router: Any,
    prompt: str,
    *,
    page: _PlannerPageRequest,
    state_symbols: Any = None,
    section_context: Mapping[str, Any] | None = None,
    record_counts: Mapping[str, int | None] | None = None,
) -> dict[str, Any]:
    """Generate exactly one host-bounded concern field page."""

    from .worksheet_atomic_chunker import (
        worksheet_chunk_schema,
    )

    if len(page.concerns) != 1:
        raise ValueError(
            "AUTHORED_STRUCTURED_DESIGN: planner page must contain exactly one concern"
        )
    concern = str(page.concerns[0])
    if not isinstance(record_counts, Mapping) or concern not in record_counts:
        raise ValueError(
            "AUTHORED_STRUCTURED_DESIGN: planner field page requires a host-fixed "
            f"record count for {page.section}.{concern}"
        )

    schema = worksheet_chunk_schema(
        page.section,
        page.concerns,
        record_counts=record_counts,
        state_symbols=state_symbols,
    )
    messages: Sequence[Mapping[str, Any]]
    if page.section == "state_model":
        messages = ()
    else:
        messages = _authored_chunk_messages(
            prompt,
            section=page.section,
            chunk_index=page.chunk_index,
            chunk_count=page.chunk_count,
            concerns=page.concerns,
            completed=page.completed,
            state_symbols=state_symbols,
            section_context=section_context,
            record_counts=record_counts,
        )
    value = _generate_authored_page_value(
        router,
        prompt,
        messages,
        page,
        schema,
        state_symbols,
        section_context=section_context,
    )
    if not isinstance(value, Mapping):
        raise ValueError(
            f"AUTHORED_STRUCTURED_DESIGN: {page.section} page "
            f"{page.chunk_index} must be an object"
        )
    return dict(value)


def _generate_authored_page_value(
    router: Any,
    prompt: str,
    messages: Sequence[Mapping[str, Any]],
    page: _PlannerPageRequest,
    schema: Mapping[str, Any],
    state_symbols: Any,
    *,
    section_context: Mapping[str, Any] | None = None,
) -> Any:
    from .fixed_template_generation import generate_fixed_template_value

    if page.section == "state_model":
        from .planner_state_authoring import (
            STATE_EXECUTABLE_FIELDS,
            author_state_field_page,
            author_state_semantic_page,
        )

        concern = str(page.concerns[0])
        concern_schema = schema["properties"][concern]
        fields = tuple(concern_schema["items"]["required"])

        executable = [
            field for field in fields
            if field in STATE_EXECUTABLE_FIELDS
        ]

        if executable:
            if len(fields) != 1 or len(executable) != 1:
                raise ValueError(
                    "STATE_EXECUTABLE_PAGE_MUST_BE_SINGLETON: "
                    f"{concern}: {fields!r}"
                )

            existing_rows: Sequence[Mapping[str, Any]] = ()
            if isinstance(section_context, Mapping):
                candidate_rows = section_context.get(concern)
                if isinstance(candidate_rows, Sequence) and not isinstance(
                    candidate_rows, (str, bytes, bytearray)
                ):
                    existing_rows = tuple(
                        row for row in candidate_rows if isinstance(row, Mapping)
                    )
            return author_state_field_page(
                router,
                prompt,
                concern=concern,
                field=executable[0],
                count=concern_schema["minItems"],
                symbols=state_symbols,
                existing_rows=existing_rows,
            )

        if concern_schema.get("minItems") == concern_schema.get("maxItems"):
            existing_rows: Sequence[Mapping[str, Any]] = ()
            if isinstance(section_context, Mapping):
                candidate_rows = section_context.get(concern)
                if isinstance(candidate_rows, Sequence) and not isinstance(
                    candidate_rows, (str, bytes, bytearray)
                ):
                    existing_rows = tuple(
                        row for row in candidate_rows if isinstance(row, Mapping)
                    )
            return author_state_semantic_page(
                router,
                prompt,
                concern=concern,
                fields=fields,
                count=int(concern_schema["minItems"]),
                item_schema=concern_schema["items"],
                existing_rows=existing_rows,
            )
    return generate_fixed_template_value(
        router,
        "planner",
        messages,
        response_schema=schema,
        media_paths=page.media_paths,
        enable_tools=False,
        description=f"Author bounded canonical {page.section} field page.",
        output_token_ceiling=structured_output_token_ceiling(schema),
    )


def _index_concern_pages(
    chunks: Sequence[Sequence[str]],
) -> tuple[
    list[tuple[int, Sequence[str]]],
    dict[str, list[tuple[int, Sequence[str]]]],
    list[str],
]:
    indexed = list(enumerate(chunks, start=1))
    pages: dict[str, list[tuple[int, Sequence[str]]]] = {}
    order: list[str] = []
    for index, page in indexed:
        if len(page) != 1:
            raise ValueError(
                "AUTHORED_STRUCTURED_DESIGN: host planner page contains multiple concerns"
            )
        concern = str(page[0])
        if concern not in pages:
            pages[concern] = []
            order.append(concern)
        pages[concern].append((index, page))
    return indexed, pages, order


def _merge_page_rows(
    current: list[dict[str, Any]],
    rows: Any,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return current
    while len(current) < len(rows):
        current.append({})
    for row_index, row in enumerate(rows):
        if isinstance(row, Mapping):
            current[row_index].update(deepcopy(dict(row)))
    return current


def _generate_concern_pages(
    request: _PlannerSectionRequest,
    concern: str,
    pages: Sequence[tuple[int, Sequence[str]]],
    *,
    state_symbols: Any = None,
    base_context: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, Any]]]]:
    if not pages:
        return [], []

    authored_rows: list[dict[str, Any]] = []
    results: list[tuple[int, dict[str, Any]]] = []
    fixed_count: int | None = None

    if request.section == "state_model":
        if request.budget is not None:
            request.budget.consume("structured.state_model.cardinality")
        fixed_count = _generate_concern_record_count(
            request.router,
            request.prompt,
            section=request.section,
            concern=concern,
            completed=request.completed,
            state_symbols=state_symbols,
            section_context=base_context,
        )
        if fixed_count == 0:
            return [], [(idx, {concern: []}) for idx, _page in pages]

    for i, (index, concerns) in enumerate(pages):
        context = deepcopy(dict(base_context or {}))
        if authored_rows:
            context[concern] = deepcopy(authored_rows)
        page = _PlannerPageRequest(
            section=request.section,
            chunk_index=index,
            chunk_count=len(request.chunks),
            concerns=concerns,
            completed=request.completed,
            media_paths=request.media_paths,
        )
        if request.budget is not None:
            call_count = 1
            if request.section == "state_model" and fixed_count is not None:
                # Every state page now emits exactly one model call per host-fixed
                # row, regardless of how many semantic fields the page projects.
                call_count = max(1, int(fixed_count))
            request.budget.consume(
                f"structured.{request.section}.page",
                count=call_count,
            )
        value = _generate_authored_chunk(
            request.router,
            request.prompt,
            page=page,
            state_symbols=state_symbols,
            section_context=context or None,
            record_counts={concern: fixed_count},
        )
        rows = value.get(concern, [])
        if not isinstance(rows, list):
            raise ValueError(
                f"AUTHORED_STRUCTURED_DESIGN: {request.section}.{concern} page "
                f"{index} must return a record array"
            )
        if request.section != "state_model" and i == 0:
            fixed_count = len(rows)
            if fixed_count == 0:
                return [], [(idx, {concern: []}) for idx, _page in pages]
        elif fixed_count is not None and len(rows) != fixed_count:
            raise ValueError(
                f"AUTHORED_STRUCTURED_DESIGN_CARDINALITY_DRIFT: "
                f"{request.section}.{concern} expected {fixed_count} rows, "
                f"got {len(rows)}"
            )
        authored_rows = _merge_page_rows(authored_rows, rows)
        results.append((index, value))
    return authored_rows, results


def _state_section_page_results(
    request: _PlannerSectionRequest,
    indexed: Sequence[tuple[int, Sequence[str]]],
    pages: Mapping[str, Sequence[tuple[int, Sequence[str]]]],
    order: Sequence[str],
) -> list[dict[str, Any]]:
    from .structured_state_runtime import StateSymbolTable, validate_state_concern

    ordered = list(order)
    if "variables" in ordered:
        ordered.remove("variables")
        ordered.insert(0, "variables")

    symbols: StateSymbolTable | None = None
    accumulated: dict[str, Any] = {}
    by_index: dict[int, dict[str, Any]] = {}
    for concern in ordered:
        rows, results = _generate_concern_pages(
            request,
            concern,
            pages[concern],
            state_symbols=symbols,
            base_context=accumulated,
        )
        validate_state_concern(concern, rows, symbols=symbols)
        if rows:
            accumulated[concern] = deepcopy(rows)
        if concern == "variables":
            symbols = StateSymbolTable(rows)
        by_index.update(dict(results))
    return [by_index[index] for index, _page in indexed]


def _ordered_bounded_planning_results(
    tasks: Sequence[str],
    worker: Any,
    *,
    slots: int,
    stage: str,
) -> list[Any]:
    """One deadline-bounded scheduler for independent design concerns and waves.

    Restore the caller's stable worksheet order after completion-order execution.
    This retains parallelism without nested, unbounded executor ownership.
    """
    if slots <= 1 or len(tasks) <= 1:
        return [worker(task) for task in tasks]
    from .deadline_executor import iter_completed_with_deadlines

    completed = dict(iter_completed_with_deadlines(
        tasks, worker, max_workers=min(slots, len(tasks)), stage=stage,
    ))
    return [completed[task] for task in tasks]


def _regular_section_page_results(
    request: _PlannerSectionRequest,
    indexed: Sequence[tuple[int, Sequence[str]]],
    pages: Mapping[str, Sequence[tuple[int, Sequence[str]]]],
    order: Sequence[str],
    *,
    allow_parallel: bool,
) -> list[dict[str, Any]]:
    def run(concern: str) -> tuple[str, list[tuple[int, dict[str, Any]]]]:
        _rows, results = _generate_concern_pages(
            request,
            concern,
            pages[concern],
        )
        return concern, results

    groups = _ordered_bounded_planning_results(
        order, run,
        slots=request.slots if allow_parallel else 1,
        stage=f"authored-section-{request.section}",
    )

    by_index: dict[int, dict[str, Any]] = {}
    for _concern, results in groups:
        by_index.update(dict(results))
    return [by_index[index] for index, _page in indexed]


def _author_section(
    request: _PlannerSectionRequest,
    *,
    allow_parallel: bool,
) -> dict[str, Any]:
    from .worksheet_atomic_chunker import merge_worksheet_section_chunks

    indexed, pages, order = _index_concern_pages(request.chunks)
    if request.section == "state_model":
        results = _state_section_page_results(request, indexed, pages, order)
    else:
        results = _regular_section_page_results(
            request,
            indexed,
            pages,
            order,
            allow_parallel=allow_parallel,
        )
    return merge_worksheet_section_chunks(request.section, results, set())


def author_structured_sections(
    router: Any,
    prompt: str,
    *,
    media_paths: Sequence[str | Path] = (),
    budget: Any = None,
) -> dict[str, Any]:
    """Author the canonical design as a finite host-owned dependency DAG."""

    from .model_concurrency import router_native_model_parallelism
    from .planning_section_dependencies import SECTION_DEPENDENCIES
    from .worksheet_atomic_chunker import pack_section_concerns

    slots = max(1, router_native_model_parallelism(router, role="planner"))
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

        def run(section: str) -> tuple[str, dict[str, Any]]:
            request = _PlannerSectionRequest(
                router=router,
                prompt=prompt,
                section=section,
                completed=snapshot,
                media_paths=media_paths,
                chunks=pack_section_concerns(section),
                slots=slots,
                budget=budget,
            )
            return section, _author_section(
                request,
                allow_parallel=len(ready) == 1,
            )

        authored = _ordered_bounded_planning_results(
            ready, run, slots=slots, stage="authored-design-wave",
        )

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
