"""Assemble authored criterion records without promoting prose into contracts."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from .acceptance_contracts import (
    canonical_public_acceptance,
    project_requirement_public_acceptance,
)
from .planning_detail_slots import DETAIL_RECORDS
from .planning_detail_template import (
    normalize_required_sections,
    validate_worksheet_section,
)
from .task_template_batch_runner import (
    record_template_batches,
    run_record_template_batch,
    supports_record_batching,
)
from .task_template_catalog import load_template
from .task_template_runner import run_record_template

_PROGRESS_SCHEMA = "mmm/detail-criterion-records"
_PROGRESS_KEY = "detail_progress"
_FRAGMENT_KEY = "section_updates"


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def requirement_acceptance_criteria(requirement: Mapping[str, Any]) -> tuple[str, ...]:
    return canonical_public_acceptance(requirement.get("acceptance")) or (
        project_requirement_public_acceptance(requirement),
    )


def _section_results(
    router: Any,
    identifiers: list[str],
    *,
    context: Mapping[str, Any],
    allowed_refs: set[str],
    progress: Mapping[str, Any] | None,
    checkpoint: Any,
) -> dict[str, dict[str, Any]]:
    """Use bounded batching for real fixed-tool routers; keep fixture routers atomic."""

    if supports_record_batching(router):
        results: dict[str, dict[str, Any]] = {}
        for batch in record_template_batches(identifiers):
            results.update(
                run_record_template_batch(
                    router,
                    batch,
                    context=context,
                    allowed_refs=allowed_refs,
                    progress=progress,
                    checkpoint=checkpoint,
                )
            )
        return results

    # Deterministic fixture routers intentionally exercise the canonical single-template
    # runner. This is not a production fallback: real ModelRouter transports expose native
    # tool decisions and therefore take the batched path above.
    return {
        identifier: run_record_template(
            router,
            identifier,
            context=context,
            allowed_refs=allowed_refs,
            progress=progress,
            checkpoint=checkpoint,
        )
        for identifier in identifiers
    }


def generate_section_records(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    criterion: str,
    section: str,
    evidence: Any,
    allowed_refs: set[str],
    progress: Mapping[str, Any] | None = None,
    checkpoint: Any = None,
) -> dict[str, Any]:
    specification: dict[str, Any] = {}
    refs: list[str] = []
    reasons: list[dict[str, str]] = []
    identifiers = list(load_template(f"feature/{section}")["steps"])
    if isinstance(evidence, list) and any(isinstance(row, Mapping)
            and row.get("source") == "host_verified_semantic_research" for row in evidence):
        from .planning_semantic_research import evidence_for_obligation
        evidence = evidence_for_obligation(evidence, criterion)
        allowed_refs = {ref for row in evidence for ref in row.get("evidence_refs", [])}
    context = {
        "requirement_id": requirement.get("requirement_id", ""),
        "requirement": requirement.get("statement", ""),
        "criterion": criterion,
        "evidence": evidence,
    }
    results = _section_results(
        router,
        identifiers,
        context=context,
        allowed_refs=allowed_refs,
        progress=progress,
        checkpoint=checkpoint,
    )
    for identifier in identifiers:
        concern = identifier.rsplit("/", 1)[1]
        result = results[identifier]
        specification[concern] = result["records"]
        if not result["records"]:
            reasons.append({"concern": concern, "reason": result["reason"]})
        refs.extend(ref for ref in result["evidence_refs"] if ref not in refs)

    specification["inapplicable_concerns"] = reasons
    row = validate_worksheet_section(
        {"specification": specification, "constraint_evidence_refs": refs},
        allowed_refs,
        section,
    )
    return {"section": section, **row}


def validate_criterion_fragment(
    value: Any,
    *,
    selected_sections: Iterable[str],
    allowed_refs: set[str],
) -> dict[str, Any]:
    selected = normalize_required_sections(selected_sections)
    if (
        not isinstance(value, Mapping)
        or set(value) != {_FRAGMENT_KEY}
        or not isinstance(value[_FRAGMENT_KEY], list)
    ):
        raise ValueError("DETAILED_PLAN_CRITERION: expected authored section records")

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in value[_FRAGMENT_KEY]:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"section", "specification", "constraint_evidence_refs"}
        ):
            raise ValueError(
                "DETAILED_PLAN_CRITERION: prose or malformed records cannot certify a section"
            )
        section = str(row["section"])
        if section not in selected or section in seen:
            raise ValueError("DETAILED_PLAN_CRITERION: unknown or duplicate section")
        seen.add(section)
        validated = validate_worksheet_section(
            {
                key: row[key]
                for key in ("specification", "constraint_evidence_refs")
            },
            allowed_refs,
            section,
        )
        rows.append({"section": section, **validated})
    if not rows:
        raise ValueError("DETAILED_PLAN_NO_PROGRESS: no authored concern records")
    return {_FRAGMENT_KEY: rows}


def generate_criterion_fragment(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    criterion: str,
    selected_sections: Iterable[str],
    evidence: Any,
    allowed_refs: set[str],
    progress: Mapping[str, Any] | None = None,
    checkpoint: Any = None,
) -> dict[str, Any]:
    selected = normalize_required_sections(selected_sections)
    return {
        _FRAGMENT_KEY: [
            generate_section_records(
                router,
                requirement=requirement,
                criterion=criterion,
                section=section,
                evidence=evidence,
                allowed_refs=allowed_refs,
                progress=progress,
                checkpoint=checkpoint,
            )
            for section in selected
        ]
    }


def load_requirement_progress(
    state: Mapping[str, Any],
    *,
    requirement_ref: str,
    selected_sections: Iterable[str],
    criteria: tuple[str, ...],
    allowed_refs: set[str],
) -> dict[int, dict[str, Any]]:
    selected = normalize_required_sections(selected_sections)
    rows = state.get(_PROGRESS_KEY, []) or []
    if not isinstance(rows, list):
        raise ValueError("DETAILED_PLAN_PROGRESS: detail_progress must be an array")

    completed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("DETAILED_PLAN_PROGRESS: progress rows must be objects")
        if _text(row.get("requirement_ref")) != requirement_ref:
            continue
        if row.get("schema_version") != _PROGRESS_SCHEMA:
            raise ValueError("DETAILED_PLAN_PROGRESS: unsupported criterion progress schema")
        if tuple(row.get("required_detail_sections") or ()) != selected:
            raise ValueError("DETAILED_PLAN_PROGRESS: checkpoint section selection changed")
        index = row.get("criterion_index")
        if not isinstance(index, int) or index < 0 or index >= len(criteria):
            raise ValueError("DETAILED_PLAN_PROGRESS: invalid criterion index")
        if _text(row.get("criterion")) != criteria[index]:
            raise ValueError("DETAILED_PLAN_PROGRESS: checkpoint acceptance criterion changed")
        if index in completed:
            raise ValueError("DETAILED_PLAN_PROGRESS: duplicate criterion checkpoint")
        completed[index] = validate_criterion_fragment(
            row.get("fragment"),
            selected_sections=selected,
            allowed_refs=allowed_refs,
        )
    return completed


def store_criterion_progress(
    state: Mapping[str, Any],
    *,
    requirement_ref: str,
    selected_sections: Iterable[str],
    criterion_index: int,
    criterion: str,
    fragment: Mapping[str, Any],
) -> dict[str, Any]:
    selected = normalize_required_sections(selected_sections)
    value = deepcopy(dict(state))
    rows = value.get(_PROGRESS_KEY, []) or []
    if not isinstance(rows, list):
        raise ValueError("DETAILED_PLAN_PROGRESS: detail_progress must be an array")

    kept = [
        deepcopy(row)
        for row in rows
        if not (
            isinstance(row, Mapping)
            and _text(row.get("requirement_ref")) == requirement_ref
            and row.get("criterion_index") == criterion_index
        )
    ]
    kept.append(
        {
            "schema_version": _PROGRESS_SCHEMA,
            "requirement_ref": requirement_ref,
            "required_detail_sections": list(selected),
            "criterion_index": criterion_index,
            "criterion": criterion,
            "fragment": deepcopy(dict(fragment)),
        }
    )
    kept.sort(
        key=lambda row: (
            _text(row.get("requirement_ref")) if isinstance(row, Mapping) else "",
            int(row.get("criterion_index", -1))
            if isinstance(row, Mapping)
            else -1,
        )
    )
    value[_PROGRESS_KEY] = kept
    return value


def clear_requirement_progress(
    state: Mapping[str, Any], requirement_ref: str
) -> dict[str, Any]:
    value = deepcopy(dict(state))
    rows = value.get(_PROGRESS_KEY, []) or []
    if not isinstance(rows, list):
        raise ValueError("DETAILED_PLAN_PROGRESS: detail_progress must be an array")
    value[_PROGRESS_KEY] = [
        deepcopy(row)
        for row in rows
        if not (
            isinstance(row, Mapping)
            and _text(row.get("requirement_ref")) == requirement_ref
        )
    ]
    return value


class MissingWorksheetSections(ValueError):
    def __init__(self, sections: Iterable[str]):
        self.sections = tuple(sections)
        super().__init__(
            "DETAILED_PLAN_MISSING_SECTIONS: " + ", ".join(self.sections)
        )


def assemble_worksheet_from_fragments(
    requirement: Mapping[str, Any],
    *,
    selected_sections: Iterable[str],
    criteria: tuple[str, ...],
    fragments: Mapping[int, Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[str, Any]:
    del requirement
    selected = normalize_required_sections(selected_sections)
    if set(fragments) != set(range(len(criteria))):
        raise ValueError("DETAILED_PLAN_NO_PROGRESS: unfinished acceptance criteria")

    by_section: dict[str, list[dict[str, Any]]] = {
        section: [] for section in selected
    }
    for index in range(len(criteria)):
        fragment = validate_criterion_fragment(
            fragments[index],
            selected_sections=selected,
            allowed_refs=allowed_refs,
        )
        for row in fragment[_FRAGMENT_KEY]:
            by_section[row["section"]].append(row)

    missing = [
        section for section, rows in by_section.items() if len(rows) != len(criteria)
    ]
    if missing:
        raise MissingWorksheetSections(missing)

    worksheet: dict[str, Any] = {}
    for section, rows in by_section.items():
        specification: dict[str, Any] = {}
        reasons: list[dict[str, str]] = []
        refs: list[str] = []
        for concern in DETAIL_RECORDS[section]:
            merged: list[dict[str, Any]] = []
            seen: set[str] = set()
            for row in rows:
                for record in row["specification"][concern]:
                    key = json.dumps(record, sort_keys=True, ensure_ascii=False)
                    if key not in seen:
                        merged.append(deepcopy(record))
                        seen.add(key)
            specification[concern] = merged
            if not merged:
                authored = [
                    reason["reason"]
                    for row in rows
                    for reason in row["specification"]["inapplicable_concerns"]
                    if reason["concern"] == concern
                ]
                reasons.append(
                    {
                        "concern": concern,
                        "reason": " | ".join(dict.fromkeys(authored)),
                    }
                )
        for row in rows:
            refs.extend(
                ref
                for ref in row["constraint_evidence_refs"]
                if ref not in refs
            )
        specification["inapplicable_concerns"] = reasons
        worksheet[section] = validate_worksheet_section(
            {
                "specification": specification,
                "constraint_evidence_refs": refs,
            },
            allowed_refs,
            section,
        )
    return worksheet


__all__ = [
    "MissingWorksheetSections",
    "assemble_worksheet_from_fragments",
    "clear_requirement_progress",
    "generate_criterion_fragment",
    "generate_section_records",
    "load_requirement_progress",
    "requirement_acceptance_criteria",
    "store_criterion_progress",
    "validate_criterion_fragment",
]
