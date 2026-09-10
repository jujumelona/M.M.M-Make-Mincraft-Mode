"""Assemble only authored fixed records; prose is never promoted into contracts."""
from __future__ import annotations
from collections.abc import Mapping, Iterable
from typing import Any
from copy import deepcopy
import json

from .acceptance_contracts import canonical_public_acceptance, project_requirement_public_acceptance
from .planning_detail_slots import DETAIL_RECORDS
from .planning_detail_template import normalize_required_sections, validate_worksheet_section
from .task_template_catalog import load_template
from .task_template_runner import run_record_template

_PROGRESS_SCHEMA = "mmm/detail-criterion-records"
_PROGRESS_KEY = "detail_progress"
_FRAGMENT_KEY = "section_updates"


def _text(value):
    return " ".join(str(value or "").split()).strip()


def requirement_acceptance_criteria(requirement):
    return canonical_public_acceptance(requirement.get("acceptance")) or (project_requirement_public_acceptance(requirement),)


def generate_section_records(router, *, requirement, criterion, section, evidence, allowed_refs):
    specification, refs, reasons = {}, [], []
    for identifier in load_template(f"feature/{section}")["steps"]:
        concern = identifier.rsplit("/", 1)[1]
        result = run_record_template(router, identifier, context={
            "requirement": requirement.get("statement", ""),
            "criterion": criterion, "evidence": evidence,
        }, allowed_refs=allowed_refs)
        specification[concern] = result["records"]
        if not result["records"]:
            reasons.append({"concern": concern, "reason": result["reason"]})
        refs.extend(ref for ref in result["evidence_refs"] if ref not in refs)
    specification["inapplicable_concerns"] = reasons
    row = validate_worksheet_section({"specification": specification, "constraint_evidence_refs": refs}, allowed_refs, section)
    return {"section": section, **row}


def validate_criterion_fragment(value, *, selected_sections, allowed_refs):
    selected = normalize_required_sections(selected_sections)
    if not isinstance(value, Mapping) or set(value) != {_FRAGMENT_KEY} or not isinstance(value[_FRAGMENT_KEY], list):
        raise ValueError("DETAILED_PLAN_CRITERION: expected authored section records")
    seen, rows = set(), []
    for row in value[_FRAGMENT_KEY]:
        if not isinstance(row, Mapping) or set(row) != {"section", "specification", "constraint_evidence_refs"}:
            raise ValueError("DETAILED_PLAN_CRITERION: prose or malformed records cannot certify a section")
        section = row["section"]
        if section not in selected or section in seen:
            raise ValueError("DETAILED_PLAN_CRITERION: unknown or duplicate section")
        seen.add(section)
        validated = validate_worksheet_section({k: row[k] for k in ("specification", "constraint_evidence_refs")}, allowed_refs, section)
        rows.append({"section": section, **validated})
    if not rows:
        raise ValueError("DETAILED_PLAN_NO_PROGRESS: no authored concern records")
    return {_FRAGMENT_KEY: rows}


def generate_criterion_fragment(router, *, requirement, criterion, selected_sections, evidence, allowed_refs):
    selected = normalize_required_sections(selected_sections)
    return {_FRAGMENT_KEY: [generate_section_records(router, requirement=requirement,
        criterion=criterion, section=section, evidence=evidence, allowed_refs=allowed_refs)
        for section in selected]}


def generate_criterion_fragments_batch(router, *, requirement, criteria, selected_sections, evidence, allowed_refs):
    # Host aggregation only. Each model call still authors one concern record.
    return {index: generate_criterion_fragment(router, requirement=requirement,
        criterion=criterion, selected_sections=selected_sections, evidence=evidence,
        allowed_refs=allowed_refs) for index, criterion in criteria.items()}


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
            int(row.get("criterion_index", -1)) if isinstance(row, Mapping) else -1,
        )
    )
    value[_PROGRESS_KEY] = kept
    return value


def clear_requirement_progress(state: Mapping[str, Any], requirement_ref: str) -> dict[str, Any]:
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
    def __init__(self, sections):
        self.sections = tuple(sections)
        super().__init__("DETAILED_PLAN_MISSING_SECTIONS: " + ", ".join(self.sections))


def assemble_worksheet_from_fragments(requirement, *, selected_sections, criteria, fragments, allowed_refs):
    selected = normalize_required_sections(selected_sections)
    if set(fragments) != set(range(len(criteria))):
        raise ValueError("DETAILED_PLAN_NO_PROGRESS: unfinished acceptance criteria")
    by_section = {section: [] for section in selected}
    for index in range(len(criteria)):
        fragment = validate_criterion_fragment(fragments[index], selected_sections=selected, allowed_refs=allowed_refs)
        for row in fragment[_FRAGMENT_KEY]:
            by_section[row["section"]].append(row)
    missing = [section for section, rows in by_section.items() if len(rows) != len(criteria)]
    if missing:
        raise MissingWorksheetSections(missing)
    worksheet = {}
    for section, rows in by_section.items():
        spec, reasons, refs = {}, [], []
        for concern in DETAIL_RECORDS[section]:
            merged, seen = [], set()
            for row in rows:
                for record in row["specification"][concern]:
                    key = json.dumps(record, sort_keys=True, ensure_ascii=False)
                    if key not in seen:
                        merged.append(deepcopy(record))
                        seen.add(key)
            spec[concern] = merged
            if not merged:
                authored = [r["reason"] for row in rows for r in row["specification"]["inapplicable_concerns"] if r["concern"] == concern]
                # Every empty concern was explicitly resolved by every criterion.
                reasons.append({"concern": concern, "reason": " | ".join(dict.fromkeys(authored))})
        for row in rows:
            refs.extend(ref for ref in row["constraint_evidence_refs"] if ref not in refs)
        spec["inapplicable_concerns"] = reasons
        worksheet[section] = validate_worksheet_section({"specification": spec, "constraint_evidence_refs": refs}, allowed_refs, section)
    return worksheet
