"""Assemble only authored fixed records; prose is never promoted into contracts."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any
import json

from jsonschema import Draft202012Validator

from .acceptance_contracts import canonical_public_acceptance, project_requirement_public_acceptance
from .fixed_template_generation import generate_fixed_template_value
from .model_output_atomicity_contract import MAX_MODEL_FIELDS
from .planning_detail_slots import DETAIL_RECORDS
from .planning_detail_template import normalize_required_sections, validate_worksheet_section
from .task_template_catalog import load_record_template, load_template
from .task_template_input import task_binding, task_context
from .task_template_runner import TemplateBlocked, record_response_schema

_PROGRESS_SCHEMA = "mmm/detail-criterion-records"
_PROGRESS_KEY = "detail_progress"
_FRAGMENT_KEY = "section_updates"


def _text(value):
    return " ".join(str(value or "").split()).strip()


def requirement_acceptance_criteria(requirement):
    return canonical_public_acceptance(requirement.get("acceptance")) or (
        project_requirement_public_acceptance(requirement),
    )


def _contains_blank_string(value: Any, schema: Mapping[str, Any] | None = None) -> bool:
    schema = schema or {}
    if isinstance(value, str):
        return not value.strip() and schema.get("minLength", 1) > 0
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        return any(
            _contains_blank_string(item, properties.get(key))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_blank_string(item, schema.get("items")) for item in value)
    return False


def _grounded_evidence_refs(context: Mapping[str, Any], allowed_refs: set[str]) -> list[str]:
    grounded: list[str] = []
    for key in ("source_evidence_id", "evidence_ref", "shard_id"):
        ref = context.get(key)
        if isinstance(ref, str) and ref in allowed_refs and ref not in grounded:
            grounded.append(ref)
    evidence = context.get("evidence")
    if isinstance(evidence, str) and evidence in allowed_refs and evidence not in grounded:
        grounded.append(evidence)
    elif isinstance(evidence, (list, tuple, set)):
        for ref in evidence:
            if isinstance(ref, str) and ref in allowed_refs and ref not in grounded:
                grounded.append(ref)
    return grounded


def _concern_batches(identifiers: list[str]):
    width = max(1, int(MAX_MODEL_FIELDS))
    for start in range(0, len(identifiers), width):
        yield identifiers[start : start + width]


def _run_concern_batch(
    router,
    identifiers: list[str],
    *,
    context: Mapping[str, Any],
    allowed_refs: set[str],
    progress: Mapping[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, dict[str, Any]]:
    """Advance up to MAX_MODEL_FIELDS independent concern state machines per model call.

    Each concern keeps the exact legacy task binding and checkpoint response array. Existing
    checkpoints therefore replay without a model call. Only concerns that need a new response
    are included in the next fixed-template batch; completed concerns immediately leave it.
    """
    if not identifiers or len(identifiers) > MAX_MODEL_FIELDS:
        raise ValueError(
            f"DETAIL_CONCERN_BATCH: expected 1..{MAX_MODEL_FIELDS} identifiers"
        )

    states: dict[str, dict[str, Any]] = {}
    slots: dict[str, str] = {}
    for index, identifier in enumerate(identifiers):
        template = load_record_template(identifier)
        normalized = task_context(template, context)
        binding = task_binding(template, normalized, allowed_refs)
        saved = deepcopy((progress or {}).get(binding, []))
        if not isinstance(saved, list):
            raise ValueError("TEMPLATE_PROGRESS: expected response array")
        slot = f"c{index}"
        slots[slot] = identifier
        states[identifier] = {
            "slot": slot,
            "template": template,
            "context": normalized,
            "binding": binding,
            "saved": saved,
            "accepted": [],
            "records": [],
            "refs": [],
            "seen": set(),
            "done": False,
            "reason": "",
        }

    while not all(bool(state["done"]) for state in states.values()):
        generated: dict[str, Any] = {}
        need_generation: list[str] = []

        for identifier, state in states.items():
            if state["done"]:
                continue
            accepted = state["accepted"]
            saved = state["saved"]
            if len(accepted) < len(saved):
                generated[state["slot"]] = deepcopy(saved[len(accepted)])
            else:
                need_generation.append(identifier)

        if need_generation:
            response_properties: dict[str, Any] = {}
            user_payload: dict[str, Any] = {}
            instructions: list[str] = [
                "Advance each independent Minecraft engineering concern by exactly one fixed-template transition.",
                "For each slot, emit one of record/done/not_applicable/blocked and obey only that slot's task and rules.",
                "Do not merge concerns, invent neighboring work, or repeat an already accepted record.",
            ]
            required_slots: list[str] = []
            for identifier in need_generation:
                state = states[identifier]
                slot = state["slot"]
                template = state["template"]
                response_properties[slot] = record_response_schema(template)
                required_slots.append(slot)
                user_payload[slot] = {
                    "template_id": identifier,
                    **state["context"],
                    "accepted_records": state["records"],
                    "allowed_evidence_refs": sorted(allowed_refs),
                }
                instructions.append(
                    f"[{slot}] {identifier}: {template['task']}\n"
                    + "\n".join(str(rule) for rule in template.get("rules", ()))
                )

            response_schema = {
                "type": "object",
                "properties": response_properties,
                "required": required_slots,
                "additionalProperties": False,
            }
            fresh = generate_fixed_template_value(
                router,
                "planner",
                [
                    {"role": "system", "content": "\n\n".join(instructions)},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                response_schema=response_schema,
                enable_tools=False,
                tool_name="submit_detail_concern_batch",
            )
            Draft202012Validator(response_schema).validate(fresh)
            generated.update(fresh)

        for identifier, state in states.items():
            if state["done"]:
                continue
            slot = state["slot"]
            if slot not in generated:
                raise ValueError(f"DETAIL_CONCERN_BATCH: missing response for {identifier}")
            value = deepcopy(generated[slot])
            validator = Draft202012Validator(record_response_schema(state["template"]))
            validator.validate(value)

            evidence_from_value = value.pop("evidence_refs", None)
            if evidence_from_value is not None:
                value["evidence_refs"] = list(evidence_from_value)
            else:
                value["evidence_refs"] = _grounded_evidence_refs(
                    state["context"], allowed_refs
                )
            if any(ref not in allowed_refs for ref in value["evidence_refs"]):
                raise ValueError(f"TEMPLATE_EVIDENCE: unknown evidence in {identifier}")

            status = value["status"]
            record = value["record"]
            reason = value["reason"].strip()
            if status == "record":
                if record is None or _contains_blank_string(
                    record, state["template"]["record_schema"]
                ):
                    raise ValueError(f"TEMPLATE_RECORD: empty record in {identifier}")
                key = json.dumps(record, sort_keys=True, ensure_ascii=False)
                if key in state["seen"]:
                    raise TemplateBlocked(
                        f"TEMPLATE_NO_PROGRESS: repeated record in {identifier}"
                    )
                state["seen"].add(key)
                state["records"].append(record)
                state["refs"].extend(
                    ref for ref in value["evidence_refs"] if ref not in state["refs"]
                )
            elif record is not None:
                raise ValueError(f"TEMPLATE_STATUS: {status} cannot carry a record")
            elif status == "blocked":
                raise TemplateBlocked(
                    f"TEMPLATE_BLOCKED: {identifier}: {reason or 'missing blocking reason'}"
                )
            elif status == "done" and state["records"]:
                state["refs"].extend(
                    ref for ref in value["evidence_refs"] if ref not in state["refs"]
                )
                state["done"] = True
            elif status == "not_applicable" and not state["records"] and reason:
                state["refs"] = value["evidence_refs"]
                state["reason"] = reason
                state["done"] = True
            else:
                raise ValueError(
                    f"TEMPLATE_STATUS: invalid {status} transition in {identifier}"
                )

            state["accepted"].append(deepcopy(value))
            replaying = len(state["accepted"]) <= len(state["saved"])
            if state["done"] and len(state["accepted"]) < len(state["saved"]):
                raise ValueError("TEMPLATE_PROGRESS: responses after completion")
            if not replaying and checkpoint is not None:
                checkpoint(state["binding"], deepcopy(state["accepted"]))

    return {
        identifier: {
            "records": deepcopy(state["records"]),
            "reason": state["reason"],
            "evidence_refs": list(state["refs"]),
        }
        for identifier, state in states.items()
    }


def generate_section_records(
    router,
    *,
    requirement,
    criterion,
    section,
    evidence,
    allowed_refs,
    progress=None,
    checkpoint=None,
):
    specification, refs, reasons = {}, [], []
    identifiers = list(load_template(f"feature/{section}")["steps"])
    context = {
        "requirement_id": requirement.get("requirement_id", ""),
        "requirement": requirement.get("statement", ""),
        "criterion": criterion,
        "evidence": evidence,
    }
    for identifiers_batch in _concern_batches(identifiers):
        batch_results = _run_concern_batch(
            router,
            identifiers_batch,
            context=context,
            allowed_refs=allowed_refs,
            progress=progress,
            checkpoint=checkpoint,
        )
        for identifier in identifiers_batch:
            concern = identifier.rsplit("/", 1)[1]
            result = batch_results[identifier]
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


def validate_criterion_fragment(value, *, selected_sections, allowed_refs):
    selected = normalize_required_sections(selected_sections)
    if (
        not isinstance(value, Mapping)
        or set(value) != {_FRAGMENT_KEY}
        or not isinstance(value[_FRAGMENT_KEY], list)
    ):
        raise ValueError("DETAILED_PLAN_CRITERION: expected authored section records")
    seen, rows = set(), []
    for row in value[_FRAGMENT_KEY]:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"section", "specification", "constraint_evidence_refs"}
        ):
            raise ValueError(
                "DETAILED_PLAN_CRITERION: prose or malformed records cannot certify a section"
            )
        section = row["section"]
        if section not in selected or section in seen:
            raise ValueError("DETAILED_PLAN_CRITERION: unknown or duplicate section")
        seen.add(section)
        validated = validate_worksheet_section(
            {k: row[k] for k in ("specification", "constraint_evidence_refs")},
            allowed_refs,
            section,
        )
        rows.append({"section": section, **validated})
    if not rows:
        raise ValueError("DETAILED_PLAN_NO_PROGRESS: no authored concern records")
    return {_FRAGMENT_KEY: rows}


def generate_criterion_fragment(
    router,
    *,
    requirement,
    criterion,
    selected_sections,
    evidence,
    allowed_refs,
    progress=None,
    checkpoint=None,
):
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
        super().__init__(
            "DETAILED_PLAN_MISSING_SECTIONS: " + ", ".join(self.sections)
        )


def assemble_worksheet_from_fragments(
    requirement,
    *,
    selected_sections,
    criteria,
    fragments,
    allowed_refs,
):
    selected = normalize_required_sections(selected_sections)
    if set(fragments) != set(range(len(criteria))):
        raise ValueError("DETAILED_PLAN_NO_PROGRESS: unfinished acceptance criteria")
    by_section = {section: [] for section in selected}
    for index in range(len(criteria)):
        fragment = validate_criterion_fragment(
            fragments[index], selected_sections=selected, allowed_refs=allowed_refs
        )
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
                authored = [
                    r["reason"]
                    for row in rows
                    for r in row["specification"]["inapplicable_concerns"]
                    if r["concern"] == concern
                ]
                reasons.append(
                    {
                        "concern": concern,
                        "reason": " | ".join(dict.fromkeys(authored)),
                    }
                )
        for row in rows:
            refs.extend(
                ref for ref in row["constraint_evidence_refs"] if ref not in refs
            )
        spec["inapplicable_concerns"] = reasons
        worksheet[section] = validate_worksheet_section(
            {"specification": spec, "constraint_evidence_refs": refs},
            allowed_refs,
            section,
        )
    return worksheet
