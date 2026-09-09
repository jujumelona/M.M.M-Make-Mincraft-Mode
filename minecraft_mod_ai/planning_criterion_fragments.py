from __future__ import annotations

"""Atomic acceptance-criterion planning contracts and deterministic worksheet assembly.

The model never authors the full engineering worksheet. One already-approved public
acceptance criterion is the semantic work unit. Its model-facing schema stays bounded even
when every worksheet section applies, while the host merges criterion fragments into the
canonical detailed-plan worksheet required by downstream coding.
"""

from collections.abc import Iterable, Mapping
from copy import deepcopy
import json
from typing import Any

from .acceptance_contracts import (
    canonical_public_acceptance,
    project_requirement_public_acceptance,
)
from .model_output_atomicity_contract import assert_atomic_model_schema
from .planning_detail_slots import DETAIL_RECORDS
from .planning_detail_template import (
    _PLACEHOLDERS,
    _section_description,
    normalize_required_sections,
    validate_worksheet_section,
)

_PROGRESS_SCHEMA = "mmm/detail-criterion-progress-v1"
_PROGRESS_KEY = "detail_progress"


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def requirement_acceptance_criteria(requirement: Mapping[str, Any]) -> tuple[str, ...]:
    criteria = canonical_public_acceptance(requirement.get("acceptance"))
    if criteria:
        return criteria
    return (project_requirement_public_acceptance(requirement),)


def _implementation_key(section: str) -> str:
    return f"{section}__implementation"


def _constraint_key(section: str) -> str:
    return f"{section}__constraint"


def _evidence_key(section: str) -> str:
    return f"{section}__evidence_refs"


def criterion_fragment_schema(selected_sections: Iterable[str]) -> dict[str, Any]:
    selected = normalize_required_sections(selected_sections)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for section in selected:
        for key, description in (
            (_implementation_key(section), f"Concrete {section} implementation contract for this acceptance criterion. Empty only when genuinely unrelated."),
            (_constraint_key(section), f"Concrete {section} boundary, failure rule, invariant, or limit. Empty only when no additional constraint applies."),
        ):
            properties[key] = {"type": "string", "description": description}
            required.append(key)
        evidence_key = _evidence_key(section)
        properties[evidence_key] = {
            "type": "array",
            "uniqueItems": True,
            "description": f"Host-supplied evidence IDs that constrain the {section} fragment.",
            "items": {"type": "string"},
        }
        required.append(evidence_key)
    schema = {
        "type": "object",
        "description": "Atomic implementation contract for one approved public acceptance criterion.",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    assert_atomic_model_schema(schema, surface="one acceptance-criterion planning contract")
    return schema


def _evidence_context(evidence: list[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for item in evidence:
        refs = ", ".join(_text(ref) for ref in item.get("evidence_refs", []) if _text(ref))
        claims = item.get("claims") or []
        claim_text = " | ".join(_text(claim) for claim in claims if _text(claim))
        rows.append(
            f"- research_ref={_text(item.get('research_ref'))}; evidence_refs=[{refs}]; "
            f"claims={claim_text or 'no claim prose'}"
        )
    return "\n".join(rows) or "- no external constraint; author only grounded design decisions"


def criterion_fragment_messages(
    requirement: Mapping[str, Any],
    criterion: str,
    selected_sections: Iterable[str],
    evidence: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    selected = normalize_required_sections(selected_sections)
    section_guidance = "\n".join(
        f"- {section}: {_section_description(section)}" for section in selected
    )
    field_guidance = "\n".join(
        (
            f"- {_implementation_key(section)}: concrete design needed for this criterion in {section}.\n"
            f"- {_constraint_key(section)}: corresponding boundary/failure/invariant/limit.\n"
            f"- {_evidence_key(section)}: only host evidence IDs below that constrain this section."
        )
        for section in selected
    )
    return [
        {
            "role": "system",
            "content": (
                "Complete exactly one approved public acceptance criterion as one bounded engineering contract. "
                "Return only the JSON object required by the supplied schema. The host assembles the larger worksheet. "
                "Do not emit analysis, markdown, extra keys, JSON Schema definitions, TODO/TBD, or invented APIs, "
                "symbols, versions, repository paths, external facts, or evidence IDs. Keep fields concise and concrete. "
                "Empty strings are permitted only when this criterion genuinely has no bearing on that section."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requirement: {_text(requirement.get('statement'))}\n"
                f"Acceptance criterion: {criterion}\n\n"
                "Grounded implementation evidence:\n"
                f"{_evidence_context(evidence)}\n\n"
                "Selected worksheet section purposes:\n"
                f"{section_guidance}\n\n"
                "Fill these exact fields:\n"
                f"{field_guidance}"
            ),
        },
    ]


def validate_criterion_fragment(
    value: Any,
    *,
    selected_sections: Iterable[str],
    allowed_refs: set[str],
) -> dict[str, Any]:
    selected = normalize_required_sections(selected_sections)
    schema = criterion_fragment_schema(selected)
    expected = set(schema["properties"])
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("DETAILED_PLAN_CRITERION: fragment must contain exactly the host-selected fields")
    normalized: dict[str, Any] = {}
    meaningful = False
    for section in selected:
        for key in (_implementation_key(section), _constraint_key(section)):
            raw = value.get(key)
            if not isinstance(raw, str):
                raise ValueError(f"DETAILED_PLAN_CRITERION: {key} must be a string")
            text = _text(raw)
            if text.casefold() in _PLACEHOLDERS:
                text = ""
            normalized[key] = text
            meaningful = meaningful or bool(text)
        evidence_key = _evidence_key(section)
        raw_refs = value.get(evidence_key)
        if not isinstance(raw_refs, list):
            raise ValueError(f"DETAILED_PLAN_CRITERION: {evidence_key} must be an array")
        normalized[evidence_key] = list(
            dict.fromkeys(
                _text(ref)
                for ref in raw_refs
                if isinstance(ref, str) and _text(ref) in allowed_refs
            )
        )
    if not meaningful:
        raise ValueError("DETAILED_PLAN_NO_PROGRESS: acceptance criterion produced no implementation content")
    return normalized


def generate_criterion_fragment(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    criterion: str,
    selected_sections: Iterable[str],
    evidence: list[Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[str, Any]:
    selected = normalize_required_sections(selected_sections)
    schema = criterion_fragment_schema(selected)
    raw = router.generate_text(
        "planner",
        criterion_fragment_messages(requirement, criterion, selected, evidence),
        response_format="json",
        response_schema=schema,
        enable_tools=False,
    )
    decoded = json.loads(raw)
    return validate_criterion_fragment(decoded, selected_sections=selected, allowed_refs=allowed_refs)


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
            row.get("fragment"), selected_sections=selected, allowed_refs=allowed_refs
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


def _append_section_records(
    section: str,
    specification: dict[str, list[dict[str, str]]],
    *,
    requirement_statement: str,
    criterion: str,
    implementation: str,
    constraint: str,
) -> None:
    if section == "behavior_contract":
        if implementation:
            specification["success_postconditions"].append({"condition": criterion, "observation": implementation})
        if constraint:
            specification["rejection_postconditions"].append({"condition": constraint, "preserved_state": "state before the rejected criterion path", "observation": constraint})
    elif section == "state_model":
        if implementation:
            specification["invariants"].append({"condition": criterion, "enforcement": implementation})
        if constraint:
            specification["transitions"].append({"from_state": "current valid requirement state", "trigger": criterion, "guard": constraint, "mutation": implementation or constraint, "to_state": "criterion-satisfying requirement state"})
    elif section == "algorithm":
        if implementation:
            specification["steps"].append({"operation": implementation, "input": criterion, "output": "observable criterion result", "next_step": "evaluate the criterion verification contract"})
        if constraint:
            specification["edge_cases"].append({"condition": constraint, "result": "apply the declared constraint without a partial state commit"})
    elif section == "integration":
        if implementation:
            specification["responsibilities"].append({"caller": "requirement implementation", "callee": "selected integration boundary", "contract": implementation})
        if constraint:
            specification["compatibility"].append({"assumption": constraint, "extension_point": "grounded target integration boundary"})
    elif section == "authority_and_network":
        if implementation:
            specification["decisions"].append({"decision": implementation, "authoritative_side": constraint or "authority defined by this criterion contract"})
        if constraint:
            specification["synchronization"].append({"state": implementation or criterion, "recipients": "observers authorized by the criterion authority contract", "trigger": constraint})
    elif section == "persistence":
        if implementation:
            specification["stored_state"].append({"state": implementation, "owner": "requirement implementation", "scope": criterion})
        if constraint:
            specification["save_triggers"].append({"trigger": constraint, "dirty_rule": "mark dirty when criterion-owned persisted state changes", "action": implementation or "persist the criterion-owned state"})
    elif section == "resources_and_ui":
        if implementation:
            specification["data_resources"].append({"kind": "criterion resource or presentation contract", "purpose": implementation, "owner": "requirement implementation"})
        if constraint:
            specification["interactions"].append({"trigger": criterion, "server_action": constraint, "client_feedback": implementation or constraint})
    elif section == "failure_and_limits":
        if implementation:
            specification["invalid_inputs"].append({"condition": implementation, "rejection": constraint or "reject the invalid criterion path", "preserved_state": "pre-failure requirement state"})
        if constraint:
            specification["bounds"].append({"resource": criterion, "limit": constraint, "unit": "criterion-defined boundary", "enforcement": "fail closed at the declared boundary"})
    elif section == "reuse_assessment":
        if implementation:
            specification["verdicts"].append({"candidate": criterion, "verdict": implementation, "reason": constraint or implementation})
        if constraint:
            specification["evidence_gaps"].append({"missing_fact": constraint, "blocked_reuse": "do not claim reuse beyond the grounded criterion evidence"})
    elif section == "verification":
        if implementation:
            specification["success_cases"].append({"given": requirement_statement, "when": criterion, "then": implementation, "measurement": constraint or criterion})
        if constraint:
            specification["failure_cases"].append({"given": requirement_statement, "when": constraint, "then": "the criterion must not be falsely accepted", "measurement": constraint})
    else:
        raise ValueError(f"DETAILED_PLAN_CRITERION: unsupported section {section!r}")


def assemble_worksheet_from_fragments(
    requirement: Mapping[str, Any],
    *,
    selected_sections: Iterable[str],
    criteria: tuple[str, ...],
    fragments: Mapping[int, Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[str, dict[str, Any]]:
    selected = normalize_required_sections(selected_sections)
    if set(fragments) != set(range(len(criteria))):
        raise ValueError("DETAILED_PLAN_NO_PROGRESS: cannot assemble worksheet with unfinished acceptance criteria")
    worksheet: dict[str, dict[str, Any]] = {}
    requirement_statement = _text(requirement.get("statement"))
    for section in selected:
        specification: dict[str, list[dict[str, str]]] = {
            concern: [] for concern in DETAIL_RECORDS[section]
        }
        section_refs: list[str] = []
        for index, criterion in enumerate(criteria):
            fragment = validate_criterion_fragment(
                fragments[index], selected_sections=selected, allowed_refs=allowed_refs
            )
            implementation = _text(fragment[_implementation_key(section)])
            constraint = _text(fragment[_constraint_key(section)])
            _append_section_records(
                section,
                specification,
                requirement_statement=requirement_statement,
                criterion=criterion,
                implementation=implementation,
                constraint=constraint,
            )
            for ref in fragment[_evidence_key(section)]:
                if ref not in section_refs:
                    section_refs.append(ref)
        inapplicable = [
            {
                "concern": concern,
                "reason": (
                    f"No {concern.replace('_', ' ')} is required by the completed "
                    f"acceptance-criterion contracts for {section.replace('_', ' ')}."
                ),
            }
            for concern, records in specification.items()
            if not records
        ]
        specification["inapplicable_concerns"] = inapplicable
        worksheet[section] = validate_worksheet_section(
            {"specification": specification, "constraint_evidence_refs": section_refs},
            allowed_refs,
            section,
        )
    return worksheet


__all__ = [
    "assemble_worksheet_from_fragments",
    "clear_requirement_progress",
    "criterion_fragment_messages",
    "criterion_fragment_schema",
    "generate_criterion_fragment",
    "load_requirement_progress",
    "requirement_acceptance_criteria",
    "store_criterion_progress",
    "validate_criterion_fragment",
]