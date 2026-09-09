from __future__ import annotations

"""Atomic acceptance-criterion planning contracts and deterministic worksheet assembly.

The model never authors the full engineering worksheet. One already-approved public
acceptance criterion is the semantic work unit. The model emits one compact
``section_updates`` array with exactly one bounded record for each host-selected worksheet
section; the host validates section identity, checkpoints the fragment, and deterministically
assembles the canonical detailed-plan worksheet required by downstream coding.
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
_FRAGMENT_KEY = "section_updates"
_FRAGMENT_FIELDS = frozenset({"section", "implementation", "constraint", "evidence_refs"})
_NO_PROGRESS_FRAGMENT_ERROR = (
    "DETAILED_PLAN_NO_PROGRESS: acceptance criterion produced no implementation content"
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def requirement_acceptance_criteria(requirement: Mapping[str, Any]) -> tuple[str, ...]:
    criteria = canonical_public_acceptance(requirement.get("acceptance"))
    if criteria:
        return criteria
    return (project_requirement_public_acceptance(requirement),)


def criterion_fragment_schema(selected_sections: Iterable[str]) -> dict[str, Any]:
    """Return one compact schema whose topology is independent of section count.

    Selected section names live in an enum inside one array item schema instead of being
    expanded into three top-level properties per section. This keeps the model contract
    below the global atomic structured-output boundary even when all worksheet sections
    apply, while host validation still requires exactly one update for every selected
    section.
    """

    selected = normalize_required_sections(selected_sections)
    schema = {
        "type": "object",
        "description": "Atomic implementation contract for one approved public acceptance criterion.",
        "properties": {
            _FRAGMENT_KEY: {
                "type": "array",
                "description": "Exactly one concise engineering update for every host-selected worksheet section.",
                "minItems": 1,
                "maxItems": len(selected),
                "items": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": list(selected),
                        },
                        "implementation": {
                            "type": "string",
                            "description": "Concrete implementation contract for this criterion in this section; empty only when genuinely unrelated.",
                        },
                        "constraint": {
                            "type": "string",
                            "description": "Concrete boundary, failure rule, invariant, or limit for this criterion in this section; empty only when none applies.",
                        },
                        "evidence_refs": {
                            "type": "array",
                            "uniqueItems": True,
                            "description": "Only host-supplied evidence IDs that constrain this section update.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["section", "implementation", "constraint", "evidence_refs"],
                    "additionalProperties": False,
                },
            }
        },
        "required": [_FRAGMENT_KEY],
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
    *,
    repair_no_progress: bool = False,
) -> list[dict[str, str]]:
    selected = normalize_required_sections(selected_sections)
    section_guidance = "\n".join(
        f"- {section}: {_section_description(section)}" for section in selected
    )
    selected_text = ", ".join(selected)
    repair_instruction = ""
    if repair_no_progress:
        repair_instruction = (
            "\n\nCorrection required: the previous response was structurally valid but every "
            "implementation and constraint string was empty. Rewrite the contract from the "
            "criterion semantics. Keep unrelated fields empty, but make at least one selected "
            "section concrete. If an exact API is not grounded, describe the semantic state "
            "mutation, data flow, boundary, or observable verification without inventing symbols."
        )
    return [
        {
            "role": "system",
            "content": (
                "Complete exactly one approved public acceptance criterion as one bounded engineering contract. "
                "Return only the JSON object required by the supplied schema. The host assembles the larger worksheet. "
                "Do not emit analysis, markdown, extra keys, JSON Schema definitions, TODO/TBD, or invented APIs, "
                "symbols, versions, repository paths, external facts, or evidence IDs. Keep each update concise and concrete. "
                "Return exactly one section_updates record for every host-selected section, with no duplicate sections. "
                "Empty implementation or constraint strings are permitted only when this criterion genuinely has no bearing on that field. "
                "Across the complete response, at least one selected section MUST contain a non-empty implementation or constraint; "
                "an observable acceptance criterion must never be represented by an all-empty response."
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
                f"Required section_updates sections, exactly once each: {selected_text}\n"
                "For each record provide: section, implementation, constraint, evidence_refs."
                f"{repair_instruction}"
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
    criterion_fragment_schema(selected)
    if not isinstance(value, Mapping) or set(value) != {_FRAGMENT_KEY}:
        raise ValueError(
            "DETAILED_PLAN_CRITERION: fragment must contain exactly the section_updates array"
        )
    raw_updates = value.get(_FRAGMENT_KEY)
    if not isinstance(raw_updates, list):
        raise ValueError("DETAILED_PLAN_CRITERION: section_updates must be an array")

    by_section: dict[str, dict[str, Any]] = {}
    meaningful = False
    selected_set = set(selected)
    for raw_update in raw_updates:
        if not isinstance(raw_update, Mapping) or set(raw_update) != _FRAGMENT_FIELDS:
            raise ValueError(
                "DETAILED_PLAN_CRITERION: each section update must contain exactly "
                "section, implementation, constraint, evidence_refs"
            )
        section = _text(raw_update.get("section"))
        if section not in selected_set:
            raise ValueError(f"DETAILED_PLAN_CRITERION: unexpected section {section!r}")
        if section in by_section:
            raise ValueError(f"DETAILED_PLAN_CRITERION: duplicate section {section!r}")

        normalized_text: dict[str, str] = {}
        for field in ("implementation", "constraint"):
            raw = raw_update.get(field)
            if not isinstance(raw, str):
                raise ValueError(f"DETAILED_PLAN_CRITERION: {section}.{field} must be a string")
            text = _text(raw)
            if text.casefold() in _PLACEHOLDERS:
                text = ""
            normalized_text[field] = text
            meaningful = meaningful or bool(text)

        raw_refs = raw_update.get("evidence_refs")
        if not isinstance(raw_refs, list):
            raise ValueError(
                f"DETAILED_PLAN_CRITERION: {section}.evidence_refs must be an array"
            )
        refs = list(
            dict.fromkeys(
                _text(ref)
                for ref in raw_refs
                if isinstance(ref, str) and _text(ref) in allowed_refs
            )
        )
        by_section[section] = {
            "section": section,
            "implementation": normalized_text["implementation"],
            "constraint": normalized_text["constraint"],
            "evidence_refs": refs,
        }

    if set(by_section) != selected_set:
        missing = [section for section in selected if section not in by_section]
        raise ValueError(
            "DETAILED_PLAN_CRITERION: section_updates must cover every selected section exactly once; "
            f"missing={missing}"
        )
    if not meaningful:
        raise ValueError(_NO_PROGRESS_FRAGMENT_ERROR)
    return {_FRAGMENT_KEY: [by_section[section] for section in selected]}


def _generate_criterion_fragment_once(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    criterion: str,
    selected: tuple[str, ...],
    evidence: list[Mapping[str, Any]],
    allowed_refs: set[str],
    schema: Mapping[str, Any],
    repair_no_progress: bool,
) -> dict[str, Any]:
    raw = router.generate_text(
        "planner",
        criterion_fragment_messages(
            requirement,
            criterion,
            selected,
            evidence,
            repair_no_progress=repair_no_progress,
        ),
        response_format="json",
        response_schema=schema,
        enable_tools=False,
    )
    decoded = json.loads(raw)
    return validate_criterion_fragment(
        decoded,
        selected_sections=selected,
        allowed_refs=allowed_refs,
    )


def generate_criterion_fragment(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    criterion: str,
    selected_sections: Iterable[str],
    evidence: list[Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[str, Any]:
    """Generate one criterion fragment with one bounded repair for all-empty output.

    The repair is deliberately narrow: only a schema-valid fragment that contains no
    implementation or constraint content gets one corrective call. JSON failures, section
    contract violations, and every other error remain terminal. A second all-empty response
    also propagates immediately, so this cannot become a count-driven or open-ended retry loop.
    """

    selected = normalize_required_sections(selected_sections)
    schema = criterion_fragment_schema(selected)
    try:
        return _generate_criterion_fragment_once(
            router,
            requirement=requirement,
            criterion=criterion,
            selected=selected,
            evidence=evidence,
            allowed_refs=allowed_refs,
            schema=schema,
            repair_no_progress=False,
        )
    except ValueError as exc:
        if str(exc) != _NO_PROGRESS_FRAGMENT_ERROR:
            raise

    return _generate_criterion_fragment_once(
        router,
        requirement=requirement,
        criterion=criterion,
        selected=selected,
        evidence=evidence,
        allowed_refs=allowed_refs,
        schema=schema,
        repair_no_progress=True,
    )


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
            specification["success_postconditions"].append(
                {"condition": criterion, "observation": implementation}
            )
        if constraint:
            specification["rejection_postconditions"].append(
                {
                    "condition": constraint,
                    "preserved_state": "state before the rejected criterion path",
                    "observation": constraint,
                }
            )
    elif section == "state_model":
        if implementation:
            specification["invariants"].append(
                {"condition": criterion, "enforcement": implementation}
            )
        if constraint:
            specification["transitions"].append(
                {
                    "from_state": "current valid requirement state",
                    "trigger": criterion,
                    "guard": constraint,
                    "mutation": implementation or constraint,
                    "to_state": "criterion-satisfying requirement state",
                }
            )
    elif section == "algorithm":
        if implementation:
            specification["steps"].append(
                {
                    "operation": implementation,
                    "input": criterion,
                    "output": "observable criterion result",
                    "next_step": "evaluate the criterion verification contract",
                }
            )
        if constraint:
            specification["edge_cases"].append(
                {
                    "condition": constraint,
                    "result": "apply the declared constraint without a partial state commit",
                }
            )
    elif section == "integration":
        if implementation:
            specification["responsibilities"].append(
                {
                    "caller": "requirement implementation",
                    "callee": "selected integration boundary",
                    "contract": implementation,
                }
            )
        if constraint:
            specification["compatibility"].append(
                {
                    "assumption": constraint,
                    "extension_point": "grounded target integration boundary",
                }
            )
    elif section == "authority_and_network":
        if implementation:
            specification["decisions"].append(
                {
                    "decision": implementation,
                    "authoritative_side": constraint
                    or "authority defined by this criterion contract",
                }
            )
        if constraint:
            specification["synchronization"].append(
                {
                    "state": implementation or criterion,
                    "recipients": "observers authorized by the criterion authority contract",
                    "trigger": constraint,
                }
            )
    elif section == "persistence":
        if implementation:
            specification["stored_state"].append(
                {
                    "state": implementation,
                    "owner": "requirement implementation",
                    "scope": criterion,
                }
            )
        if constraint:
            specification["save_triggers"].append(
                {
                    "trigger": constraint,
                    "dirty_rule": "mark dirty when criterion-owned persisted state changes",
                    "action": implementation or "persist the criterion-owned state",
                }
            )
    elif section == "resources_and_ui":
        if implementation:
            specification["data_resources"].append(
                {
                    "kind": "criterion resource or presentation contract",
                    "purpose": implementation,
                    "owner": "requirement implementation",
                }
            )
        if constraint:
            specification["interactions"].append(
                {
                    "trigger": criterion,
                    "server_action": constraint,
                    "client_feedback": implementation or constraint,
                }
            )
    elif section == "failure_and_limits":
        if implementation:
            specification["invalid_inputs"].append(
                {
                    "condition": implementation,
                    "rejection": constraint or "reject the invalid criterion path",
                    "preserved_state": "pre-failure requirement state",
                }
            )
        if constraint:
            specification["bounds"].append(
                {
                    "resource": criterion,
                    "limit": constraint,
                    "unit": "criterion-defined boundary",
                    "enforcement": "fail closed at the declared boundary",
                }
            )
    elif section == "reuse_assessment":
        if implementation:
            specification["verdicts"].append(
                {
                    "candidate": criterion,
                    "verdict": implementation,
                    "reason": constraint or implementation,
                }
            )
        if constraint:
            specification["evidence_gaps"].append(
                {
                    "missing_fact": constraint,
                    "blocked_reuse": "do not claim reuse beyond the grounded criterion evidence",
                }
            )
    elif section == "verification":
        if implementation:
            specification["success_cases"].append(
                {
                    "given": requirement_statement,
                    "when": criterion,
                    "then": implementation,
                    "measurement": constraint or criterion,
                }
            )
        if constraint:
            specification["failure_cases"].append(
                {
                    "given": requirement_statement,
                    "when": constraint,
                    "then": "the criterion must not be falsely accepted",
                    "measurement": constraint,
                }
            )
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
        raise ValueError(
            "DETAILED_PLAN_NO_PROGRESS: cannot assemble worksheet with unfinished acceptance criteria"
        )
    worksheet: dict[str, dict[str, Any]] = {}
    requirement_statement = _text(requirement.get("statement"))
    validated_fragments = {
        index: validate_criterion_fragment(
            fragments[index],
            selected_sections=selected,
            allowed_refs=allowed_refs,
        )
        for index in range(len(criteria))
    }
    updates_by_criterion = {
        index: {
            update["section"]: update
            for update in validated_fragments[index][_FRAGMENT_KEY]
        }
        for index in range(len(criteria))
    }

    for section in selected:
        specification: dict[str, list[dict[str, str]]] = {
            concern: [] for concern in DETAIL_RECORDS[section]
        }
        section_refs: list[str] = []
        for index, criterion in enumerate(criteria):
            update = updates_by_criterion[index][section]
            implementation = _text(update["implementation"])
            constraint = _text(update["constraint"])
            _append_section_records(
                section,
                specification,
                requirement_statement=requirement_statement,
                criterion=criterion,
                implementation=implementation,
                constraint=constraint,
            )
            for ref in update["evidence_refs"]:
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
            {
                "specification": specification,
                "constraint_evidence_refs": section_refs,
            },
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
