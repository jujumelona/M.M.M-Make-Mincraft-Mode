from __future__ import annotations

from .fixed_template_generation import generate_fixed_template_value

"""Acceptance-criterion planning with tolerant model-output normalization.

One approved public acceptance criterion is the semantic work unit. The model only has
to describe the worksheet sections that actually matter to that criterion. Host code
normalizes small-model template drift, checkpoints useful progress, and deterministically
assembles the canonical worksheet required by downstream coding.
"""

from collections.abc import Iterable, Mapping
from copy import deepcopy
import json
from typing import Any

from .acceptance_contracts import (
    canonical_public_acceptance,
    project_requirement_public_acceptance,
)
from .planning_detail_slots import DETAIL_RECORDS
from .planning_mod_discovery import discovery_context
from .planning_detail_template import (
    _PLACEHOLDERS,
    _section_description,
    normalize_required_sections,
    validate_worksheet_section,
)

_PROGRESS_SCHEMA = "mmm/detail-criterion-progress-v1"
_PROGRESS_KEY = "detail_progress"
_FRAGMENT_KEY = "section_updates"
_CRITERION_TOOL_NAME = "submit_criterion_fragment"
_CRITERION_BATCH_TOOL_NAME = "submit_criterion_fragments"
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
    return {
        "type": "object",
        "description": "Atomic implementation contract for one approved public acceptance criterion.",
        "properties": {
            _FRAGMENT_KEY: {
                "type": "array",
                "description": "Concise engineering updates for host-selected worksheet sections.",
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
                    "required": ["section"],
                    "additionalProperties": False,
                },
            }
        },
        "required": [_FRAGMENT_KEY],
        "additionalProperties": False,
    }



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
        discovery = item.get("mod_discovery")
        if isinstance(discovery, Mapping) and discovery:
            rows.append("Catalog discovery: " + discovery_context(discovery))
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
    repair_instruction = ""
    if repair_no_progress:
        repair_instruction = (
            "\n\nCorrection required: the previous template fill contained no useful "
            "implementation or constraint. Fill at least one selected section concretely "
            "for this acceptance criterion. Do not add unrelated sections merely to "
            "fill the template."
        )

    return [
        {
            "role": "system",
            "content": (
                "Complete exactly one approved public acceptance criterion as a compact engineering contract. "
                "Fill the supplied fixed template fields; do not write serialization syntax. Include only worksheet "
                "sections that actually matter to this criterion; never manufacture content for unrelated sections. "
                "Each update needs a section name and may include implementation, constraint, and evidence_refs when "
                "those values exist. Do not emit TODO/TBD, invented APIs, symbols, versions, repository paths, "
                "external facts, or evidence IDs. At least one update must contain a concrete implementation or constraint."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requirement: {_text(requirement.get('statement'))}\n"
                f"Acceptance criterion: {criterion}\n\n"
                "Grounded implementation evidence:\n"
                f"{_evidence_context(evidence)}\n\n"
                "Available worksheet sections; choose only relevant ones:\n"
                f"{section_guidance}\n\n"
                "Fill section_updates using only the supplied template fields and allowed section names."
                f"{repair_instruction}"
            ),
        },
    ]


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, (Mapping, list, tuple)):
        if not value:
            return ""
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _text(value)


def _coerce_update(section: str, raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return {
            "section": section,
            "implementation": _text(raw),
            "constraint": "",
            "evidence_refs": [],
        }
    if not isinstance(raw, Mapping):
        return {
            "section": section,
            "implementation": _json_text(raw),
            "constraint": "",
            "evidence_refs": [],
        }

    implementation = raw.get("implementation")
    constraint = raw.get("constraint")
    refs = raw.get("evidence_refs", raw.get("constraint_evidence_refs", []))

    if implementation is None and constraint is None:
        # Compatibility with the old/direct worksheet-shaped model response. Preserve the
        # authored content instead of rejecting it merely because the wrapper was omitted.
        payload = raw.get("specification", raw)
        implementation = _json_text(payload)

    return {
        "section": section,
        "implementation": _json_text(implementation),
        "constraint": _json_text(constraint),
        "evidence_refs": refs if isinstance(refs, list) else [],
    }


def _raw_updates(value: Any, selected: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, Mapping):
        wrapped = value.get(_FRAGMENT_KEY)
        if isinstance(wrapped, list):
            raw_items = wrapped
        else:
            # Accept legacy/direct worksheet-shaped payloads when loading old checkpoints.
            return [
                _coerce_update(section, value[section])
                for section in selected
                if section in value
            ]
    else:
        raise ValueError("DETAILED_PLAN_CRITERION: fragment must be an object or array")

    updates: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        section = _text(raw.get("section"))
        if not section:
            continue
        updates.append(_coerce_update(section, raw))
    return updates


def validate_criterion_fragment(
    value: Any,
    *,
    selected_sections: Iterable[str],
    allowed_refs: set[str],
) -> dict[str, Any]:
    selected = normalize_required_sections(selected_sections)
    selected_set = set(selected)
    raw_updates = _raw_updates(value, selected)

    by_section: dict[str, dict[str, Any]] = {}
    meaningful = False
    for raw_update in raw_updates:
        section = _text(raw_update.get("section"))
        if section not in selected_set:
            continue

        implementation = _text(raw_update.get("implementation"))
        constraint = _text(raw_update.get("constraint"))
        if implementation.casefold() in _PLACEHOLDERS:
            implementation = ""
        if constraint.casefold() in _PLACEHOLDERS:
            constraint = ""
        meaningful = meaningful or bool(implementation or constraint)

        raw_refs = raw_update.get("evidence_refs")
        refs = list(
            dict.fromkeys(
                _text(ref)
                for ref in raw_refs
                if isinstance(raw_refs, list)
                and isinstance(ref, str)
                and _text(ref) in allowed_refs
            )
        ) if isinstance(raw_refs, list) else []

        current = by_section.get(section)
        if current is None:
            by_section[section] = {
                "section": section,
                "implementation": implementation,
                "constraint": constraint,
                "evidence_refs": refs,
            }
            continue

        # Duplicate section rows are merged rather than making otherwise-useful model work
        # terminal. This is deterministic and does not invent semantics.
        if implementation:
            current["implementation"] = " | ".join(
                part for part in (current["implementation"], implementation) if part
            )
        if constraint:
            current["constraint"] = " | ".join(
                part for part in (current["constraint"], constraint) if part
            )
        current["evidence_refs"] = list(
            dict.fromkeys([*current["evidence_refs"], *refs])
        )

    if not meaningful:
        raise ValueError(_NO_PROGRESS_FRAGMENT_ERROR)

    return {
        _FRAGMENT_KEY: [
            by_section[section]
            for section in selected
            if section in by_section
        ]
    }



def criterion_fragment_batch_schema(
    selected_sections: Iterable[str],
    criterion_indices: Iterable[int],
) -> dict[str, Any]:
    """Compact requirement-scoped transport for several acceptance criteria.

    The host still validates and checkpoints each criterion independently. Only the model
    transport is batched, eliminating repeated prompt/evidence/schema prefill on single-slot
    local runtimes.
    """
    selected = normalize_required_sections(selected_sections)
    indices = tuple(dict.fromkeys(int(index) for index in criterion_indices))
    if not indices:
        raise ValueError("DETAILED_PLAN_BATCH: criterion indices must not be empty")
    update_schema = deepcopy(criterion_fragment_schema(selected)["properties"][_FRAGMENT_KEY])
    return {
        "type": "object",
        "properties": {
            "criterion_fragments": {
                "type": "array",
                "minItems": len(indices),
                "maxItems": len(indices),
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_index": {"type": "integer", "enum": list(indices)},
                        _FRAGMENT_KEY: update_schema,
                    },
                    "required": ["criterion_index", _FRAGMENT_KEY],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["criterion_fragments"],
        "additionalProperties": False,
    }


def criterion_fragment_batch_messages(
    requirement: Mapping[str, Any],
    criteria: Mapping[int, str],
    selected_sections: Iterable[str],
    evidence: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    selected = normalize_required_sections(selected_sections)
    section_guidance = "\n".join(
        f"- {section}: {_section_description(section)}" for section in selected
    )
    criterion_rows = "\n".join(
        f"- criterion_index={index}: {criterion}"
        for index, criterion in sorted(criteria.items())
    )
    return [
        {
            "role": "system",
            "content": (
                "Complete the listed approved acceptance criteria for one requirement by filling the supplied "
                "fixed template. Fill exactly one criterion_fragments row per supplied criterion_index. Each row "
                "uses the same compact section_updates fields as a single criterion. Include only relevant worksheet "
                "sections. Do not write serialization syntax and do not emit TODO/TBD, invented APIs, symbols, versions, "
                "repository paths, external facts, or evidence IDs. Every criterion row must contain concrete "
                "implementation or constraint content. Do not merge criteria together."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requirement: {_text(requirement.get('statement'))}\n\n"
                "Acceptance criteria to complete:\n"
                f"{criterion_rows}\n\n"
                "Grounded implementation evidence shared by this requirement:\n"
                f"{_evidence_context(evidence)}\n\n"
                "Available worksheet sections; choose only relevant ones:\n"
                f"{section_guidance}\n\n"
                "Fill criterion_fragments with the exact supplied criterion_index values."
            ),
        },
    ]


def _fill_criterion_template(
    router: Any,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    *,
    batch: bool,
) -> Mapping[str, Any]:
    value = generate_fixed_template_value(
        router,
        "planner",
        messages,
        response_schema=schema,
        enable_tools=False,
        tool_name=_CRITERION_BATCH_TOOL_NAME if batch else _CRITERION_TOOL_NAME,
        description=(
            "Fill the fixed criterion-fragment batch template with exactly one row per supplied criterion_index."
            if batch
            else "Fill the fixed criterion-fragment template for the supplied acceptance criterion."
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError("DETAILED_PLAN_CRITERION: fixed template arguments must be an object")
    return value


def generate_criterion_fragments_batch(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    criteria: Mapping[int, str],
    selected_sections: Iterable[str],
    evidence: list[Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[int, dict[str, Any]]:
    """Generate missing criterion fragments with one requirement-scoped model call.

    A malformed or omitted row falls back only for that criterion, preserving the prior
    fail-closed single-criterion behavior without paying N model calls on the healthy path.
    """
    requested = {
        int(index): _text(criterion)
        for index, criterion in criteria.items()
        if _text(criterion)
    }
    if not requested:
        return {}
    if len(requested) == 1:
        index, criterion = next(iter(requested.items()))
        return {
            index: generate_criterion_fragment(
                router,
                requirement=requirement,
                criterion=criterion,
                selected_sections=selected_sections,
                evidence=evidence,
                allowed_refs=allowed_refs,
            )
        }

    selected = normalize_required_sections(selected_sections)
    schema = criterion_fragment_batch_schema(selected, requested)
    decoded = _fill_criterion_template(
        router,
        criterion_fragment_batch_messages(requirement, requested, selected, evidence),
        schema,
        batch=True,
    )
    rows = decoded.get("criterion_fragments")
    result: dict[int, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            index = row.get("criterion_index")
            if not isinstance(index, int) or index not in requested or index in result:
                continue
            try:
                result[index] = validate_criterion_fragment(
                    row,
                    selected_sections=selected,
                    allowed_refs=allowed_refs,
                )
            except ValueError:
                continue

    for index, criterion in requested.items():
        if index in result:
            continue
        result[index] = generate_criterion_fragment(
            router,
            requirement=requirement,
            criterion=criterion,
            selected_sections=selected,
            evidence=evidence,
            allowed_refs=allowed_refs,
        )
    return result


def generate_criterion_fragment(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    criterion: str,
    selected_sections: Iterable[str],
    evidence: list[Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[str, Any]:
    """Generate one criterion fragment with one bounded semantic repair.

    The model fills forced function arguments instead of authoring serialized JSON. Host
    validation remains tolerant of legacy checkpoint shapes while current model generation
    is syntax-independent and cannot fail merely because a small model omitted JSON syntax.
    """

    selected = normalize_required_sections(selected_sections)
    schema = criterion_fragment_schema(selected)
    decoded = _fill_criterion_template(
        router,
        criterion_fragment_messages(requirement, criterion, selected, evidence),
        schema,
        batch=False,
    )
    try:
        return validate_criterion_fragment(
            decoded,
            selected_sections=selected,
            allowed_refs=allowed_refs,
        )
    except ValueError as exc:
        if str(exc) != _NO_PROGRESS_FRAGMENT_ERROR:
            raise

    repaired = _fill_criterion_template(
        router,
        criterion_fragment_messages(
            requirement,
            criterion,
            selected,
            evidence,
            repair_no_progress=True,
        ),
        schema,
        batch=False,
    )
    return validate_criterion_fragment(
        repaired,
        selected_sections=selected,
        allowed_refs=allowed_refs,
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


class MissingWorksheetSections(ValueError):
    """Host-selected sections cannot be declared inapplicable by omission."""

    def __init__(self, sections: Iterable[str]) -> None:
        self.sections = tuple(sections)
        super().__init__(
            "DETAILED_PLAN_MISSING_SECTIONS: no authored contract for "
            + ", ".join(self.sections)
        )


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

    covered = {
        update["section"]
        for fragment in validated_fragments.values()
        for update in fragment[_FRAGMENT_KEY]
        if _text(update.get("implementation")) or _text(update.get("constraint"))
    }
    missing = [section for section in selected if section not in covered]
    if missing:
        raise MissingWorksheetSections(missing)

    for section in selected:
        specification: dict[str, list[dict[str, str]]] = {
            concern: [] for concern in DETAIL_RECORDS[section]
        }
        section_refs: list[str] = []
        for index, criterion in enumerate(criteria):
            update = updates_by_criterion[index].get(section)
            if update is None:
                continue
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
