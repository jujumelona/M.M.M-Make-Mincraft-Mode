from __future__ import annotations

"""Host-owned applicability contract for selective detailed-planning sections.

Detailed planning is intentionally sparse: the baseline core is always planned, while
optional sections are scheduled only when host state explicitly marks them ``required``.
``unknown`` means that no obligation has established the section yet; it is not a reason
to manufacture work. Prompt wording and model-authored prose never gain omission or
inclusion authority here.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .planning_detail_template import (
    CONDITIONAL_WORKSHEET_SECTIONS,
    CORE_WORKSHEET_SECTIONS,
    normalize_required_sections,
)

APPLICABILITY_FIELD = "detail_section_applicability"
APPLICABILITY_STATUSES = ("required", "not_applicable", "unknown")


def default_detail_section_applicability() -> dict[str, str]:
    """Return undecided optional branches without scheduling speculative work."""

    return {section: "unknown" for section in CONDITIONAL_WORKSHEET_SECTIONS}


def normalize_detail_section_applicability(value: Any) -> dict[str, str]:
    """Validate a host applicability mapping while treating omissions as unknown."""

    if value is None:
        return default_detail_section_applicability()
    if not isinstance(value, Mapping):
        raise ValueError(
            "DETAILED_PLAN_APPLICABILITY: host applicability must be a section mapping"
        )

    unknown_sections = set(str(key) for key in value) - set(
        CONDITIONAL_WORKSHEET_SECTIONS
    )
    if unknown_sections:
        raise ValueError(
            "DETAILED_PLAN_APPLICABILITY: unknown conditional section(s): "
            + ", ".join(sorted(unknown_sections))
        )

    normalized = default_detail_section_applicability()
    for section in CONDITIONAL_WORKSHEET_SECTIONS:
        if section not in value:
            continue
        status = str(value[section] or "").strip()
        if status not in APPLICABILITY_STATUSES:
            raise ValueError(
                f"DETAILED_PLAN_APPLICABILITY: {section} has invalid status {status!r}"
            )
        normalized[section] = status
    return normalized


def required_detail_sections_for_requirement(
    requirement: Mapping[str, Any],
) -> tuple[str, ...]:
    """Project one requirement into only its established planning obligations."""

    applicability = normalize_detail_section_applicability(
        requirement.get(APPLICABILITY_FIELD)
    )
    selected = tuple(CORE_WORKSHEET_SECTIONS) + tuple(
        section
        for section in CONDITIONAL_WORKSHEET_SECTIONS
        if applicability[section] == "required"
    )
    return normalize_required_sections(selected)


def required_sections_by_requirement(
    state: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Project requirement-owned applicability into the planner selection API."""

    decisions = state.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("DETAILED_PLAN_APPLICABILITY: planning decisions must be an array")

    result: dict[str, tuple[str, ...]] = {}
    for item in decisions:
        if not isinstance(item, Mapping) or item.get("decision_type") != "requirement":
            continue
        requirement_id = str(item.get("requirement_id") or "")
        if not requirement_id:
            raise ValueError(
                "DETAILED_PLAN_APPLICABILITY: requirement ID must not be empty"
            )
        if requirement_id in result:
            raise ValueError(
                "DETAILED_PLAN_APPLICABILITY: duplicate requirement ID "
                + requirement_id
            )
        result[requirement_id] = required_detail_sections_for_requirement(item)
    return result


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def ensure_host_detail_section_applicability(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize requirement applicability without inventing optional obligations."""

    from .planning_state_contract import validate_planning_state

    validate_planning_state(state)
    value = deepcopy(dict(state))
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("DETAILED_PLAN_APPLICABILITY: planning decisions must be an array")

    for item in decisions:
        if not isinstance(item, dict) or item.get("decision_type") != "requirement":
            continue
        item[APPLICABILITY_FIELD] = normalize_detail_section_applicability(
            item.get(APPLICABILITY_FIELD)
        )

    result = _rehash(value)
    validate_planning_state(result)
    return result


def apply_host_detail_section_applicability(
    state: Mapping[str, Any],
    applicability_by_requirement: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Apply explicit host decisions to requirement SSOT and re-hash the planning state."""

    from .planning_state_contract import validate_planning_state

    validate_planning_state(state)
    if not isinstance(applicability_by_requirement, Mapping):
        raise ValueError(
            "DETAILED_PLAN_APPLICABILITY: host applicability must be a requirement mapping"
        )

    value = ensure_host_detail_section_applicability(state)
    requirements = {
        str(item.get("requirement_id") or ""): item
        for item in value.get("decisions", [])
        if isinstance(item, dict) and item.get("decision_type") == "requirement"
    }
    unknown = set(str(key) for key in applicability_by_requirement) - set(requirements)
    if unknown:
        raise ValueError(
            "DETAILED_PLAN_APPLICABILITY: applicability cites unknown requirement(s): "
            + ", ".join(sorted(unknown))
        )

    for requirement_id, raw in applicability_by_requirement.items():
        requirements[str(requirement_id)][APPLICABILITY_FIELD] = (
            normalize_detail_section_applicability(raw)
        )

    result = _rehash(value)
    validate_planning_state(result)
    return result


__all__ = [
    "APPLICABILITY_FIELD",
    "APPLICABILITY_STATUSES",
    "apply_host_detail_section_applicability",
    "default_detail_section_applicability",
    "ensure_host_detail_section_applicability",
    "normalize_detail_section_applicability",
    "required_detail_sections_for_requirement",
    "required_sections_by_requirement",
]
