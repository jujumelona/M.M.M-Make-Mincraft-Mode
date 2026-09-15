from __future__ import annotations

"""Contract for lowering canonical detailed plans into the request-catalog boundary.

The planning state owns detailed-plan semantics. The request catalog may expose flattened
legacy views for downstream planners, but those views must be derived from and remain
losslessly tied to one canonical detailed-plan snapshot. Keeping that projection and its
validation here prevents producer/consumer field drift at the planning handoff.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .planning_detail_contract import validate_detailed_plan_grounding
from .planning_detail_template import normalize_required_sections, validate_worksheet

_DETAIL_COLLECTIONS = (
    "implementation_capabilities",
    "implementation_obligations",
    "artifact_obligations",
    "grounded_bindings",
    "reuse_candidates",
    "verification_obligations",
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _rows(detail: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = detail.get(key)
    if not isinstance(value, list):
        raise ValueError(f"PLANNING_HANDOFF_DETAIL: {key} must be an array")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"PLANNING_HANDOFF_DETAIL: {key} must contain objects")
    return [deepcopy(dict(item)) for item in value]


def _canonical_detail(detail: Mapping[str, Any]) -> dict[str, Any]:
    raw_sections = detail.get("required_detail_sections")
    if raw_sections is not None and not isinstance(raw_sections, list):
        raise ValueError(
            "PLANNING_HANDOFF_DETAIL: required_detail_sections must be an array"
        )
    sections = normalize_required_sections(raw_sections)
    if raw_sections is not None and list(sections) != raw_sections:
        raise ValueError(
            "PLANNING_HANDOFF_DETAIL: required_detail_sections must use canonical order"
        )
    worksheet = detail.get("engineering_worksheet")
    if not isinstance(worksheet, Mapping):
        raise ValueError(
            "PLANNING_HANDOFF_DETAIL: engineering_worksheet must be an object"
        )

    result: dict[str, Any] = {
        "requirement_ref": _text(detail.get("requirement_ref")),
        "required_detail_sections": list(sections),
        "engineering_worksheet": deepcopy(dict(worksheet)),
    }
    if not result["requirement_ref"]:
        raise ValueError("PLANNING_HANDOFF_DETAIL: requirement_ref is required")
    for key in _DETAIL_COLLECTIONS:
        result[key] = _rows(detail, key)
    return result


def _allowed_refs_from_detail(detail: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in (
        "implementation_capabilities",
        "implementation_obligations",
        "artifact_obligations",
        "verification_obligations",
    ):
        for row in detail.get(key, []):
            if isinstance(row, Mapping) and isinstance(
                row.get("constraint_evidence_refs"), list
            ):
                refs.update(
                    _text(ref)
                    for ref in row["constraint_evidence_refs"]
                    if _text(ref)
                )
    for row in detail.get("grounded_bindings", []):
        if isinstance(row, Mapping) and isinstance(row.get("evidence_refs"), list):
            refs.update(_text(ref) for ref in row["evidence_refs"] if _text(ref))
    for row in detail.get("reuse_candidates", []):
        if isinstance(row, Mapping) and _text(row.get("evidence_ref")):
            refs.add(_text(row.get("evidence_ref")))
    worksheet = detail.get("engineering_worksheet")
    if isinstance(worksheet, Mapping):
        for row in worksheet.values():
            if isinstance(row, Mapping) and isinstance(
                row.get("constraint_evidence_refs"), list
            ):
                refs.update(
                    _text(ref)
                    for ref in row["constraint_evidence_refs"]
                    if _text(ref)
                )
    return refs


def _flatten(detail: Mapping[str, Any], key: str, value_key: str) -> list[str]:
    values: list[str] = []
    for row in detail.get(key, []):
        if not isinstance(row, Mapping):
            continue
        value = _text(row.get(value_key))
        if value and value not in values:
            values.append(value)
    return values


def _legacy_artifacts(detail: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for row in detail.get("artifact_obligations", []):
        if not isinstance(row, Mapping):
            continue
        artifacts.append(
            {
                "kind": _text(row.get("kind")),
                "purpose": _text(row.get("purpose")),
                "status": "REQUIRED_DESIGN_AND_GENERATION",
                # Downstream artifact contracts still call these evidence_refs. They are
                # derived here from the canonical authored constraint field; no consumer
                # may read a nonexistent legacy field from the detailed plan itself.
                "evidence_refs": list(row.get("constraint_evidence_refs") or []),
            }
        )
    return artifacts


def project_detailed_plan_for_request_catalog(
    detail: Mapping[str, Any], sufficient_refs: set[str]
) -> dict[str, Any]:
    """Validate one source detail and return its lossless request-catalog projection."""

    canonical = _canonical_detail(detail)
    validate_detailed_plan_grounding(canonical, sufficient_refs)
    validate_worksheet(
        canonical["engineering_worksheet"],
        sufficient_refs,
        canonical["required_detail_sections"],
    )
    capabilities = _flatten(
        canonical, "implementation_capabilities", "capability"
    )
    obligations = _flatten(
        canonical, "implementation_obligations", "obligation"
    )
    if not capabilities or not obligations:
        raise ValueError(
            "PLANNING_HANDOFF_DETAIL: concrete capabilities and obligations are required"
        )
    verification = _flatten(canonical, "verification_obligations", "check")
    return {
        "planning_detail_contract": canonical,
        "required_detail_sections": deepcopy(canonical["required_detail_sections"]),
        "engineering_worksheet": deepcopy(canonical["engineering_worksheet"]),
        "implementation_capabilities": capabilities,
        "implementation_obligations": obligations,
        "artifact_obligations": _legacy_artifacts(canonical),
        "grounded_bindings": deepcopy(canonical["grounded_bindings"]),
        "reuse_candidates": deepcopy(canonical["reuse_candidates"]),
        "verification_obligations": deepcopy(canonical["verification_obligations"]),
        "verification_checks": verification,
    }


def validate_request_requirement_detail(requirement: Mapping[str, Any]) -> None:
    """Reject request-catalog detail loss or drift before downstream planning starts."""

    canonical = requirement.get("planning_detail_contract")
    if not isinstance(canonical, Mapping):
        raise ValueError(
            "PLANNING_HANDOFF_DETAIL: canonical planning_detail_contract is required"
        )
    canonical = _canonical_detail(canonical)
    allowed = _allowed_refs_from_detail(canonical)
    validate_detailed_plan_grounding(canonical, allowed)
    validate_worksheet(
        canonical["engineering_worksheet"],
        allowed,
        canonical["required_detail_sections"],
    )

    expected = {
        "required_detail_sections": canonical["required_detail_sections"],
        "engineering_worksheet": canonical["engineering_worksheet"],
        "implementation_capabilities": _flatten(
            canonical, "implementation_capabilities", "capability"
        ),
        "implementation_obligations": _flatten(
            canonical, "implementation_obligations", "obligation"
        ),
        "artifact_obligations": _legacy_artifacts(canonical),
        "grounded_bindings": canonical["grounded_bindings"],
        "reuse_candidates": canonical["reuse_candidates"],
        "verification_obligations": canonical["verification_obligations"],
        "verification_checks": _flatten(
            canonical, "verification_obligations", "check"
        ),
    }
    for key, value in expected.items():
        if requirement.get(key) != value:
            raise ValueError(
                f"PLANNING_HANDOFF_DETAIL: request-catalog {key} drifted from canonical detail"
            )


__all__ = [
    "project_detailed_plan_for_request_catalog",
    "validate_request_requirement_detail",
]
