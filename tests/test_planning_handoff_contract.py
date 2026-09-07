from __future__ import annotations

from worksheet_fixtures import specification

from copy import deepcopy

import pytest

from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS
from minecraft_mod_ai.planning_handoff_contract import (
    project_detailed_plan_for_request_catalog,
    validate_request_requirement_detail,
)


def _worksheet() -> dict[str, dict[str, object]]:
    return {
        key: {
            "specification": (
                specification(key)
            ),
            "constraint_evidence_refs": [],
        }
        for key in WORKSHEET_SECTIONS
    }


def _detail() -> dict[str, object]:
    return {
        "requirement_ref": "req_001",
        "required_detail_sections": list(WORKSHEET_SECTIONS),
        "engineering_worksheet": _worksheet(),
        "implementation_capabilities": [
            {
                "capability": "The server owns the authoritative economy state.",
                "constraint_evidence_refs": [],
            }
        ],
        "implementation_obligations": [
            {
                "obligation": (
                    "The server applies a validated transaction and exposes the resulting "
                    "balance to the requesting player."
                ),
                "constraint_evidence_refs": [],
            }
        ],
        "artifact_obligations": [
            {
                "kind": "source component",
                "purpose": "Own authoritative economy transaction state.",
                "constraint_evidence_refs": ["source:1"],
            }
        ],
        "grounded_bindings": [
            {
                "kind": "repository_fact",
                "fact": "The target repository exposes the researched integration surface.",
                "evidence_refs": ["source:1"],
            }
        ],
        "reuse_candidates": [
            {
                "evidence_ref": "source:1",
                "mode": "adapt",
                "reason": "The researched source establishes the integration shape but not the authored economy behavior.",
            }
        ],
        "verification_obligations": [
            {
                "check": "Given a valid transaction, when the server applies it, then the authoritative balance changes exactly once.",
                "constraint_evidence_refs": [],
            }
        ],
    }


def test_projection_preserves_canonical_grounding_and_verification() -> None:
    projection = project_detailed_plan_for_request_catalog(_detail(), {"source:1"})
    canonical = projection["planning_detail_contract"]

    assert canonical["artifact_obligations"][0]["constraint_evidence_refs"] == [
        "source:1"
    ]
    assert "evidence_refs" not in canonical["artifact_obligations"][0]
    assert projection["artifact_obligations"][0]["evidence_refs"] == ["source:1"]
    assert projection["grounded_bindings"] == canonical["grounded_bindings"]
    assert projection["verification_obligations"] == canonical[
        "verification_obligations"
    ]
    assert projection["required_detail_sections"] == canonical[
        "required_detail_sections"
    ]


def test_request_requirement_detail_rejects_projection_drift() -> None:
    requirement = project_detailed_plan_for_request_catalog(_detail(), {"source:1"})
    validate_request_requirement_detail(requirement)

    requirement["artifact_obligations"][0]["evidence_refs"] = []
    with pytest.raises(ValueError, match="artifact_obligations drifted"):
        validate_request_requirement_detail(requirement)


def test_request_requirement_detail_rejects_grounded_binding_without_evidence() -> None:
    requirement = project_detailed_plan_for_request_catalog(_detail(), {"source:1"})
    broken = deepcopy(requirement)
    broken["planning_detail_contract"]["grounded_bindings"][0]["evidence_refs"] = []

    with pytest.raises(ValueError, match="grounded evidence is required"):
        validate_request_requirement_detail(broken)


def test_request_requirement_detail_rejects_legacy_artifact_field_in_canonical_contract() -> None:
    requirement = project_detailed_plan_for_request_catalog(_detail(), {"source:1"})
    broken = deepcopy(requirement)
    artifact = broken["planning_detail_contract"]["artifact_obligations"][0]
    artifact.pop("constraint_evidence_refs")
    artifact["evidence_refs"] = ["source:1"]

    with pytest.raises(ValueError, match="evidence refs must be an array"):
        validate_request_requirement_detail(broken)
