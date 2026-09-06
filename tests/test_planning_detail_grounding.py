from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_detail_contract import validate_detailed_plan_grounding
from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS, validate_worksheet
from minecraft_mod_ai.planning_state_implementation import (
    _PARAMETERS,
    _preflight_detailed_planning,
    _validate_refs,
)
from minecraft_mod_ai.planning_state_invariants import validate_state_links


def _authored_worksheet() -> dict[str, dict[str, object]]:
    return {
        key: {
            "specification": (
                f"{key} has a distinct authored implementation contract with an owner, "
                "condition, boundary, and observable result."
            ),
            "constraint_evidence_refs": [],
        }
        for key in WORKSHEET_SECTIONS
    }


def _authored_plan() -> dict[str, object]:
    return {
        "requirement_ref": "req_001",
        "required_detail_sections": list(WORKSHEET_SECTIONS),
        "engineering_worksheet": _authored_worksheet(),
        "implementation_capabilities": [
            {
                "capability": "Maintain an explicit server-owned economy state.",
                "constraint_evidence_refs": [],
            }
        ],
        "implementation_obligations": [
            {
                "obligation": "Server validates each transaction and emits the accepted balance.",
                "constraint_evidence_refs": [],
            }
        ],
        "artifact_obligations": [],
        "grounded_bindings": [],
        "reuse_candidates": [],
        "verification_obligations": [
            {
                "check": "Given insufficient funds, when purchase is requested, then balance and inventory remain unchanged.",
                "constraint_evidence_refs": [],
            }
        ],
    }


def test_authored_design_sections_do_not_require_fake_evidence() -> None:
    sheet = _authored_worksheet()

    assert validate_worksheet(sheet, {"source:1"}) == sheet


def test_constraint_evidence_must_still_be_host_allowed_when_present() -> None:
    sheet = _authored_worksheet()
    sheet["algorithm"]["constraint_evidence_refs"] = ["model:guess"]

    with pytest.raises(ValueError, match="invalid constraint evidence"):
        validate_worksheet(sheet, {"source:1"})


def test_grounded_external_fact_requires_real_allowed_evidence() -> None:
    assert _validate_refs(["source:1"], {"source:1"}, field="binding") == ["source:1"]

    with pytest.raises(ValueError, match="grounded evidence is required"):
        _validate_refs([], {"source:1"}, field="binding")
    with pytest.raises(ValueError, match="unknown evidence refs"):
        _validate_refs(["model:guess"], {"source:1"}, field="binding")


def test_shared_grounding_contract_allows_unconstrained_authored_design() -> None:
    validate_detailed_plan_grounding(_authored_plan(), set())


def test_shared_grounding_contract_rejects_unproven_external_fact() -> None:
    plan = _authored_plan()
    plan["grounded_bindings"] = [
        {
            "kind": "api_symbol",
            "fact": "ExampleApi.call exists.",
            "evidence_refs": [],
        }
    ]

    with pytest.raises(ValueError, match="grounded evidence is required"):
        validate_detailed_plan_grounding(plan, {"source:1"})


def test_canonical_state_accepts_authored_rows_without_fake_evidence() -> None:
    plan = _authored_plan()
    state = {
        "goal": {"statement": "Create an economy feature."},
        "references": [],
        "known": [],
        "unresolved": [],
        "research_queue": [],
        "evidence": [],
        "resolved": [],
        "decisions": [
            {
                "decision_id": "requirement_001",
                "decision_type": "requirement",
                "requirement_id": "req_001",
                "prompt_refs": ["goal"],
                "evidence_refs": [],
            },
            {
                "decision_id": "detail_001",
                "decision_type": "detailed_implementation_plan",
                **plan,
            },
        ],
        "blockers": [],
        "plan_ready": False,
    }

    validate_state_links(state)


def test_preflight_rejects_any_unready_requirement_before_compilation() -> None:
    state = {
        "research_queue": [
            {
                "research_id": "research_001",
                "requirement_ref": "req_001",
                "status": "complete",
            }
        ],
        "evidence": [
            {
                "research_ref": "research_001",
                "sufficient": True,
                "evidence_refs": ["source:1"],
            }
        ],
    }
    requirements = [
        {"requirement_id": "req_001"},
        {"requirement_id": "req_002"},
    ]

    with pytest.raises(ValueError, match="req_002 has no sufficient grounded"):
        _preflight_detailed_planning(state, requirements)


def test_tool_schema_keeps_design_constraints_separate_from_grounded_bindings() -> None:
    properties = _PARAMETERS["properties"]
    capability_refs = properties["implementation_capabilities"]["items"]["properties"][
        "constraint_evidence_refs"
    ]
    binding_refs = properties["grounded_bindings"]["items"]["properties"]["evidence_refs"]

    assert "minItems" not in capability_refs
    assert binding_refs["minItems"] == 1
    assert "grounded_bindings" in _PARAMETERS["required"]
