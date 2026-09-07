from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_detail_contract import validate_detailed_plan_grounding
from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS, validate_worksheet
from minecraft_mod_ai.planning_state_implementation import (
    _compile_requirement_plan,
    _implementation_evidence,
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
            {"capability": "Maintain an explicit server-owned economy state.", "constraint_evidence_refs": []}
        ],
        "implementation_obligations": [
            {"obligation": "Server validates each transaction and emits the accepted balance.", "constraint_evidence_refs": []}
        ],
        "artifact_obligations": [],
        "grounded_bindings": [],
        "reuse_candidates": [],
        "verification_obligations": [
            {"check": "Given insufficient funds, when purchase is requested, then balance and inventory remain unchanged.", "constraint_evidence_refs": []}
        ],
    }


class _TextOnlyRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def generate_text(self, role, messages, **kwargs):
        assert kwargs["response_format"] == "text"
        assert kwargs["enable_tools"] is False
        rendered = str(messages[-1]["content"])
        marker = next(line for line in rendered.splitlines() if line.startswith("Section: "))
        section = marker.split(": ", 1)[1]
        self.calls.append((section, kwargs))
        return (
            f"For {section}, the server-owned design uses explicit state owners, guarded transitions, "
            f"bounded failure behavior, and observable postconditions unique to {section}."
        )

    def generate_tool_decision(self, *args, **kwargs):
        raise AssertionError("detailed planning must not ask the model for a structured tool payload")


def _grounded_state() -> dict[str, object]:
    return {
        "research_queue": [
            {"research_id": "research_001", "requirement_ref": "req_001", "status": "complete"}
        ],
        "evidence": [
            {
                "evidence_id": "legacy_container_must_not_escape",
                "research_ref": "research_001",
                "claims": ["A grounded implementation fact."],
                "sufficient": True,
                "source": "grounded_materialized_pages",
                "evidence_refs": ["source:1"],
            }
        ],
    }


def test_detailed_plan_is_host_assembled_from_plain_text_sections() -> None:
    router = _TextOnlyRouter()
    requirement = {
        "requirement_id": "req_001",
        "statement": "Provide a server-owned economy.",
        "acceptance": ["Given insufficient funds, a purchase is rejected without mutation."],
    }

    plan = _compile_requirement_plan(router, _grounded_state(), requirement, WORKSHEET_SECTIONS)

    assert len(router.calls) == len(WORKSHEET_SECTIONS)
    assert tuple(plan["engineering_worksheet"]) == WORKSHEET_SECTIONS
    assert plan["grounded_bindings"] == []
    assert plan["reuse_candidates"] == []
    assert plan["verification_obligations"][0]["check"] == requirement["acceptance"][0]
    assert all(row["constraint_evidence_refs"] == [] for row in plan["implementation_obligations"])


def test_legacy_evidence_container_id_is_removed_before_model_context() -> None:
    rows = _implementation_evidence(_grounded_state(), "req_001")
    assert rows
    assert "evidence_id" not in rows[0]
    assert rows[0]["evidence_refs"] == ["source:1"]


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
        {"kind": "api_symbol", "fact": "ExampleApi.call exists.", "evidence_refs": []}
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
            {"decision_id": "detail_001", "decision_type": "detailed_implementation_plan", **plan},
        ],
        "blockers": [],
        "plan_ready": False,
    }
    validate_state_links(state)


def test_preflight_rejects_any_unready_requirement_before_compilation() -> None:
    state = _grounded_state()
    requirements = [{"requirement_id": "req_001"}, {"requirement_id": "req_002"}]
    with pytest.raises(ValueError, match="req_002 has no sufficient grounded"):
        _preflight_detailed_planning(state, requirements)
