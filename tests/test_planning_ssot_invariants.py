from copy import deepcopy

import pytest

from minecraft_mod_ai.planning_detail_template import DETAIL_FIELDS, validate_worksheet
from minecraft_mod_ai.planning_state_contract import _build_host_state, _hash_without, validate_planning_state
from minecraft_mod_ai.planning_state_resolution import compile_researched_requirements
from minecraft_mod_ai.evidence_first_planning import EvidencePlanError, build_request_catalog


def initial(reason="user_preference", route="user_only"):
    return _build_host_state("Build my favorite game", {
        "goal": {"statement": "Build my favorite game", "source_quote": "Build my favorite game"},
        "known": [], "references": [], "scope_status": "explicit",
        "unresolved": [{"question": "Which game?", "reason": reason, "blocks": ["requirement_selection"],
                        "information_needed": "The user's favorite game", "resolution_route": route, "source_kinds": []}],
    })


def rehash(state):
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def worksheet(prefix="Concrete requirement"):
    return {
        key: {
            "specification": (
                f"{prefix} {key} specification defines its own owner, condition, boundary, "
                "and observable implementation result."
            ),
            "evidence_refs": ["source:1"],
        }
        for key in DETAIL_FIELDS
    }


def test_user_only_never_bypasses_requirement_gate():
    state = compile_researched_requirements(None, "Build my favorite game", initial())
    assert state["plan_ready"] is False
    assert state["unresolved"][0]["status"] == "open"
    assert state["unresolved"][0]["resolution_route"] == "user_only"
    assert not state["research_queue"]
    assert any(item.get("stage") == "requirement_selection" for item in state["blockers"])


def test_pipeline_stops_on_blocked_requirement_selection_before_detail_plan():
    from minecraft_mod_ai.planning_state_pipeline import prepare_planning_state

    with pytest.raises(
        ValueError,
        match="PLANNING_REQUIREMENT_SELECTION_BLOCKED:.*user_only",
    ):
        prepare_planning_state(
            None,
            "Build my favorite game",
            existing_state=initial(),
        )


def test_user_only_never_bypasses_ready_gate():
    state = initial()
    state.update(plan_ready=True, coverage=[{"status": "covered"}])
    with pytest.raises(ValueError, match="PROMPT_STATE_READY"):
        validate_planning_state(rehash(state))


def test_contradiction_route_is_owned_by_host_not_model_payload():
    state = initial("contradiction", "default_policy")
    assert state["unresolved"][0]["resolution_route"] == "user_only"
    assert state["unresolved"][0]["source_kinds"] == []
    assert not state["research_queue"]


def test_restored_state_cannot_lose_information_need():
    state = initial("external_fact", "external_research")
    state["research_queue"][0]["information_needed"] = ""
    state["research_queue"][0]["queries"] = ["guess"]
    with pytest.raises(ValueError, match="PROMPT_STATE_RESEARCH"):
        validate_planning_state(rehash(state))


def test_restored_state_cannot_forge_resolved_status():
    state = initial()
    state["unresolved"][0]["status"] = "resolved"
    with pytest.raises(ValueError, match="PROMPT_STATE_RESOLVED"):
        validate_planning_state(rehash(state))


def test_research_source_route_is_owned_by_host():
    state = initial("minecraft_api", "minecraft_research")
    assert set(state["research_queue"][0]["source_kinds"]) == {"minecraft_docs", "minecraft_source"}
    for row in (state["research_queue"][0], state["unresolved"][0]):
        row["source_kinds"] = ["web_sources"]
    with pytest.raises(ValueError, match="PROMPT_STATE_ROUTE"):
        validate_planning_state(rehash(state))


def test_unknown_prompt_cannot_become_synthetic_resolved_semantic():
    with pytest.raises(EvidencePlanError, match="UNRESOLVED_SEMANTICS"):
        build_request_catalog("Keep the weather compass.", {})


def test_worksheet_requires_each_section_and_real_refs():
    sheet = worksheet()
    assert validate_worksheet(sheet, {"source:1"}) == sheet
    partial = deepcopy(sheet)
    partial.pop("state_model")
    with pytest.raises(ValueError, match="WORKSHEET"):
        validate_worksheet(partial, {"source:1"})
    sheet["algorithm"]["evidence_refs"] = ["model:guess"]
    with pytest.raises(ValueError, match="WORKSHEET"):
        validate_worksheet(sheet, {"source:1"})


def test_grounded_detail_survives_handoff_and_resume_without_model_calls():
    from minecraft_mod_ai.planning_state_implementation import compile_detailed_implementation_plans
    from minecraft_mod_ai.planning_state_handoff import build_request_catalog_from_planning_state
    from minecraft_mod_ai.planning_state_pipeline import prepare_planning_state

    prompt = "Build a block"
    state = _build_host_state(prompt, {
        "goal": {"statement": prompt, "source_quote": prompt}, "known": [],
        "references": [], "scope_status": "explicit", "unresolved": [],
    })

    class Router:
        def __init__(self, answer):
            self.answer = answer
        def generate_tool_decision(self, *args, **kwargs):
            return self.answer

    state = compile_researched_requirements(Router({"requirements": [{
        "statement": prompt, "prompt_refs": ["goal"], "evidence_refs": [], "acceptance": ["Block can be placed"]
    }]}), prompt, state)
    research = state["research_queue"][0]
    research.update(status="complete", queries=["block registration source"])
    state["unresolved"][0]["status"] = "resolved"
    state["evidence"].append({"evidence_id": "e_001", "research_ref": research["research_id"],
        "claims": [{"finding": "Block registration source", "evidence_refs": ["source:1"]}],
        "evidence_refs": ["source:1"], "sufficient": True, "source": "grounded_materialized_pages"})
    state["resolved"].append({"unresolved_id": state["unresolved"][0]["unresolved_id"],
        "resolution": "Block registration", "basis": "grounded_research", "evidence_refs": ["source:1"]})
    detailed_worksheet = worksheet("Grounded block implementation")
    answer = {"engineering_worksheet": detailed_worksheet,
        "implementation_capabilities": [{"capability": "block registration", "evidence_refs": ["source:1"]}],
        "implementation_obligations": [{"obligation": "Register the placeable block", "evidence_refs": ["source:1"]}],
        "artifact_obligations": [], "reuse_candidates": [{"evidence_ref": "source:1", "mode": "reference_only", "reason": "Pattern only"}],
        "verification_obligations": [{"check": "Place block", "evidence_refs": ["source:1"]}]}
    state = compile_detailed_implementation_plans(Router(answer), prompt, rehash(state))
    validate_planning_state(state)
    catalog = build_request_catalog_from_planning_state(prompt, state)
    assert catalog["requirements"][0]["engineering_worksheet"] == detailed_worksheet
    assert catalog["requirements"][0]["reuse_candidates"][0]["mode"] == "reference_only"
    assert prepare_planning_state(None, prompt, existing_state=state) == state
    state["coverage"][0]["detailed_plan_ref"] = "nonexistent"
    with pytest.raises(ValueError, match="PROMPT_STATE_COVERAGE"):
        validate_planning_state(rehash(state))
