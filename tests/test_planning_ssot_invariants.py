from __future__ import annotations

from copy import deepcopy

import pytest

from minecraft_mod_ai.evidence_first_planning import EvidencePlanError, build_request_catalog
from minecraft_mod_ai.planning_state_contract import (
    planning_state_hash,
    validate_planning_state,
)
from minecraft_mod_ai.planning_state_implementation import validate_worksheet


def rehash(state):
    state["state_sha256"] = planning_state_hash(state)
    return state


def initial(target="minecraft_api", route="minecraft_research"):
    state = {
        "schema_version": "mmm/prompt-first-planning-state-v1",
        "original_prompt": "Keep the weather compass.",
        "references": [],
        "decisions": [],
        "unresolved": [
            {
                "question_id": "q1",
                "question": "How is the weather compass implemented?",
                "kind": "implementation",
                "target": target,
                "status": "open",
                "resolved_by": [],
                "source_kinds": [],
                "route": "",
                "research_query": "",
            }
        ],
        "research_queue": [
            {
                "question_id": "q1",
                "query": "weather compass implementation",
                "kind": "implementation",
                "target": target,
                "source_kinds": [],
                "route": "",
                "status": "pending",
            }
        ],
        "research_evidence": [],
        "detailed_plans": [],
        "plan_ready": False,
        "state_sha256": "",
    }
    # These fields are compiler-owned.  The helper only mirrors the canonical route
    # expected by the validator so mutation tests can focus on one invariant at a time.
    state["unresolved"][0]["source_kinds"] = (
        ["minecraft_docs", "minecraft_source"]
        if route == "minecraft_research"
        else ["web_sources"]
    )
    state["unresolved"][0]["route"] = route
    state["research_queue"][0]["source_kinds"] = list(
        state["unresolved"][0]["source_kinds"]
    )
    state["research_queue"][0]["route"] = route
    return rehash(state)


def worksheet():
    return {
        "algorithm": {"summary": "algorithm", "evidence_refs": ["source:1"]},
        "state_model": {"summary": "state", "evidence_refs": ["source:1"]},
        "data_model": {"summary": "data", "evidence_refs": ["source:1"]},
        "integration_points": {"summary": "integration", "evidence_refs": ["source:1"]},
        "failure_modes": {"summary": "failure", "evidence_refs": ["source:1"]},
        "verification": {"summary": "verify", "evidence_refs": ["source:1"]},
    }


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


def test_raw_prompt_cannot_bypass_planning_state_authority():
    with pytest.raises(EvidencePlanError, match="PLANNING_STATE_AUTHORITY_REQUIRED"):
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
