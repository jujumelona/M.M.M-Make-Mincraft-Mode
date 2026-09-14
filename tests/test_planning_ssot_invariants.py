from __future__ import annotations

from worksheet_fixtures import specification

from copy import deepcopy

import pytest

from minecraft_mod_ai.planning_authority import build_authoritative_request_catalog
from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS, validate_worksheet
from minecraft_mod_ai.planning_state_contract import (
    _hash_without,
    _build_host_state,
    validate_planning_state,
)
from minecraft_mod_ai.planning_state_pipeline import prepare_planning_state
from minecraft_mod_ai.planning_state_resolution import compile_researched_requirements


def _host_state(router, prompt):
    return _build_host_state(prompt, router.responses.pop(0))


class _Router:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_tool_decision(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


def _payload(*, unresolved=None):
    prompt = "Keep the weather compass."
    return {
        "goal": {"statement": prompt, "source_quote": prompt},
        "known": [{"statement": prompt, "source_quote": prompt}],
        "references": [],
        "scope_status": "explicit",
        "unresolved": list(unresolved or []),
    }


def _user_only_state():
    return _host_state(
        _Router(
            [
                _payload(
                    unresolved=[
                        {
                            "question": "Which player preference is required?",
                            "reason": "user_preference",
                            "information_needed": "The authored player preference.",
                        }
                    ]
                )
            ]
        ),
        "Keep the weather compass.",
    )


def _worksheet():
    return {
        key: {
            "specification": (
                specification(key)
            ),
            "constraint_evidence_refs": ["source:1"],
        }
        for key in WORKSHEET_SECTIONS
    }


def _rehash(state):
    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def test_user_only_unknown_remains_a_requirement_blocker_without_fake_research():
    prompt = "Keep the weather compass."
    state = compile_researched_requirements(None, prompt, _user_only_state())

    assert state["plan_ready"] is False
    assert state["unresolved"][0]["status"] == "open"
    assert state["unresolved"][0]["resolution_route"] == "user_only"
    assert state["research_queue"] == []
    assert any(
        item.get("stage") == "requirement_selection"
        for item in state["blockers"]
    )
    validate_planning_state(state, prompt=prompt)


def test_pipeline_preserves_original_user_only_unknown_as_resumable_state():
    state = prepare_planning_state(
        None,
        "Keep the weather compass.",
        existing_state=_user_only_state(),
    )

    assert state["plan_ready"] is False
    assert state["unresolved"][0]["status"] == "open"
    assert state["unresolved"][0]["resolution_route"] == "user_only"
    assert state["research_queue"] == []
    validate_planning_state(state, prompt="Keep the weather compass.")


def test_requirement_selection_does_not_create_implementation_research_obligations():
    prompt = "Keep the weather compass."
    state = _host_state(_Router([_payload()]), prompt)
    router = _Router(
        [
            {
                "requirements": [
                    {
                        "statement": "The player can keep and use the weather compass.",
                        "semantic_capability": "custom.semantic",
                        "prompt_refs": ["goal"],
                        "evidence_refs": [],
                        "acceptance": ["The compass remains available to the player."],
                    }
                ]
            }
        ]
    )

    resolved = compile_researched_requirements(router, prompt, state)

    assert any(
        item.get("decision_type") == "requirement"
        for item in resolved["decisions"]
    )
    assert all(
        item.get("reason") != "implementation_method"
        and item.get("resolution_route") != "implementation_research"
        for item in resolved["unresolved"]
    )
    assert all(
        item.get("objective") != "Find compass implementation evidence"
        for item in resolved["research_queue"]
    )

def test_state_integrity_rejects_unhashed_restored_state_mutation():
    state = _user_only_state()
    state["unresolved"][0]["status"] = "resolved"

    with pytest.raises(ValueError, match="PROMPT_STATE_HASH"):
        validate_planning_state(state, prompt="Keep the weather compass.")


def test_external_fact_is_host_owned_and_cannot_be_authored_by_prompt_model():
    raw = _payload(
        unresolved=[
            {
                "question": "Which external weather rule is authoritative?",
                "reason": "external_fact",
                "information_needed": "The authoritative external weather rule.",
                "resolution_route": "user_only",
                "source_kinds": [],
                "blocks": ["implementation_plan"],
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="PROMPT_STATE_UNRESOLVED: model cannot author reason 'external_fact'",
    ):
        _host_state(
            _Router([raw]),
            "Keep the weather compass.",
        )


def test_raw_prompt_cannot_bypass_grounded_planning_state_authority():
    with pytest.raises(ValueError, match="PLANNING_AUTHORITY_STATE_REQUIRED"):
        build_authoritative_request_catalog("Keep the weather compass.")


def test_worksheet_requires_all_selected_sections_and_grounded_refs():
    sheet = _worksheet()
    assert validate_worksheet(sheet, {"source:1"}) == sheet

    partial = deepcopy(sheet)
    partial.pop("state_model")
    with pytest.raises(ValueError, match="WORKSHEET"):
        validate_worksheet(partial, {"source:1"})

    invalid_ref = deepcopy(sheet)
    invalid_ref["algorithm"]["constraint_evidence_refs"] = ["model:guess"]
    with pytest.raises(ValueError, match="WORKSHEET"):
        validate_worksheet(invalid_ref, {"source:1"})
