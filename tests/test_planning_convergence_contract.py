from __future__ import annotations

from copy import deepcopy

import pytest

import minecraft_mod_ai.planning_convergence_contract as convergence
from minecraft_mod_ai.planning_state_contract import (
    ROUTE_SOURCES,
    _build_host_state,
    _hash_without,
    validate_planning_state,
)


PROMPT = "Build the requested behavior using externally verified reference semantics."


def _rehash(state: dict) -> dict:
    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _base_state(*, unknown_count: int = 1) -> dict:
    references = [
        {
            "name": f"Reference {index}",
            "what_must_be_learned": f"Verify documented semantics for Reference {index}.",
        }
        for index in range(1, unknown_count + 1)
    ]
    return _build_host_state(
        PROMPT,
        {
            "goal": {"statement": "Build the requested behavior."},
            "known": [{"statement": "The requested behavior must be implemented."}],
            "references": references,
            "scope_status": "explicit",
            "unresolved": [],
        },
    )


def test_terminal_blocked_frontier_never_reopens_or_calls_research(monkeypatch):
    state = _base_state()
    state["research_queue"][0]["status"] = "blocked"
    state["unresolved"][0]["status"] = "blocked"
    _rehash(state)
    validate_planning_state(state, prompt=PROMPT)

    def forbidden(*args, **kwargs):
        raise AssertionError("terminal blocked research must not be executed again")

    monkeypatch.setattr(convergence, "collect_planning_state_research", forbidden)
    result = convergence.collect_planning_state_research_convergent(None, PROMPT, state)

    assert result["research_queue"][0]["status"] == "blocked"
    assert result["unresolved"][0]["status"] == "blocked"
    assert convergence.planning_progress_fingerprint(result) == convergence.planning_progress_fingerprint(state)


def test_blocked_rows_stay_terminal_while_only_pending_delta_is_processed(monkeypatch):
    state = _base_state(unknown_count=2)
    state["research_queue"][0]["status"] = "blocked"
    state["unresolved"][0]["status"] = "blocked"
    _rehash(state)

    calls = 0

    def fake_collect(router, prompt, current, *, trace_metadata=None):
        nonlocal calls
        calls += 1
        assert current["research_queue"][0]["status"] == "blocked"
        assert current["research_queue"][1]["status"] == "pending"
        value = deepcopy(current)
        value["research_queue"][1]["status"] = "blocked"
        return _rehash(value)

    monkeypatch.setattr(convergence, "collect_planning_state_research", fake_collect)
    result = convergence.collect_planning_state_research_convergent(None, PROMPT, state)

    assert calls == 1
    assert [row["status"] for row in result["research_queue"]] == ["blocked", "blocked"]
    assert [row["status"] for row in result["unresolved"]] == ["blocked", "blocked"]


def test_semantic_fixed_point_terminalizes_pending_frontier_and_resume_makes_zero_calls(monkeypatch):
    state = _base_state()
    calls = 0

    def no_progress(router, prompt, current, *, trace_metadata=None):
        nonlocal calls
        calls += 1
        return deepcopy(current)

    monkeypatch.setattr(convergence, "collect_planning_state_research", no_progress)
    terminal = convergence.collect_planning_state_research_convergent(None, PROMPT, state)

    assert calls == 1
    assert terminal["research_queue"][0]["status"] == "blocked"
    assert terminal["unresolved"][0]["status"] == "blocked"

    resumed = convergence.collect_planning_state_research_convergent(None, PROMPT, terminal)
    assert calls == 1
    assert convergence.planning_progress_fingerprint(resumed) == convergence.planning_progress_fingerprint(terminal)


def test_evidence_wording_churn_does_not_count_as_progress():
    left = _base_state()
    right = deepcopy(left)
    left["evidence"] = [
        {
            "research_ref": "r_001",
            "claims": [{"claim": "First wording."}],
            "evidence_refs": ["source:1"],
            "sufficient": True,
            "source": "grounded_materialized_pages",
        }
    ]
    right["evidence"] = [
        {
            "research_ref": "r_001",
            "claims": [{"claim": "Same evidence, different wording."}],
            "evidence_refs": ["source:1"],
            "sufficient": True,
            "source": "grounded_materialized_pages",
        }
    ]

    assert convergence.planning_progress_fingerprint(left) == convergence.planning_progress_fingerprint(right)


def test_research_cannot_create_out_of_universe_obligation():
    before = _base_state()
    after = deepcopy(before)
    invented_unknown = deepcopy(before["unresolved"][0])
    invented_unknown["unresolved_id"] = "u_999"
    invented_unknown["research_ref"] = "r_999"
    invented_research = deepcopy(before["research_queue"][0])
    invented_research["research_id"] = "r_999"
    invented_research["resolves"] = ["u_999"]
    after["unresolved"].append(invented_unknown)
    after["research_queue"].append(invented_research)

    with pytest.raises(convergence.PlanningConvergenceError, match="PLANNING_OUT_OF_UNIVERSE_OBLIGATION"):
        convergence.assert_research_transition_monotone(before, after)


def test_terminal_research_status_cannot_move_back_to_pending():
    before = _base_state()
    before["research_queue"][0]["status"] = "blocked"
    after = deepcopy(before)
    after["research_queue"][0]["status"] = "pending"

    with pytest.raises(convergence.PlanningConvergenceError, match="PLANNING_NON_MONOTONE_TRANSITION"):
        convergence.assert_research_transition_monotone(before, after)


def test_requirement_boundary_freezes_exactly_one_blocking_implementation_obligation_per_requirement(monkeypatch):
    state = _base_state()
    state["unresolved"][0]["status"] = "resolved"
    state["research_queue"][0]["status"] = "complete"
    state["evidence"].append(
        {
            "research_ref": "r_001",
            "claims": [{"claim": "verified"}],
            "evidence_refs": ["source:1"],
            "sufficient": True,
            "source": "grounded_materialized_pages",
        }
    )
    state["resolved"].append(
        {
            "unresolved_id": "u_001",
            "resolution": "verified",
            "basis": "grounded_research",
            "evidence_refs": ["source:1"],
        }
    )
    _rehash(state)
    validate_planning_state(state, prompt=PROMPT)

    def fake_compile(router, prompt, original):
        value = deepcopy(original)
        value["decisions"].append(
            {
                "decision_id": "d_001",
                "decision_type": "requirement",
                "requirement_id": "req_001",
                "statement": "Implement the behavior.",
                "semantic_capability": "custom",
                "acceptance": ["Behavior is observable."],
                "status": "implementation_research_pending",
            }
        )
        value["unresolved"].append(
            {
                "unresolved_id": "u_002",
                "question": "How is req_001 implemented?",
                "reason": "implementation_method",
                "blocks": [],
                "information_needed": "Concrete implementation evidence.",
                "resolution_route": "implementation_research",
                "source_kinds": list(ROUTE_SOURCES["implementation_research"]),
                "status": "open",
                "research_ref": "r_002",
                "requirement_ref": "req_001",
            }
        )
        value["research_queue"].append(
            {
                "research_id": "r_002",
                "resolves": ["u_002"],
                "requirement_ref": "req_001",
                "objective": "Find implementation evidence.",
                "information_needed": "Concrete implementation evidence.",
                "source_kinds": list(ROUTE_SOURCES["implementation_research"]),
                "queries": [],
                "status": "pending",
            }
        )
        return _rehash(value)

    monkeypatch.setattr(convergence, "compile_researched_requirements", fake_compile)
    result = convergence.compile_researched_requirements_convergent(None, PROMPT, state)

    requirements = [
        row["requirement_id"]
        for row in result["decisions"]
        if row.get("decision_type") == "requirement"
    ]
    assert requirements == ["req_001"]
    assert result["unresolved"][-1]["requirement_ref"] == "req_001"
    assert result["unresolved"][-1]["blocks"] == ["implementation_plan"]
    assert result["research_queue"][-1]["requirement_ref"] == "req_001"
