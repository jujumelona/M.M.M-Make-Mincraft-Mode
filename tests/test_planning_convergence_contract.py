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
    # Prompt-boundary models are not allowed to invent external_fact/research unknowns.
    # Named references are authored input; the host deterministically creates one
    # reference_semantics research obligation for each reference.
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
    _rehash(state)
    validate_planning_state(state, prompt=PROMPT)

    def forbidden(*args, **kwargs):
        raise AssertionError("terminal blocked research must not be executed again")

    monkeypatch.setattr(convergence, "collect_planning_state_research", forbidden)
    result = convergence.collect_planning_state_research_convergent(None, PROMPT, state)

    assert result["research_queue"][0]["status"] == "blocked"
    assert convergence.planning_progress_fingerprint(result) == convergence.planning_progress_fingerprint(state)


def test_blocked_rows_are_masked_while_only_pending_delta_is_processed(monkeypatch):
    state = _base_state(unknown_count=2)
    state["research_queue"][0]["status"] = "blocked"
    _rehash(state)

    calls = 0

    def fake_collect(router, prompt, protected, *, trace_metadata=None):
        nonlocal calls
        calls += 1
        assert protected["research_queue"][0]["status"] == "complete"
        assert protected["research_queue"][1]["status"] == "pending"
        value = deepcopy(protected)
        value["research_queue"][1]["status"] = "blocked"
        return _rehash(value)

    monkeypatch.setattr(convergence, "collect_planning_state_research", fake_collect)
    result = convergence.collect_planning_state_research_convergent(None, PROMPT, state)

    assert calls == 1
    assert [row["status"] for row in result["research_queue"]] == ["blocked", "blocked"]


def test_research_cannot_create_out_of_universe_obligation():
    before = _base_state()
    after = deepcopy(before)
    after["unresolved"].append(
        {
            "unresolved_id": "u_999",
            "question": "invented",
            "reason": "external_fact",
            "blocks": ["requirement_selection"],
            "information_needed": "invented",
            "resolution_route": "external_research",
            "source_kinds": list(ROUTE_SOURCES["external_research"]),
            "status": "open",
            "research_ref": "r_999",
        }
    )
    after["research_queue"].append(
        {
            "research_id": "r_999",
            "resolves": ["u_999"],
            "objective": "invented",
            "information_needed": "invented",
            "source_kinds": list(ROUTE_SOURCES["external_research"]),
            "queries": [],
            "status": "pending",
        }
    )

    with pytest.raises(convergence.PlanningConvergenceError, match="PLANNING_OUT_OF_UNIVERSE_OBLIGATION"):
        convergence.assert_research_transition_monotone(before, after)


def test_terminal_research_status_cannot_move_back_to_pending():
    before = _base_state()
    before["research_queue"][0]["status"] = "blocked"
    after = deepcopy(before)
    after["research_queue"][0]["status"] = "pending"

    with pytest.raises(convergence.PlanningConvergenceError, match="PLANNING_NON_MONOTONE_TRANSITION"):
        convergence.assert_research_transition_monotone(before, after)


def test_requirement_boundary_freezes_exactly_one_implementation_obligation_per_requirement(monkeypatch):
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

    assert [row["requirement_id"] for row in result["decisions"] if row.get("decision_type") == "requirement"] == ["req_001"]
    assert result["unresolved"][-1]["requirement_ref"] == "req_001"
    assert result["research_queue"][-1]["requirement_ref"] == "req_001"