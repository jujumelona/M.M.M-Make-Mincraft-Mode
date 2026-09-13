from __future__ import annotations

from minecraft_mod_ai.planning_state_implementation import (
    _preflight_detailed_planning,
    _requirement_grounding,
)


def test_missing_optional_implementation_evidence_does_not_abort_detailed_planning() -> None:
    state = {
        "research_queue": [],
        "evidence": [],
    }
    requirement = {
        "requirement_id": "req_001",
        "statement": "Players can build a spacecraft from modular parts.",
    }

    evidence, allowed_refs = _requirement_grounding(state, "req_001")

    assert evidence == []
    assert allowed_refs == set()
    _preflight_detailed_planning(state, [requirement])


def test_multiple_requirements_without_evidence_all_pass_preflight() -> None:
    state = {
        "research_queue": [],
        "evidence": [],
    }
    requirements = [
        {"requirement_id": "req_001", "statement": "Gather resources."},
        {"requirement_id": "req_002", "statement": "Trade resources."},
        {"requirement_id": "req_003", "statement": "Upgrade spacecraft parts."},
    ]

    _preflight_detailed_planning(state, requirements)


def test_duplicate_requirement_ids_still_fail_preflight() -> None:
    state = {
        "research_queue": [],
        "evidence": [],
    }
    requirements = [
        {"requirement_id": "req_001", "statement": "First."},
        {"requirement_id": "req_001", "statement": "Duplicate."},
    ]

    try:
        _preflight_detailed_planning(state, requirements)
    except ValueError as exc:
        assert "DETAILED_PLAN_REQUIREMENTS" in str(exc)
    else:
        raise AssertionError("duplicate requirement IDs must remain a hard failure")
