from minecraft_mod_ai.planning_state_implementation import _requirement_grounding


def test_requirement_grounding_allows_missing_research_evidence() -> None:
    state = {
        "research_queue": [],
        "evidence": [],
    }

    evidence, allowed_refs = _requirement_grounding(state, "req_001")

    assert evidence == []
    assert allowed_refs == set()
