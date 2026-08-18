from __future__ import annotations

from minecraft_mod_ai.tool_transition_registry import reviewed_transition


def test_deterministic_content_is_a_reviewed_low_cost_generation_transition() -> None:
    transition = reviewed_transition("apply_minecraft_content_spec")

    assert transition is not None
    assert transition.preconditions == frozenset({"project_observed"})
    assert transition.effects == frozenset(
        {"project_changed", "source_generated", "generated"}
    )
    assert transition.cost == 1


def test_partial_source_edit_is_reviewed_and_cheaper_than_full_file_patch() -> None:
    partial = reviewed_transition("apply_source_edit")
    full = reviewed_transition("apply_source_patch")

    assert partial is not None
    assert full is not None
    assert partial.preconditions == frozenset({"project_observed", "evidence_ready"})
    assert partial.effects == frozenset({"project_changed", "repaired"})
    assert partial.cost < full.cost
