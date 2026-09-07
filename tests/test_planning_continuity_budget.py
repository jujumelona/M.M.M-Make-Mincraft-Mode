from __future__ import annotations

from minecraft_mod_ai import planning_state_implementation as implementation


def test_small_continuity_context_is_unchanged() -> None:
    specifications = {
        "behavior_contract": "Player enters the state and receives the expected observable effect.",
        "state_model": "State is explicit and transitions are deterministic.",
    }

    assert implementation._continuity_context(specifications) == (
        "- behavior_contract: Player enters the state and receives the expected observable effect.\n"
        "- state_model: State is explicit and transitions are deterministic."
    )


def test_large_continuity_context_is_bounded_without_dropping_sections() -> None:
    sections = (
        "behavior_contract",
        "state_model",
        "algorithm",
        "integration",
        "failure_and_limits",
        "reuse_assessment",
        "verification",
    )
    specifications = {
        section: f"{section}-HEAD " + ("x" * 8_000) + f" {section}-TAIL"
        for section in sections
    }

    context = implementation._continuity_context(specifications)

    assert len(context) <= implementation._CONTINUITY_CONTEXT_MAX_CHARS
    for section in sections:
        assert f"- {section}: " in context
        assert f"{section}-HEAD" in context
        assert f"{section}-TAIL" in context
    assert implementation._CONTINUITY_ELLIPSIS in context
