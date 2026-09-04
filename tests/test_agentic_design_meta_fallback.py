from __future__ import annotations

from minecraft_mod_ai import agentic_research_game_design as design


class _DirtySystemsRouter:
    def generate_text(self, role, messages, **kwargs):
        del role, messages, kwargs
        return """## progression
- I need to decide progression before writing the answer.
## combat
### encounters
- I should inspect branch-policy before answering.
## mod_context
### persistence
- Save and restore progression state.
"""


def test_dirty_nested_fields_fall_back_without_discarding_clean_neighbors(monkeypatch):
    monkeypatch.setenv("MMM_PLANNER_TRACE", "0")
    monkeypatch.setenv("MMM_PLANNER_TRACE_CONSOLE", "0")

    prompt = "Add persistent seasonal progression."
    section = design._generate_section(
        _DirtySystemsRouter(),
        prompt=prompt,
        section_id="systems_and_progression",
        fields=("progression", "combat", "mod_context"),
        research={},
        media_paths=(),
        trace_metadata=None,
    )

    assert section["progression"] == [prompt]
    assert section["combat"] == {}
    assert section["mod_context"] == {
        "persistence": ["Save and restore progression state."]
    }
