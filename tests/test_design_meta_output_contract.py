from __future__ import annotations

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.game_design import _validate_design
from minecraft_mod_ai.model_meta_output_contract import contains_internal_model_meta
from minecraft_mod_ai.spec import SpecValidationError


@pytest.mark.parametrize(
    ("field", "body"),
    [
        ("title", "<think>check internals</think> Space Colony"),
        ("pitch", "I need to review the branch-policy before designing this."),
        ("core_loop", "- The user wants the player to mine ore"),
    ],
)
def test_section_parser_rejects_internal_meta(field: str, body: str):
    with pytest.raises(SpecValidationError):
        design._parse_field_output(body, field)


def test_final_design_validation_rejects_internal_meta():
    payload = {
        "title": "<think>private reasoning</think> Space Colony",
        "pitch": "Explore and build.",
        "core_loop": ["Explore", "Build"],
        "progression": [],
        "combat": {},
        "mod_context": {},
        "modules": [],
        "assets": [],
        "acceptance_tests": [],
        "art_direction": {},
    }
    with pytest.raises(SpecValidationError):
        _validate_design(payload)


def test_normal_design_text_is_allowed():
    assert not contains_internal_model_meta("Explore planets and establish colonies.")
    assert design._parse_field_output("Galactic Settlers", "title") == "Galactic Settlers"
