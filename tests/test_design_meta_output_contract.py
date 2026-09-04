from __future__ import annotations

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.game_design import _validate_design, _validate_ready_design
from minecraft_mod_ai.model_meta_output_contract import contains_internal_model_meta
from minecraft_mod_ai.spec import SpecValidationError


def _valid_design() -> dict[str, object]:
    return {
        "title": "Space Colony",
        "pitch": "Explore, build, and preserve progression.",
        "core_loop": ["Explore", "Gather", "Build"],
        "progression": ["Unlock upgrades through authored milestones."],
        "combat": {"encounters": ["Resolve bounded hostile encounters."]},
        "mod_context": {"persistence": ["Persist authored progression state."]},
        "modules": [
            {
                "plugin_id": "progression_system",
                "status": "custom",
                "reason": "Implements authored progression.",
                "requirement_refs": [],
                "implementation_obligations": ["Store and advance progression state."],
            }
        ],
        "assets": [
            {
                "id": "progress_badge",
                "kind": "item",
                "brief": "A readable progression badge.",
            }
        ],
        "acceptance_tests": ["Progression remains observable after reload."],
        "art_direction": {"summary": ["Use readable Minecraft-native silhouettes."]},
    }


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


@pytest.mark.parametrize(
    ("field", "dirty_value"),
    [
        ("title", "<think>private reasoning</think> Space Colony"),
        ("pitch", "I need to decide the pitch before answering."),
        ("core_loop", ["The user wants me to decide the loop first."]),
        ("progression", ["I need to decide progression before writing the answer."]),
        ("combat", {"encounters": ["I should reason about combat before answering."]}),
        ("mod_context", {"persistence": ["The user asked me to inspect persistence first."]}),
        (
            "modules",
            [
                {
                    "plugin_id": "meta_module",
                    "status": "custom",
                    "reason": "I need to decide what the user wants.",
                    "requirement_refs": [],
                    "implementation_obligations": ["Implement the approved behavior."],
                }
            ],
        ),
        (
            "assets",
            [
                {
                    "id": "meta_asset",
                    "kind": "item",
                    "brief": "<think>choose an asset</think>",
                }
            ],
        ),
        ("acceptance_tests", ["I should verify the branch-policy first."]),
        ("art_direction", {"summary": ["The user wants a palette, so I need to think."]}),
    ],
)
def test_final_design_validation_rejects_meta_in_every_guarded_field(
    field: str,
    dirty_value: object,
):
    payload = _valid_design()
    payload[field] = dirty_value
    with pytest.raises(SpecValidationError, match=rf"game_design\.{field}"):
        _validate_design(payload)


def test_ready_design_revalidates_nested_semantics_before_planning():
    payload = _valid_design()
    payload["combat"] = {"encounters": ["I need to decide combat first."]}
    with pytest.raises(SpecValidationError, match=r"game_design\.combat"):
        _validate_ready_design("add a persistent space colony loop", payload)


def test_normal_design_text_is_allowed():
    assert not contains_internal_model_meta("Explore planets and establish colonies.")
    assert design._parse_field_output("Galactic Settlers", "title") == "Galactic Settlers"
