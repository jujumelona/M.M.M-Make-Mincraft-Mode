from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.spec import SpecValidationError
from minecraft_mod_ai.structured_output import (
    StructuredOutputValidationError,
    validate_structured_output,
)


def _valid_design() -> dict[str, object]:
    return {
        "title": "Space Colony",
        "pitch": "Build the authored space-colony behavior without expanding scope.",
        "core_loop": ["gather", "build", "launch"],
        "progression": ["gather", "launch"],
        "combat": {"authored_combat": ["defend the colony"]},
        "mod_context": {"authored_scope": ["space colony"]},
        "modules": [
            {
                "plugin_id": "design_req_space",
                "status": "custom_required",
                "capability": "space.colony",
                "reason": "Implement the authored colony behavior.",
                "requirement_refs": ["req_space"],
                "implementation_obligations": [
                    "Implement the authored colony behavior.",
                    "The player can launch after completing the colony requirements.",
                ],
            }
        ],
        "assets": [],
        "acceptance_tests": [
            "The player can launch after completing the colony requirements."
        ],
    }


def test_unrelated_structured_transport_still_validates_json_shape() -> None:
    schema = {
        "type": "object",
        "properties": {
            "section": {
                "type": "object",
                "properties": {
                    "progression": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "combat": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "mod_context": {"type": "object"},
                },
                "required": ["progression", "combat", "mod_context"],
                "additionalProperties": False,
            }
        },
        "required": ["section"],
        "additionalProperties": False,
    }
    raw = json.dumps(
        {
            "section": {
                "progression": ["gather", "launch"],
                "combat": ["alien_combat", "colony_defense"],
                "mod_context": {},
            }
        }
    )

    validated = validate_structured_output(
        raw,
        response_format="json",
        response_schema=schema,
    )

    assert json.loads(validated)["section"]["combat"] == [
        "alien_combat",
        "colony_defense",
    ]


def test_host_design_canonicalization_drops_private_transport_state() -> None:
    payload = _valid_design()
    payload["_planner_private"] = {"legacy_parser": "must not escape"}

    canonical = design.canonical_game_design(payload)

    assert canonical == _valid_design()
    assert "_planner_private" not in canonical


def test_host_design_rejects_incomplete_canonical_shape() -> None:
    payload = _valid_design()
    payload["combat"] = []

    with pytest.raises(SpecValidationError, match="game_design.combat must be an object"):
        design.canonical_game_design(payload)


def test_canonical_module_shape_keeps_semantic_execution_obligations() -> None:
    canonical = design.canonical_game_design(_valid_design())
    module = canonical["modules"][0]

    assert module["requirement_refs"] == ["req_space"]
    assert module["implementation_obligations"] == [
        "Implement the authored colony behavior.",
        "The player can launch after completing the colony requirements.",
    ]


def test_unrelated_structured_schema_remains_strict() -> None:
    schema = {
        "type": "object",
        "properties": {"payload": {"type": "object"}},
        "required": ["payload"],
        "additionalProperties": False,
    }

    with pytest.raises(StructuredOutputValidationError):
        validate_structured_output(
            '{"payload":["not","an","object"]}',
            response_format="json",
            response_schema=schema,
        )
