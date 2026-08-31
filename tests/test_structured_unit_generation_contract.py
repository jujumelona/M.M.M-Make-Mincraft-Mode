from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.spec import SpecValidationError
from minecraft_mod_ai.structured_output import (
    StructuredOutputValidationError,
    validate_structured_output,
)


def _systems_schema() -> dict:
    section_id, fields, properties = design._SECTION_SPECS[1]
    assert section_id == "systems_and_progression"
    return {
        "type": "object",
        "properties": {
            "section": {
                "type": "object",
                "properties": dict(properties),
                "required": list(fields),
                "additionalProperties": False,
            }
        },
        "required": ["section"],
        "additionalProperties": False,
    }


def test_game_design_section_transport_leaves_recoverable_shape_to_owner() -> None:
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
        response_schema=_systems_schema(),
    )

    assert json.loads(validated)["section"]["combat"] == [
        "alien_combat",
        "colony_defense",
    ]


def test_section_owner_rejects_semantically_malformed_combat_shape() -> None:
    class Router:
        def __init__(self) -> None:
            self.calls = 0

        def generate_text(self, *_args, **_kwargs) -> str:
            self.calls += 1
            return json.dumps(
                {
                    "section": {
                        "progression": ["gather", "launch"],
                        "combat": ["alien_combat", "colony_defense"],
                        "mod_context": {},
                    }
                }
            )

    router = Router()
    _section_id, fields, properties = design._SECTION_SPECS[1]
    with pytest.raises(SpecValidationError, match="combat must be an object"):
        design._generate_section(
            router,
            prompt="우주 모드를 설계해줘",
            section_id="systems_and_progression",
            fields=fields,
            properties=properties,
            research={},
            media_paths=(),
            trace_metadata=None,
        )

    assert router.calls >= 1


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
