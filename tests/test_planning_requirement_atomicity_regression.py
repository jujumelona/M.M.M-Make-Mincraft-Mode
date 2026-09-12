from __future__ import annotations

from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema
from minecraft_mod_ai.planning_contract_ssot import SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA
from minecraft_mod_ai.planning_state_resolution import _normalize_requirement_rows


def test_submit_researched_requirements_schema_is_atomic() -> None:
    assert_atomic_model_schema(
        SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA,
        surface="planner native tool submit_researched_requirements",
    )


def test_atomic_acceptance_string_preserves_internal_acceptance_list() -> None:
    rows = _normalize_requirement_rows(
        {
            "requirements": [
                {
                    "statement": "Build and upgrade a modular spacecraft.",
                    "semantic_capability": "spacecraft progression",
                    "acceptance": "Player can install or upgrade a spacecraft component.",
                }
            ]
        },
        {},
        "space mod",
    )

    assert rows == [
        {
            "statement": "Build and upgrade a modular spacecraft.",
            "semantic_capability": "spacecraft progression",
            "acceptance": ["Player can install or upgrade a spacecraft component."],
        }
    ]
