from __future__ import annotations

import pytest

from minecraft_mod_ai.structural_artifact_mapping import (
    CANONICAL_ARTIFACT_KINDS,
    detect_structural_artifacts,
    expand_artifact_dependencies,
    validate_artifact_kinds,
)


def _requirement(name: str) -> dict:
    return {
        "requirement_id": name,
        "capability": name,
        "statement": name,
        "artifact_obligations": [
            {"kind": "block_entity"},
            {"kind": "network_payload"},
            {"kind": "screen"},
            {"kind": "loot_table"},
        ],
    }


def test_feature_name_cannot_change_artifact_selection() -> None:
    first = detect_structural_artifacts(_requirement("alpha"))
    second = detect_structural_artifacts(_requirement("boss_economy_trading_skill_dungeon"))
    assert first == second
    assert first.artifact_kinds == (
        "block", "block_entity", "network_payload", "screen", "datagen", "loot",
    )


def test_capability_and_statement_are_not_routing_inputs() -> None:
    base = _requirement("plain")
    renamed = dict(base)
    renamed["capability"] = "worldgen.dungeon"
    renamed["statement"] = "boss trade economy skill dungeon"
    assert detect_structural_artifacts(base) == detect_structural_artifacts(renamed)


def test_technical_aliases_expand_to_canonical_artifacts() -> None:
    plan = detect_structural_artifacts(
        {"artifact_obligations": [{"kind": "entity_model"}, {"kind": "lang"}]}
    )
    assert plan.artifact_kinds == ("entity", "model", "datagen", "language")
    assert set(plan.artifact_kinds) <= set(CANONICAL_ARTIFACT_KINDS)


def test_structural_dependencies_are_deterministic() -> None:
    assert expand_artifact_dependencies(("dimension", "mob", "recipe")) == (
        "worldgen", "dimension", "entity", "mob", "datagen", "recipe",
    )


def test_unknown_artifact_is_not_guessed() -> None:
    plan = detect_structural_artifacts(
        {"artifact_obligations": [{"kind": "unclassified_surface"}]}
    )
    assert plan.artifact_kinds == ()
    assert plan.unresolved_inputs == ("unclassified_surface",)


def test_validator_rejects_noncanonical_artifacts() -> None:
    with pytest.raises(ValueError, match="unsupported canonical artifact"):
        validate_artifact_kinds(("boss",))
