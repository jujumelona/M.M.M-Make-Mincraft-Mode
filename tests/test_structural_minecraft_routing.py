from pathlib import Path

import pytest

from minecraft_mod_ai import minecraft_template_catalog as catalog
from minecraft_mod_ai import minecraft_template_steps as steps

ENTITY_RESPONSIBILITIES = (
    "requirement", "registry", "type", "constructor", "dimensions", "attributes",
    "state", "ai", "targeting", "movement", "interaction", "combat", "damage",
    "death", "persistence", "synchronization", "spawn", "renderer", "model",
    "texture", "animation", "sound", "loot", "datagen", "integration", "validation",
)


def test_routing_is_invariant_to_feature_names_and_descriptions():
    structural = {
        "explicit_artifacts": ["entity"], "persistent_state": True,
        "networking": True, "loot": True, "model": True, "texture": True,
    }
    first = {**structural, "feature_id": "boss.entity", "feature_description": "boss dungeon economy skill trade", "domain": "combat"}
    second = {**structural, "feature_id": "completely-renamed", "feature_description": "unrelated words", "domain": "other"}
    assert catalog.build_artifact_plan(first) == catalog.build_artifact_plan(second)


def test_structural_artifact_detection_and_dependencies_are_deterministic():
    plan = catalog.build_artifact_plan({
        "explicit_artifacts": ["screen", "mob"], "persistent_state": True,
        "networking": True, "structure": True,
    })
    assert plan.detected == ("mob", "screen", "network_payload", "saved_data", "structure")
    assert plan.expanded == (
        "entity", "mob", "inventory", "menu", "screen", "network_payload",
        "saved_data", "worldgen", "structure",
    )
    assert catalog.expand_artifact_dependencies(reversed(plan.detected)) == plan.expanded


def test_unknown_artifacts_and_untyped_flags_are_rejected():
    with pytest.raises(ValueError, match="unknown canonical artifact"):
        catalog.build_artifact_plan({"explicit_artifacts": ["boss"]})
    with pytest.raises(TypeError, match="networking must be boolean"):
        catalog.build_artifact_plan({"networking": "yes"})


def test_entity_manifest_is_exactly_responsibility_grained():
    identifiers = steps.responsibility_ids_for_artifact("entity")
    assert identifiers == tuple(f"minecraft/entity/{name}" for name in ENTITY_RESPONSIBILITIES)
    compiled = steps.steps_for_artifact("entity")
    assert tuple(step.template_id for step in compiled) == identifiers


def test_old_profile_and_broad_step_apis_are_gone():
    assert not hasattr(catalog, "MinecraftTemplateProfile")
    assert not hasattr(catalog, "profile_for_capability")
    assert not hasattr(steps, "steps_for_profile")
    assert not hasattr(steps, "merge_minecraft_template_steps")


def test_catalog_source_has_no_gameplay_archetype_routing():
    source = Path(catalog.__file__).read_text(encoding="utf-8")
    for token in ("boss_combat", "dungeon", "skill.ability", "_EXACT_TEMPLATE", "_CATEGORY_TEMPLATE"):
        assert token not in source
