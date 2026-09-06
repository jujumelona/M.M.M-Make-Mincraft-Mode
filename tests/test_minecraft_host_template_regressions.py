from __future__ import annotations

from minecraft_mod_ai.minecraft_template_catalog import (
    profile_for_capability,
    selected_predecessor_capabilities,
)
from minecraft_mod_ai.minecraft_template_steps import steps_for_profile


def _step_names(capability: str) -> tuple[str, ...]:
    profile = profile_for_capability(capability)
    return tuple(step.name for step in steps_for_profile(profile))


def test_special_mineral_is_worldgen_resource_not_alien_combat() -> None:
    profile = profile_for_capability("planet.special_mineral")
    assert profile.template_id == "worldgen_resource"

    steps = steps_for_profile(profile)
    names = {step.name for step in steps}
    assert {
        "semantic_contract",
        "resource_registry",
        "configured_feature",
        "placed_feature",
        "biome_dimension_binding",
        "mining_loot_acquisition",
        "failure_contract",
        "runtime_scenario",
    } <= names
    assert "entity_attributes_spawn" not in names
    assert "ai_damage_death" not in names
    assert all("aggro" not in step.outcome.casefold() for step in steps)
    assert all("combat" not in step.outcome.casefold() for step in steps)


def test_space_travel_has_dedicated_multistage_template_not_generic_fallback() -> None:
    profile = profile_for_capability("space.travel")
    assert profile.template_id == "space_travel"

    names = set(_step_names("space.travel"))
    assert {
        "semantic_contract",
        "launch_unlock_policy",
        "fuel_destination_transaction",
        "world_transition",
        "failure_contract",
        "runtime_scenario",
    } <= names
    assert "semantic_implementation" not in names
    assert len(names) >= 6


def test_unknown_capability_still_compiles_to_multistep_host_template() -> None:
    capability = "custom.semantic_0123456789abcdef"
    profile = profile_for_capability(capability)
    assert profile.template_id == "custom_gameplay"

    names = set(_step_names(capability))
    assert {
        "semantic_contract",
        "authoritative_behavior",
        "integration_binding",
        "failure_contract",
        "runtime_scenario",
    } <= names
    assert len(names) >= 5
    assert "semantic_implementation" not in names


def test_component_crafting_alias_feeds_upgrade_and_launch_progression() -> None:
    selected = (
        "economy.trade",
        "spaceship.component_crafting",
        "spacecraft.performance_upgrade",
        "spacecraft.expansion",
        "space.launch",
    )

    assert selected_predecessor_capabilities(
        "spacecraft.performance_upgrade",
        selected,
    ) == ("spaceship.component_crafting", "economy.trade")
    assert selected_predecessor_capabilities(
        "spacecraft.expansion",
        selected,
    ) == ("spaceship.component_crafting", "economy.trade")
    assert selected_predecessor_capabilities("space.launch", selected) == (
        "spaceship.component_crafting",
        "spacecraft.performance_upgrade",
        "spacecraft.expansion",
    )
