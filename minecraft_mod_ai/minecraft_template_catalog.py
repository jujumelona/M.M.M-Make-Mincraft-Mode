from __future__ import annotations

"""Research-backed host catalog for Minecraft implementation planning.

This module is the feature-model layer between authored gameplay capabilities and the
implementation task compiler.  The language model does not choose implementation
architecture.  Exact known capabilities map to reusable Minecraft archetypes; unknown
capabilities receive a conservative host-owned gameplay archetype instead of a one-task
free-form fallback.

The template features mirror stable Minecraft/Fabric/NeoForge implementation concerns:
registry identity, data generation/resources, persistent state, logical-side networking,
client presentation, world generation, server authority, and automated runtime tests.
Loader/version-specific symbols remain the responsibility of the resolved target adapter.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .canonical_capability_ontology import (
    CapabilityDefinition,
    atomic_capability_definitions,
    canonical_domain_map,
)

TEMPLATE_CATALOG_SCHEMA = "mmm/minecraft-template-catalog-v1"
CUSTOM_CAPABILITY_SENTINEL = "custom.semantic"

FEATURE_REGISTRY = "needs_registry"
FEATURE_DATAGEN = "needs_datagen"
FEATURE_PERSISTENCE = "needs_persistence"
FEATURE_NETWORK = "needs_network"
FEATURE_CLIENT = "needs_client_render"
FEATURE_WORLDGEN = "needs_worldgen"
FEATURE_MIXIN = "needs_mixin"
FEATURE_SERVER_AUTHORITY = "server_authority"
FEATURE_RUNTIME_TEST = "runtime_test"

_BRANCH_FEATURES = frozenset(
    {
        FEATURE_REGISTRY,
        FEATURE_DATAGEN,
        FEATURE_PERSISTENCE,
        FEATURE_NETWORK,
        FEATURE_CLIENT,
        FEATURE_WORLDGEN,
        FEATURE_MIXIN,
    }
)

_DEFINITIONS: Mapping[str, CapabilityDefinition] = atomic_capability_definitions()
_DOMAIN_MAP = canonical_domain_map()


@dataclass(frozen=True)
class MinecraftTemplateProfile:
    capability: str
    canonical_capability: str | None
    known_capability: bool
    category: str
    template_id: str
    features: frozenset[str]
    implementation_capabilities: tuple[str, ...]
    artifact_kinds: tuple[str, ...]
    design_resolution_obligations: tuple[str, ...]

    @property
    def branch_features(self) -> frozenset[str]:
        return frozenset(self.features & _BRANCH_FEATURES)


# Exact domain mechanics whose implementation topology is more specific than their
# broad ontology category.  No substring matching is permitted here.
_EXACT_TEMPLATE: dict[str, str] = {
    "boss.entity": "boss_combat",
    "combat.boss": "boss_combat",
    "combat.damage": "combat_service",
    "combat.weapon": "combat_item",
    "alien.entity": "entity",
    "alien.combat": "hostile_entity_combat",
    "mob.spawning": "entity",
    "entity.lifecycle": "entity",
    "entity.vehicle": "vehicle",
    "spaceship.vehicle": "vehicle",
    "item.weapon": "combat_item",
    "crafting.recipe": "crafting",
    "inventory.transfer": "inventory_service",
    "network.action_sync": "network",
    "network.transaction": "network",
    "persistence.state_store": "persistence",
    "persistence.balance": "persistence",
    "ui.menu": "client_ui",
    "ui.container": "container_ui",
    "ui.shop": "container_ui",
    "automation.machine": "machine",
    "block_entity.tick": "machine",
    "energy.generator": "machine",
    "energy.production": "stateful_service",
    "energy.storage": "stateful_service",
    "economy.currency": "economy",
    "economy.reward": "economy",
    "economy.trade": "economy",
    "trade.offer_model": "economy",
    "trade.shop_registry": "economy",
    "quest.state": "quest",
    "quest.progression": "quest",
    "quest.reward": "quest",
    "worldgen.dimension": "dimension_worldgen",
    "worldgen.planet": "dimension_worldgen",
    "worldgen.structure": "structure_worldgen",
    "worldgen.dungeon": "structure_worldgen",
    "loot.drop_table": "data_resource",
    "resource.mining": "resource",
    "resource.farming": "resource",
    "resource.special_ore": "worldgen_resource",
    "planet.special_mineral": "worldgen_resource",
    "spacecraft.component_construction": "spacecraft_assembly",
    "spaceship.component_crafting": "crafting",
    "spacecraft.weapon_upgrade": "spacecraft_upgrade",
    "spacecraft.performance_upgrade": "spacecraft_upgrade",
    "spacecraft.expansion": "spacecraft_upgrade",
    "crew.npc": "entity",
    "crew.recruitment": "crew",
    "space.travel": "space_travel",
    "space.launch": "space_travel",
    "colony.settlement": "colony",
    "colony.colonization": "colony",
    "colony.progression": "progression",
    "skill.ability": "ability",
    "skill.magic": "ability",
}

_CATEGORY_TEMPLATE: dict[str, str] = {
    "combat": "combat_service",
    "entity": "entity",
    "item": "item",
    "crafting": "crafting",
    "progression": "progression",
    "quest": "quest",
    "economy": "economy",
    "technology": "machine",
    "transport": "vehicle",
    "worldgen": "structure_worldgen",
    "magic": "ability",
    "ui": "client_ui",
    "storage": "persistence",
    "network": "network",
    "resource": "resource",
}

_TEMPLATE_FEATURES: dict[str, frozenset[str]] = {
    "item": frozenset({FEATURE_REGISTRY, FEATURE_DATAGEN, FEATURE_CLIENT, FEATURE_RUNTIME_TEST}),
    "combat_item": frozenset(
        {FEATURE_REGISTRY, FEATURE_DATAGEN, FEATURE_CLIENT, FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}
    ),
    "combat_service": frozenset({FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}),
    "entity": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "hostile_entity_combat": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "boss_combat": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_PERSISTENCE,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "crafting": frozenset({FEATURE_DATAGEN, FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}),
    "data_resource": frozenset({FEATURE_DATAGEN, FEATURE_RUNTIME_TEST}),
    "inventory_service": frozenset({FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}),
    "network": frozenset({FEATURE_NETWORK, FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}),
    "persistence": frozenset({FEATURE_PERSISTENCE, FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}),
    "client_ui": frozenset({FEATURE_CLIENT, FEATURE_RUNTIME_TEST}),
    "container_ui": frozenset(
        {FEATURE_REGISTRY, FEATURE_NETWORK, FEATURE_CLIENT, FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}
    ),
    "economy": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_PERSISTENCE,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "machine": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_PERSISTENCE,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "stateful_service": frozenset(
        {FEATURE_PERSISTENCE, FEATURE_NETWORK, FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}
    ),
    "vehicle": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_PERSISTENCE,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "dimension_worldgen": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_WORLDGEN,
            FEATURE_PERSISTENCE,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "structure_worldgen": frozenset(
        {FEATURE_REGISTRY, FEATURE_DATAGEN, FEATURE_WORLDGEN, FEATURE_RUNTIME_TEST}
    ),
    "resource": frozenset(
        {FEATURE_REGISTRY, FEATURE_DATAGEN, FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}
    ),
    "worldgen_resource": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_WORLDGEN,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "progression": frozenset(
        {FEATURE_PERSISTENCE, FEATURE_NETWORK, FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}
    ),
    "quest": frozenset(
        {
            FEATURE_PERSISTENCE,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "ability": frozenset(
        {FEATURE_NETWORK, FEATURE_CLIENT, FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}
    ),
    "spacecraft_assembly": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_PERSISTENCE,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "spacecraft_upgrade": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_PERSISTENCE,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "crew": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_PERSISTENCE,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "space_travel": frozenset(
        {
            FEATURE_PERSISTENCE,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "colony": frozenset(
        {
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_PERSISTENCE,
            FEATURE_NETWORK,
            FEATURE_CLIENT,
            FEATURE_SERVER_AUTHORITY,
            FEATURE_RUNTIME_TEST,
        }
    ),
    "software_quality": frozenset({FEATURE_MIXIN, FEATURE_RUNTIME_TEST}),
    "custom_gameplay": frozenset({FEATURE_SERVER_AUTHORITY, FEATURE_RUNTIME_TEST}),
}

_TEMPLATE_IMPLEMENTATION_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "item": ("registry.lifecycle", "item.behavior", "resources.item", "gametest.runtime_scenario"),
    "combat_item": (
        "registry.lifecycle",
        "combat.damage_service",
        "combat.server_validation",
        "resources.item",
        "gametest.runtime_scenario",
    ),
    "combat_service": ("combat.damage_service", "combat.server_validation", "gametest.runtime_scenario"),
    "entity": (
        "registry.lifecycle",
        "entity.attributes",
        "entity.lifecycle",
        "network.server_authority",
        "resources.entity",
        "gametest.runtime_scenario",
    ),
    "hostile_entity_combat": (
        "registry.lifecycle",
        "entity.attributes",
        "entity.spawn_rules",
        "entity.ai_goals",
        "combat.damage_service",
        "combat.server_validation",
        "loot.drop_table",
        "gametest.runtime_scenario",
    ),
    "boss_combat": (
        "registry.lifecycle",
        "entity.attributes",
        "entity.ai_goals",
        "combat.damage_service",
        "combat.phase_state",
        "persistence.entity_state",
        "network.server_authority",
        "loot.drop_table",
        "gametest.runtime_scenario",
    ),
    "crafting": ("crafting.recipe_contract", "crafting.server_validation", "gametest.runtime_scenario"),
    "data_resource": ("resources.data_definition", "gametest.runtime_scenario"),
    "inventory_service": ("inventory.transaction", "inventory.server_validation", "gametest.runtime_scenario"),
    "network": ("network.payload_codec", "network.server_authority", "network.state_sync", "gametest.runtime_scenario"),
    "persistence": ("persistence.state_codec", "persistence.state_store", "gametest.runtime_scenario"),
    "client_ui": ("ui.client_contract", "ui.client_surface", "gametest.runtime_scenario"),
    "container_ui": (
        "ui.menu_contract",
        "network.server_authority",
        "inventory.transaction",
        "ui.client_surface",
        "gametest.runtime_scenario",
    ),
    "economy": (
        "economy.transaction_service",
        "economy.price_stock_policy",
        "persistence.economy_state",
        "network.server_authority",
        "ui.transaction_surface",
        "gametest.runtime_scenario",
    ),
    "machine": (
        "registry.lifecycle",
        "block_entity.state_owner",
        "block_entity.server_tick",
        "persistence.machine_state",
        "network.server_authority",
        "ui.machine_surface",
        "gametest.runtime_scenario",
    ),
    "stateful_service": (
        "gameplay.state_schema",
        "persistence.state_store",
        "network.server_authority",
        "gametest.runtime_scenario",
    ),
    "vehicle": (
        "registry.lifecycle",
        "entity.vehicle_state",
        "vehicle.server_control",
        "persistence.vehicle_state",
        "network.server_authority",
        "ui.vehicle_surface",
        "gametest.runtime_scenario",
    ),
    "dimension_worldgen": (
        "worldgen.dimension_registry",
        "worldgen.dimension_data",
        "worldgen.destination_access",
        "persistence.travel_state",
        "gametest.runtime_scenario",
    ),
    "structure_worldgen": (
        "worldgen.structure_registry",
        "worldgen.configured_feature",
        "worldgen.placed_feature",
        "worldgen.biome_binding",
        "gametest.runtime_scenario",
    ),
    "resource": (
        "item.registry",
        "loot.acquisition",
        "inventory.transaction",
        "resources.data_definition",
        "gametest.runtime_scenario",
    ),
    "worldgen_resource": (
        "item.registry",
        "worldgen.configured_feature",
        "worldgen.placed_feature",
        "worldgen.biome_binding",
        "loot.acquisition",
        "inventory.transaction",
        "gametest.runtime_scenario",
    ),
    "progression": (
        "progression.state_schema",
        "progression.transition_service",
        "persistence.progression_state",
        "network.server_authority",
        "gametest.runtime_scenario",
    ),
    "quest": (
        "quest.state_schema",
        "quest.objective_service",
        "persistence.quest_state",
        "network.server_authority",
        "ui.quest_surface",
        "gametest.runtime_scenario",
    ),
    "ability": (
        "ability.state_schema",
        "ability.server_execution",
        "network.server_authority",
        "ui.ability_surface",
        "gametest.runtime_scenario",
    ),
    "spacecraft_assembly": (
        "spacecraft.component_registry",
        "spacecraft.assembly_validation",
        "economy.transaction_service",
        "persistence.spacecraft_state",
        "network.server_authority",
        "ui.spacecraft_surface",
        "gametest.runtime_scenario",
    ),
    "spacecraft_upgrade": (
        "spacecraft.gameplay_stat_schema",
        "spacecraft.upgrade_tier_service",
        "economy.transaction_service",
        "persistence.spacecraft_state",
        "network.server_authority",
        "ui.spacecraft_surface",
        "gametest.runtime_scenario",
    ),
    "crew": (
        "entity.crew_lifecycle",
        "crew.role_skill_schema",
        "crew.recruit_replace_service",
        "persistence.crew_state",
        "network.server_authority",
        "ui.crew_surface",
        "gametest.runtime_scenario",
    ),
    "space_travel": (
        "space.launch_unlock_policy",
        "space.fuel_transaction",
        "space.destination_registry",
        "space.travel_transition",
        "persistence.travel_state",
        "network.server_authority",
        "ui.destination_surface",
        "gametest.runtime_scenario",
    ),
    "colony": (
        "colony.placement_validation",
        "colony.ownership_state",
        "colony.progression_service",
        "colony.storage_service",
        "persistence.colony_state",
        "network.server_authority",
        "ui.colony_surface",
        "gametest.runtime_scenario",
    ),
    "software_quality": (
        "software.baseline",
        "software.compatibility_patch",
        "software.behavior_equivalence",
        "software.performance_regression",
    ),
    "custom_gameplay": (
        "gameplay.semantic_contract",
        "gameplay.server_authority",
        "gameplay.integration",
        "gameplay.failure_contract",
        "gametest.runtime_scenario",
    ),
}

_TEMPLATE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "item": ("item_model", "lang", "tag", "gametest"),
    "combat_item": ("item_model", "recipe", "tag", "lang", "gametest"),
    "combat_service": ("gametest",),
    "entity": ("entity_model", "loot_table", "tag", "lang", "gametest"),
    "hostile_entity_combat": ("entity_model", "loot_table", "tag", "lang", "gametest"),
    "boss_combat": ("entity_model", "loot_table", "tag", "lang", "gametest"),
    "crafting": ("recipe", "tag", "gametest"),
    "data_resource": ("loot_table", "recipe", "tag", "lang", "gametest"),
    "inventory_service": ("gametest",),
    "network": ("gametest",),
    "persistence": ("gametest",),
    "client_ui": ("lang", "gametest"),
    "container_ui": ("lang", "gametest"),
    "economy": ("item_model", "recipe", "tag", "lang", "gametest"),
    "machine": ("blockstate", "block_model", "item_model", "loot_table", "recipe", "tag", "lang", "gametest"),
    "stateful_service": ("lang", "gametest"),
    "vehicle": ("entity_model", "item_model", "lang", "gametest"),
    "dimension_worldgen": ("dimension_data", "worldgen_data", "tag", "lang", "gametest"),
    "structure_worldgen": ("worldgen_data", "loot_table", "tag", "gametest"),
    "resource": ("item_model", "loot_table", "recipe", "tag", "lang", "gametest"),
    "worldgen_resource": ("item_model", "worldgen_data", "loot_table", "recipe", "tag", "lang", "gametest"),
    "progression": ("lang", "gametest"),
    "quest": ("lang", "gametest"),
    "ability": ("lang", "gametest"),
    "spacecraft_assembly": ("item_model", "recipe", "tag", "lang", "gametest"),
    "spacecraft_upgrade": ("item_model", "recipe", "tag", "lang", "gametest"),
    "crew": ("entity_model", "loot_table", "lang", "gametest"),
    "space_travel": ("dimension_data", "tag", "lang", "gametest"),
    "colony": ("blockstate", "block_model", "item_model", "recipe", "loot_table", "tag", "lang", "gametest"),
    "software_quality": ("benchmark", "gametest"),
    "custom_gameplay": ("lang", "gametest"),
}

_COMMON_DESIGN_OBLIGATIONS = (
    "Name every registry identifier and authoritative state owner used by this behavior.",
    "Define success, rejection, consumption and persistence-visible state transitions that the authored behavior actually requires.",
    "Define a disposable server-authoritative GameTest scenario with concrete setup, action and assertions.",
)

_TEMPLATE_DESIGN_OBLIGATIONS: dict[str, tuple[str, ...]] = {
    "resource": (
        "Name resource types, acquisition amounts/rules, inventory destinations, sinks and authored crafting or upgrade uses.",
    ),
    "worldgen_resource": (
        "Name the resource/block/item IDs, configured-feature key, placed-feature key, placement constraints, biome/dimension targets, drops and authored consumption uses.",
    ),
    "economy": (
        "Define currency representation, trader access, buy/sell prices, stock/restock rules and atomic insufficient-funds/full-inventory rejection.",
    ),
    "spacecraft_assembly": (
        "Name spacecraft part types and slots, compatibility rules, acquisition path and assembly-completion conditions.",
    ),
    "spacecraft_upgrade": (
        "Define only the authored spacecraft stat dimensions, tier effects, caps, costs, compatibility and downgrade/removal behavior.",
    ),
    "crew": (
        "Define authored crew roles/skills, hiring costs, assignments, removal/death behavior and replacement rules.",
    ),
    "space_travel": (
        "Declare required versus optional launch unlocks, fuel cost/consumption, allowed destinations, arrival placement and return/reload behavior.",
    ),
    "dimension_worldgen": (
        "Define dimension/planet keys, terrain or generator contract, biome/data bindings, access rule, safe arrival and reload behavior.",
    ),
    "hostile_entity_combat": (
        "Define entity attributes, spawn conditions, AI goals, attack/damage/death behavior and drop table without adding unrequested combat systems.",
    ),
    "boss_combat": (
        "Define boss attributes, phase/state transitions, attacks, damage rules, death outcome and drops only where authored.",
    ),
    "colony": (
        "Define colony placement validity, ownership, authored development stages/costs, storage boundaries and save/reload behavior.",
    ),
    "machine": (
        "Define block/block-entity IDs, owned state, tick trigger, input/output rules, processing state and reload-safe behavior.",
    ),
    "container_ui": (
        "Define authoritative menu state/slots and every client action that must be validated on the logical server.",
    ),
    "network": (
        "Define payload direction, fields, codec/version contract, logical-side handler, server validation and synchronization trigger.",
    ),
    "persistence": (
        "Define state scope/owner, codec schema, default creation, dirty/change lifecycle, reload assertions and compatibility behavior.",
    ),
    "custom_gameplay": (
        "Define only the user-authored custom semantic inputs, authoritative state changes, integration boundary and observable failure behavior; do not invent a Minecraft subsystem that the request does not require.",
    ),
}

# If both capabilities are explicitly selected, these edges are safe host-owned
# integration dependencies. They never cause a missing feature to be invented.
_SELECTED_PREDECESSORS: dict[str, tuple[str, ...]] = {
    "spacecraft.component_construction": (
        "resource.farming",
        "resource.mining",
        "economy.currency",
        "economy.trade",
        "crafting.recipe",
    ),
    "spacecraft.weapon_upgrade": (
        "spacecraft.component_construction",
        "economy.trade",
    ),
    "spacecraft.performance_upgrade": (
        "spacecraft.component_construction",
        "economy.trade",
    ),
    "spacecraft.expansion": (
        "spacecraft.component_construction",
        "economy.trade",
    ),
    "space.launch": (
        "spaceship.vehicle",
        "spacecraft.component_construction",
    ),
    "planet.special_mineral": ("space.launch",),
    "alien.combat": ("space.launch", "combat.weapon"),
    "colony.colonization": ("space.launch",),
}

RESEARCH_BASIS: tuple[dict[str, str], ...] = (
    {
        "topic": "networking",
        "source": "Fabric Networking / NeoForge Networking",
        "principle": "typed payload/codec registration, logical-side handling, server validation and state synchronization",
    },
    {
        "topic": "persistent_state",
        "source": "NeoForge SavedData/Data Attachments and Minecraft codec-based persistence patterns",
        "principle": "explicit state owner, codec/serializer, dirty/change lifecycle and reload verification",
    },
    {
        "topic": "data_generation",
        "source": "Fabric Data Generation and NeoForge resource/datagen documentation",
        "principle": "recipes, tags, loot, models, language and data-driven registries are deterministic generated/validated artifacts",
    },
    {
        "topic": "world_generation",
        "source": "Fabric Feature Generation and datapack/dynamic registry documentation",
        "principle": "configured feature, placed feature and target biome/dimension binding are separate verifiable stages",
    },
    {
        "topic": "testing",
        "source": "Fabric automated testing/GameTest guidance",
        "principle": "unit-test pure helpers and use GameTest/client GameTest for Minecraft runtime behavior",
    },
    {
        "topic": "architecture",
        "source": "software product line / feature-model engineering",
        "principle": "derive variants from reusable common assets plus explicit variability and dependency constraints",
    },
)


def known_capability_ids() -> tuple[str, ...]:
    return tuple(sorted(_DEFINITIONS))


def semantic_capability_choices() -> tuple[str, ...]:
    return (*known_capability_ids(), CUSTOM_CAPABILITY_SENTINEL)


def capability_catalog_for_model() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "id": capability_id,
            "description": definition.description,
        }
        for capability_id, definition in sorted(_DEFINITIONS.items())
    )


def _known_capability(value: Any) -> str | None:
    text = str(value or "").strip().casefold().removeprefix("capability:")
    if text in _DEFINITIONS:
        return text
    mapped = _DOMAIN_MAP.get(text)
    if mapped:
        candidate = str(mapped[0]).casefold()
        if candidate in _DEFINITIONS:
            return candidate
    return None


def is_known_capability(value: Any) -> bool:
    return _known_capability(value) is not None


def _template_id_for_known(capability: str) -> str:
    if capability in _EXACT_TEMPLATE:
        return _EXACT_TEMPLATE[capability]
    definition = _DEFINITIONS[capability]
    return _CATEGORY_TEMPLATE.get(definition.category, "custom_gameplay")


def _direct_features(capability: str) -> frozenset[str]:
    template_id = _template_id_for_known(capability)
    return _TEMPLATE_FEATURES[template_id]


def _feature_closure(capability: str, seen: set[str] | None = None) -> frozenset[str]:
    visited = set() if seen is None else seen
    if capability in visited or capability not in _DEFINITIONS:
        return frozenset()
    visited.add(capability)
    features = set(_direct_features(capability))
    for dependency in _DEFINITIONS[capability].default_dependencies:
        if dependency in _DEFINITIONS:
            features.update(_feature_closure(dependency, visited))
    return frozenset(features)


def profile_for_capability(
    capability: Any,
    *,
    semantic_type: str = "gameplay_mechanic",
) -> MinecraftTemplateProfile:
    raw = str(capability or "").strip().casefold().removeprefix("capability:")
    known = _known_capability(raw)
    if str(semantic_type or "").strip().casefold() == "software_quality":
        template_id = "software_quality"
        category = "software_quality"
        features = _TEMPLATE_FEATURES[template_id]
        canonical = known
        known_flag = known is not None
    elif known is None:
        template_id = "custom_gameplay"
        category = "custom"
        features = _TEMPLATE_FEATURES[template_id]
        canonical = None
        known_flag = False
    else:
        template_id = _template_id_for_known(known)
        category = _DEFINITIONS[known].category
        features = _feature_closure(known)
        canonical = known
        known_flag = True

    implementation = _TEMPLATE_IMPLEMENTATION_CAPABILITIES[template_id]
    artifacts = _TEMPLATE_ARTIFACTS[template_id]
    obligations = tuple(
        dict.fromkeys(
            (
                *_COMMON_DESIGN_OBLIGATIONS,
                *_TEMPLATE_DESIGN_OBLIGATIONS.get(template_id, ()),
            )
        )
    )
    return MinecraftTemplateProfile(
        capability=raw,
        canonical_capability=canonical,
        known_capability=known_flag,
        category=category,
        template_id=template_id,
        features=frozenset(features),
        implementation_capabilities=implementation,
        artifact_kinds=artifacts,
        design_resolution_obligations=obligations,
    )


def selected_predecessor_capabilities(
    capability: Any,
    selected_capabilities: Sequence[Any],
) -> tuple[str, ...]:
    canonical = _known_capability(capability)
    if canonical is None:
        return ()
    selected = {
        known
        for item in selected_capabilities
        if (known := _known_capability(item)) is not None
    }
    candidates = [
        *(
            dependency
            for dependency in _DEFINITIONS[canonical].default_dependencies
            if dependency in _DEFINITIONS
        ),
        *_SELECTED_PREDECESSORS.get(canonical, ()),
    ]
    return tuple(
        dict.fromkeys(
            candidate
            for candidate in candidates
            if candidate in selected and candidate != canonical
        )
    )


def requirement_branch_features(requirement: Mapping[str, Any]) -> frozenset[str]:
    profile = profile_for_capability(
        requirement.get("capability"),
        semantic_type=str(requirement.get("semantic_type") or "gameplay_mechanic"),
    )
    return profile.branch_features


__all__ = [
    "CUSTOM_CAPABILITY_SENTINEL",
    "FEATURE_CLIENT",
    "FEATURE_DATAGEN",
    "FEATURE_MIXIN",
    "FEATURE_NETWORK",
    "FEATURE_PERSISTENCE",
    "FEATURE_REGISTRY",
    "FEATURE_RUNTIME_TEST",
    "FEATURE_SERVER_AUTHORITY",
    "FEATURE_WORLDGEN",
    "MinecraftTemplateProfile",
    "RESEARCH_BASIS",
    "TEMPLATE_CATALOG_SCHEMA",
    "capability_catalog_for_model",
    "is_known_capability",
    "known_capability_ids",
    "profile_for_capability",
    "requirement_branch_features",
    "selected_predecessor_capabilities",
    "semantic_capability_choices",
]
