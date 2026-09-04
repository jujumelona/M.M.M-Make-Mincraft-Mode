from __future__ import annotations

"""Unified Canonical Capability Ontology and Subsystem Archetype Registry.

Single source of truth for:
1. Canonical atomic Minecraft mod capabilities.
2. High-level theme & composite concept archetype expansions (e.g. medieval, sci-fi, nuclear, magic, farming, rpg).
3. Functional mechanics archetype decompositions (e.g. machine/generator, vehicle/space, weapon/gun, portal/dimension).
4. Standard English search query templates for GitHub, Modrinth, and CurseForge.
"""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# 1. Atomic Canonical Capabilities
# ---------------------------------------------------------------------------
from enum import Enum
from typing import Any


class CapabilityOrigin(str, Enum):
    EXPLICIT = "explicit"
    ONTOLOGY = "ontology"
    SEMANTIC_INFERRED = "semantic_inferred"


@dataclass(frozen=True)
class CapabilityDefinition:
    id: str
    category: str
    description: str
    search_queries: tuple[str, ...]
    default_dependencies: tuple[str, ...] = ()


_ATOMIC_CAPABILITIES: dict[str, CapabilityDefinition] = {
    # Combat & Bosses
    "boss.entity": CapabilityDefinition(
        id="boss.entity",
        category="combat",
        description="Custom boss mob entity with AI goals, boss bar, and phase transitions",
        search_queries=("minecraft boss entity", "boss fight mod", "custom boss", "rpg boss battle"),
        default_dependencies=("entity.lifecycle", "combat.damage", "ui.menu", "loot.drop_table"),
    ),
    "combat.boss": CapabilityDefinition(
        id="combat.boss",
        category="combat",
        description="Boss combat mechanics, phases, and damage validation",
        search_queries=("minecraft boss combat", "boss attack phases mod", "custom boss abilities"),
        default_dependencies=("combat.damage", "network.action_sync"),
    ),
    "mob.spawning": CapabilityDefinition(
        id="mob.spawning",
        category="entity",
        description="Custom monster and mob spawning conditions and biomes",
        search_queries=("custom mob entity", "mob spawning mod", "custom monsters", "rpg mobs"),
        default_dependencies=("entity.lifecycle",),
    ),
    "entity.lifecycle": CapabilityDefinition(
        id="entity.lifecycle",
        category="entity",
        description="Entity registration, attributes, lifecycle events, and networking",
        search_queries=("custom entity registration fabric", "entity attributes lifecycle mod"),
        default_dependencies=(),
    ),
    "combat.damage": CapabilityDefinition(
        id="combat.damage",
        category="combat",
        description="Combat damage calculations, damage sources, and attack validation",
        search_queries=("custom damage source mod", "combat damage mechanics minecraft"),
        default_dependencies=(),
    ),

    # Items, Equipment, & Upgrades
    "item.equipment": CapabilityDefinition(
        id="item.equipment",
        category="item",
        description="Custom wearable equipment, armor sets, and held tools",
        search_queries=("custom equipment mod", "rpg armor weapons mod", "custom item equipment"),
        default_dependencies=("inventory.transfer",),
    ),
    "item.weapon": CapabilityDefinition(
        id="item.weapon",
        category="item",
        description="Custom weapons with unique attack damage and weapon effects",
        search_queries=("custom weapon mod", "rpg weapons", "combat weapon fabric"),
        default_dependencies=("combat.damage",),
    ),
    "item.armor": CapabilityDefinition(
        id="item.armor",
        category="item",
        description="Custom armor material, protection stats, and set bonuses",
        search_queries=("custom armor set mod", "armor material fabric mod"),
        default_dependencies=("item.equipment",),
    ),
    "item.upgrade": CapabilityDefinition(
        id="item.upgrade",
        category="crafting",
        description="Item enhancement, upgrade anvil, blacksmith refine, and scroll systems",
        search_queries=("item upgrade mod", "blacksmith enhancement mod", "equipment upgrade fabric", "weapon refine"),
        default_dependencies=("crafting.upgrade", "ui.container", "persistence.state_store"),
    ),
    "crafting.upgrade": CapabilityDefinition(
        id="crafting.upgrade",
        category="crafting",
        description="Upgrade workstation block and container recipe validation",
        search_queries=("upgrade station block mod", "enhancement table container"),
        default_dependencies=("block_entity.tick", "ui.container"),
    ),

    # Progression, Levels, Stats, & Quests
    "progression.level": CapabilityDefinition(
        id="progression.level",
        category="progression",
        description="Player level progression, experience points, and milestone unlocks",
        search_queries=("player leveling mod", "rpg level progression", "player stats exp mod", "leveling system"),
        default_dependencies=("progression.exp", "stat.growth", "persistence.state_store"),
    ),
    "progression.exp": CapabilityDefinition(
        id="progression.exp",
        category="progression",
        description="Custom experience point gain, curve calculation, and persistence",
        search_queries=("custom experience system mod", "exp curve rpg leveling"),
        default_dependencies=("persistence.state_store",),
    ),
    "stat.growth": CapabilityDefinition(
        id="stat.growth",
        category="progression",
        description="Player attribute scaling (health, mana, strength, defense) on level up",
        search_queries=("player attribute stats mod", "stat growth rpg system"),
        default_dependencies=("stat.attribute", "network.action_sync"),
    ),
    "stat.attribute": CapabilityDefinition(
        id="stat.attribute",
        category="progression",
        description="Custom player and entity attributes and modifiers",
        search_queries=("custom entity attributes fabric", "player attribute modifier"),
        default_dependencies=(),
    ),
    "quest.state": CapabilityDefinition(
        id="quest.state",
        category="quest",
        description="Quest log, objective tracking, quest state machine, and completion rewards",
        search_queries=("quest system mod fabric", "custom quests mod", "rpg quest progression"),
        default_dependencies=("quest.progression", "quest.reward", "ui.menu", "persistence.state_store"),
    ),
    "quest.progression": CapabilityDefinition(
        id="quest.progression",
        category="quest",
        description="Quest condition triggers, kill/collect tracking, and step transitions",
        search_queries=("quest objective trigger mod", "story quest progression"),
        default_dependencies=(),
    ),
    "quest.reward": CapabilityDefinition(
        id="quest.reward",
        category="quest",
        description="Quest item, exp, and currency reward distribution",
        search_queries=("quest reward distribution mod", "quest rewards"),
        default_dependencies=("inventory.transfer",),
    ),

    # Economy, Trade, & Shops
    "economy.currency": CapabilityDefinition(
        id="economy.currency",
        category="economy",
        description="Virtual currency, coin items, player balances, and transaction ledger",
        search_queries=("minecraft economy currency mod", "coin currency mod", "player money balance ledger"),
        default_dependencies=("persistence.state_store", "network.action_sync"),
    ),
    "trade.shop_registry": CapabilityDefinition(
        id="trade.shop_registry",
        category="economy",
        description="NPC and player shop registry, item buy/sell price catalogs, and stock",
        search_queries=("player shop mod fabric", "npc shop merchant mod", "custom shop registry"),
        default_dependencies=("trade.offer_model", "ui.menu", "economy.currency"),
    ),
    "trade.offer_model": CapabilityDefinition(
        id="trade.offer_model",
        category="economy",
        description="Trade offer validation, inventory debit/credit exchange transactions",
        search_queries=("custom trading offer mod", "villager trade transaction validation"),
        default_dependencies=("inventory.transfer",),
    ),

    # Machines, Energy, & Automation
    "energy.generator": CapabilityDefinition(
        id="energy.generator",
        category="technology",
        description="Energy generating machine (fusion reactor, generator, solar panel)",
        search_queries=("energy generator mod fabric", "tech generator machine", "power generation mod"),
        default_dependencies=("energy.production", "block_entity.tick", "energy.storage", "ui.container"),
    ),
    "energy.production": CapabilityDefinition(
        id="energy.production",
        category="technology",
        description="Energy generation rate calculation based on fuel consumption or inputs",
        search_queries=("energy production rate calculation", "fuel consumption energy mod"),
        default_dependencies=("energy.storage",),
    ),
    "energy.storage": CapabilityDefinition(
        id="energy.storage",
        category="technology",
        description="Energy capacity buffer, battery storage, and transfer cables/conduits",
        search_queries=("energy storage buffer mod", "teamreborn energy fabric", "battery machine mod"),
        default_dependencies=("persistence.state_store", "network.action_sync"),
    ),
    "block_entity.tick": CapabilityDefinition(
        id="block_entity.tick",
        category="technology",
        description="Ticking block entity with state storage, inventory sync, and world interaction",
        search_queries=("block entity ticker fabric", "custom block entity ticking state"),
        default_dependencies=("persistence.state_store", "network.action_sync"),
    ),
    "automation.machine": CapabilityDefinition(
        id="automation.machine",
        category="technology",
        description="Automated processing machine with input/output slots and processing recipes",
        search_queries=("tech machine processing mod", "automated crafter fabric mod"),
        default_dependencies=("block_entity.tick", "inventory.transfer", "ui.container"),
    ),

    # Vehicles, Transport, & Portals
    "entity.vehicle": CapabilityDefinition(
        id="entity.vehicle",
        category="transport",
        description="Controllable vehicle, mount, spaceship, or boat entity with steering physics",
        search_queries=("custom vehicle mod fabric", "controllable vehicle entity", "spaceship entity mod"),
        default_dependencies=("entity.lifecycle", "network.action_sync"),
    ),
    "worldgen.dimension": CapabilityDefinition(
        id="worldgen.dimension",
        category="worldgen",
        description="Custom dimension world generator, chunk generator, and biome provider",
        search_queries=("custom dimension mod fabric", "custom dimension chunk generator"),
        default_dependencies=("worldgen.structure", "teleport.portal"),
    ),
    "teleport.portal": CapabilityDefinition(
        id="teleport.portal",
        category="transport",
        description="Dimensional teleportation portal block, structure frame, and teleport logic",
        search_queries=("custom portal block mod", "teleport gateway dimension fabric"),
        default_dependencies=("persistence.state_store",),
    ),

    # Worldgen & Structures
    "worldgen.structure": CapabilityDefinition(
        id="worldgen.structure",
        category="worldgen",
        description="Custom structure placement, jigsaw templates, and dungeon generation",
        search_queries=("custom structure mod fabric", "dungeon worldgen structure fabric", "prefab structure"),
        default_dependencies=(),
    ),
    "worldgen.dungeon": CapabilityDefinition(
        id="worldgen.dungeon",
        category="worldgen",
        description="Multi-room procedural dungeon structure with spawners and loot chests",
        search_queries=("dungeon generation mod fabric", "custom dungeon structures", "roguelike dungeon"),
        default_dependencies=("worldgen.structure", "loot.drop_table", "mob.spawning"),
    ),
    "loot.drop_table": CapabilityDefinition(
        id="loot.drop_table",
        category="worldgen",
        description="Custom loot table json definitions for mobs, chests, and block drops",
        search_queries=("custom loot table mod", "mob drop table fabric", "rpg loot drops"),
        default_dependencies=("inventory.transfer",),
    ),

    # Skills & Magic
    "skill.ability": CapabilityDefinition(
        id="skill.ability",
        category="magic",
        description="Active and passive player skills, cooldown manager, and spell casting",
        search_queries=("player skill system mod", "magic ability skills fabric", "spell casting mod"),
        default_dependencies=("skill.magic", "combat.damage", "ui.menu", "network.action_sync"),
    ),
    "skill.magic": CapabilityDefinition(
        id="skill.magic",
        category="magic",
        description="Magic projectile entities, spell particle effects, and mana cost consumption",
        search_queries=("magic spells mod fabric", "wizard spells mana mod"),
        default_dependencies=("combat.damage",),
    ),

    # UI, Menus, & Storage
    "ui.menu": CapabilityDefinition(
        id="ui.menu",
        category="ui",
        description="Custom client GUI screen, HUD overlay, and menu buttons",
        search_queries=("custom screen gui fabric", "hud overlay screen fabric mod"),
        default_dependencies=("network.action_sync",),
    ),
    "ui.container": CapabilityDefinition(
        id="ui.container",
        category="ui",
        description="Synced ScreenHandler container with inventory slot sync and widget layout",
        search_queries=("screenhandler container fabric", "gui container inventory sync"),
        default_dependencies=("inventory.transfer", "network.action_sync"),
    ),
    "inventory.transfer": CapabilityDefinition(
        id="inventory.transfer",
        category="storage",
        description="Item inventory slot validation, item stack insertion, and transfer logic",
        search_queries=("inventory helper fabric", "item stack transfer validation"),
        default_dependencies=(),
    ),
    "network.action_sync": CapabilityDefinition(
        id="network.action_sync",
        category="network",
        description="Client-to-server C2S packets, server-to-client S2C sync, and payload codecs",
        search_queries=("custom payload networking fabric", "c2s s2c packet sync fabric"),
        default_dependencies=(),
    ),
    "persistence.state_store": CapabilityDefinition(
        id="persistence.state_store",
        category="storage",
        description="Server world saved data persistent state storage with NBT codec",
        search_queries=("persistent state saved data fabric", "world nbt serialization fabric"),
        default_dependencies=(),
    ),

    # Gameplay concepts used by the semantic request compiler. These stay above
    # loader/API primitives so authored behavior is never replaced by an incidental
    # implementation detail such as ``block_entity.tick``.
    "resource.mining": CapabilityDefinition(
        id="resource.mining",
        category="resource",
        description="Mine resources and award the resulting drops to the player",
        search_queries=("minecraft resource mining mechanic", "custom mining drops mod"),
        default_dependencies=("inventory.transfer",),
    ),
    "resource.farming": CapabilityDefinition(
        id="resource.farming",
        category="resource",
        description="Repeatable acquisition of named resources with explicit drops, amounts, inventory transfer, and consumption sinks",
        search_queries=("minecraft resource farming inventory drops", "fabric resource acquisition gametest"),
        default_dependencies=("inventory.transfer", "loot.drop_table"),
    ),
    "economy.reward": CapabilityDefinition(
        id="economy.reward",
        category="economy",
        description="Award currency for completed gameplay actions",
        search_queries=("minecraft currency reward system", "economy action rewards mod"),
        default_dependencies=("economy.currency",),
    ),
    "economy.trade": CapabilityDefinition(
        id="economy.trade",
        category="economy",
        description="Buy and sell goods through validated shop transactions",
        search_queries=("minecraft shop trade transaction mod", "player shop economy fabric"),
        default_dependencies=(
            "economy.currency",
            "ui.shop",
            "network.transaction",
            "persistence.balance",
        ),
    ),
    "ui.shop": CapabilityDefinition(
        id="ui.shop",
        category="ui",
        description="Player-facing shop inventory and trade interaction",
        search_queries=("minecraft shop menu screen mod", "fabric custom shop ui"),
        default_dependencies=("ui.menu", "network.transaction"),
    ),
    "network.transaction": CapabilityDefinition(
        id="network.transaction",
        category="network",
        description="Server-authoritative transaction request and response protocol",
        search_queries=("fabric transaction packet validation",),
        default_dependencies=("network.action_sync",),
    ),
    "persistence.balance": CapabilityDefinition(
        id="persistence.balance",
        category="storage",
        description="Persistent player currency balance and transaction ledger",
        search_queries=("minecraft persistent player balance",),
        default_dependencies=("persistence.state_store",),
    ),
    "spaceship.component_crafting": CapabilityDefinition(
        id="spaceship.component_crafting",
        category="crafting",
        description="Craft functional spaceship components from gathered resources",
        search_queries=("minecraft spaceship component crafting mod",),
        default_dependencies=("crafting.recipe",),
    ),
    "spacecraft.component_construction": CapabilityDefinition(
        id="spacecraft.component_construction",
        category="transport",
        description="Acquire compatible spacecraft parts through shared economy/inventory services and assemble them into persistent slot state",
        search_queries=("minecraft fabric modular spaceship parts assembly", "vehicle component slot assembly gametest"),
        default_dependencies=("crafting.recipe", "economy.trade", "persistence.state_store"),
    ),
    "spacecraft.weapon_upgrade": CapabilityDefinition(
        id="spacecraft.weapon_upgrade",
        category="combat",
        description="Purchase and install tiered spacecraft weapon modules with server-authoritative combat stats",
        search_queries=("minecraft vehicle weapon upgrade slots", "fabric tiered weapon module server validation"),
        default_dependencies=("combat.weapon", "economy.trade", "persistence.state_store"),
    ),
    "spacecraft.performance_upgrade": CapabilityDefinition(
        id="spacecraft.performance_upgrade",
        category="transport",
        description="Upgrade gameplay spacecraft stats such as thrust, speed, fuel capacity, hull durability and cargo capacity",
        search_queries=("minecraft vehicle gameplay stat upgrade tiers", "fabric spaceship fuel durability speed upgrade"),
        default_dependencies=("economy.trade", "persistence.state_store", "network.action_sync"),
    ),
    "spacecraft.expansion": CapabilityDefinition(
        id="spacecraft.expansion",
        category="transport",
        description="Install spacecraft expansion modules that add explicit cargo, crew or module capacity",
        search_queries=("minecraft modular vehicle expansion slots", "fabric persistent cargo module capacity"),
        default_dependencies=("economy.trade", "persistence.state_store"),
    ),
    "spaceship.vehicle": CapabilityDefinition(
        id="spaceship.vehicle",
        category="transport",
        description="A player-operated spaceship used for travel between destinations",
        search_queries=("minecraft controllable spaceship vehicle mod",),
        default_dependencies=("entity.vehicle", "network.action_sync"),
    ),
    "crafting.recipe": CapabilityDefinition(
        id="crafting.recipe",
        category="crafting",
        description="Validated data-driven crafting recipes",
        search_queries=("minecraft custom crafting recipe fabric",),
        default_dependencies=(),
    ),
    "crew.npc": CapabilityDefinition(
        id="crew.npc",
        category="entity",
        description="Crew non-player characters with persistent identity",
        search_queries=("minecraft recruitable crew npc mod",),
        default_dependencies=("entity.lifecycle",),
    ),
    "crew.recruitment": CapabilityDefinition(
        id="crew.recruitment",
        category="progression",
        description="Recruit, retain, and manage a spaceship crew",
        search_queries=("minecraft npc recruitment system",),
        default_dependencies=("crew.npc", "persistence.state_store"),
    ),
    "space.travel": CapabilityDefinition(
        id="space.travel",
        category="transport",
        description="Travel from the current world into space destinations",
        search_queries=("minecraft space travel spaceship mod",),
        default_dependencies=("entity.vehicle", "network.action_sync"),
    ),
    "space.launch": CapabilityDefinition(
        id="space.launch",
        category="transport",
        description="Evaluate explicit required versus optional launch unlocks, consume fuel and transition to a selected destination",
        search_queries=("minecraft spaceship launch fuel destination dimension", "fabric server authoritative dimension travel gametest"),
        default_dependencies=("space.travel", "persistence.state_store", "network.action_sync"),
    ),
    "worldgen.planet": CapabilityDefinition(
        id="worldgen.planet",
        category="worldgen",
        description="Generate visitable planets with target-specific terrain",
        search_queries=("minecraft planet world generation mod",),
        default_dependencies=("worldgen.dimension",),
    ),
    "resource.special_ore": CapabilityDefinition(
        id="resource.special_ore",
        category="resource",
        description="Spawn and mine a distinct special ore resource",
        search_queries=("minecraft custom special ore worldgen",),
        default_dependencies=("resource.mining", "worldgen.planet"),
    ),
    "planet.special_mineral": CapabilityDefinition(
        id="planet.special_mineral",
        category="worldgen",
        description="Generate, mine, tag, loot and consume named special minerals on an accessible planet",
        search_queries=("minecraft custom planet ore configured placed feature", "fabric ore loot tag recipe datagen"),
        default_dependencies=("worldgen.planet", "resource.special_ore", "space.launch"),
    ),
    "alien.entity": CapabilityDefinition(
        id="alien.entity",
        category="entity",
        description="Alien entities that participate in hostile encounters",
        search_queries=("minecraft alien mob entity mod",),
        default_dependencies=("entity.lifecycle", "combat.damage"),
    ),
    "alien.combat": CapabilityDefinition(
        id="alien.combat",
        category="combat",
        description="Planet-aware alien spawn, AI, attributes, attacks, damage, death and loot behavior",
        search_queries=("minecraft fabric hostile alien entity AI loot", "fabric entity combat gametest spawn drops"),
        default_dependencies=("alien.entity", "combat.weapon", "space.launch"),
    ),
    "combat.weapon": CapabilityDefinition(
        id="combat.weapon",
        category="combat",
        description="Weapons and validated attacks used in combat encounters",
        search_queries=("minecraft custom combat weapon mod",),
        default_dependencies=("item.weapon", "combat.damage"),
    ),
    "colony.settlement": CapabilityDefinition(
        id="colony.settlement",
        category="progression",
        description="Create and maintain a player colony settlement",
        search_queries=("minecraft colony settlement building mod",),
        default_dependencies=("colony.progression", "persistence.state_store"),
    ),
    "colony.colonization": CapabilityDefinition(
        id="colony.colonization",
        category="progression",
        description="Validate planet colony placement and persist ownership, storage and staged development",
        search_queries=("minecraft colony placement ownership persistence", "fabric settlement progression storage gametest"),
        default_dependencies=("colony.settlement", "space.launch", "network.action_sync"),
    ),
    "colony.progression": CapabilityDefinition(
        id="colony.progression",
        category="progression",
        description="Advance colony population, buildings, and unlocks",
        search_queries=("minecraft colony progression system",),
        default_dependencies=("persistence.state_store",),
    ),
}

# ---------------------------------------------------------------------------
# 2. High-Level Theme & Functional Archetype Expansions
# ---------------------------------------------------------------------------

_THEME_ARCHETYPES: dict[str, tuple[str, ...]] = {
    # Medieval / Kingdom / Fantasy
    "medieval": (
        "trade.shop_registry", "economy.currency", "item.equipment", "item.weapon",
        "item.armor", "combat.damage", "quest.state", "crafting.upgrade",
        "worldgen.structure", "loot.drop_table", "ui.menu",
    ),
    "fantasy": (
        "skill.ability", "skill.magic", "item.equipment", "boss.entity",
        "mob.spawning", "worldgen.dungeon", "loot.drop_table", "progression.level",
    ),

    # RPG / MapleStory / MMO
    "rpg": (
        "progression.level", "stat.growth", "boss.entity", "mob.spawning",
        "item.equipment", "item.upgrade", "quest.state", "loot.drop_table",
        "skill.ability", "ui.menu",
    ),
    "maplestory": (
        "boss.entity", "mob.spawning", "item.equipment", "progression.level",
        "item.upgrade", "loot.drop_table", "skill.ability", "ui.menu",
    ),

    # Sci-Fi / Space / Cybernetics
    "space": (
        "entity.vehicle", "worldgen.dimension", "energy.generator",
        "energy.storage", "block_entity.tick", "ui.container", "item.equipment",
    ),
    "sci_fi": (
        "energy.generator", "energy.storage", "automation.machine",
        "item.equipment", "item.weapon", "ui.container", "block_entity.tick",
    ),
    "cyberpunk": (
        "stat.growth", "item.equipment", "item.weapon", "energy.storage",
        "skill.ability", "ui.menu", "combat.damage",
    ),

    # Tech / Nuclear / Energy / Industry
    "nuclear": (
        "energy.generator", "energy.production", "energy.storage",
        "block_entity.tick", "ui.container", "automation.machine",
    ),
    "industry": (
        "energy.generator", "energy.storage", "automation.machine",
        "block_entity.tick", "inventory.transfer", "ui.container",
    ),
}

_FUNCTIONAL_ARCHETYPES: dict[str, tuple[str, ...]] = {
    # Generators / Machines / Reactors
    "generator": ("energy.generator", "energy.production", "block_entity.tick", "energy.storage", "ui.container"),
    "machine": ("automation.machine", "block_entity.tick", "inventory.transfer", "ui.container"),
    "reactor": ("energy.generator", "energy.production", "energy.storage", "block_entity.tick"),
    "battery": ("energy.storage", "block_entity.tick", "network.action_sync"),

    # Weapons / Combat / Guns
    "weapon": ("item.weapon", "combat.damage"),
    "gun": ("item.weapon", "combat.damage", "network.action_sync"),
    "sword": ("item.weapon", "combat.damage"),
    "armor": ("item.armor", "item.equipment"),
    "equipment": ("item.equipment", "inventory.transfer"),

    # Bosses / Mobs / Entities
    "boss": ("boss.entity", "combat.boss", "loot.drop_table"),
    "mob": ("mob.spawning", "entity.lifecycle"),
    "monster": ("mob.spawning", "entity.lifecycle"),

    # Progression / Leveling / Upgrades
    "level": ("progression.level", "stat.growth"),
    "upgrade": ("item.upgrade", "crafting.upgrade"),
    "loot": ("loot.drop_table", "inventory.transfer"),

    # Vehicles / Space / Portals / Dimensions
    "spaceship": ("entity.vehicle", "network.action_sync"),
    "vehicle": ("entity.vehicle", "network.action_sync"),
    "portal": ("teleport.portal", "worldgen.dimension"),
    "dimension": ("worldgen.dimension", "teleport.portal"),

    # Economy / Shops / Quests
    "trade": ("trade.transaction", "trade.validation", "trade.offer_model", "trade.shop_registry", "economy.currency"),
    "shop": ("ui.shop_menu", "trade.shop_registry", "ui.menu", "economy.currency"),
    "quest": ("quest.state", "quest.progression", "quest.reward"),
    "dungeon": ("worldgen.dungeon", "worldgen.structure", "loot.drop_table"),
    "skill": ("skill.ability", "skill.magic", "combat.damage"),
}

# Additional canonical aliases.
_FUNCTIONAL_ARCHETYPES.update(
    {
        "block": ("gameplay.block",),
        "screen": ("ui.menu",),
        "packet": ("network.action_sync",),
        "packets": ("network.action_sync",),
        "saved": ("persistence.state_store",),
    }
)

# Unified domain term mapping covering atomic, theme, and functional terms
_CANONICAL_DOMAIN_MAP: dict[str, tuple[str, ...]] = {}
for _k, _v in _THEME_ARCHETYPES.items():
    _CANONICAL_DOMAIN_MAP[_k.casefold()] = _v
for _k, _v in _FUNCTIONAL_ARCHETYPES.items():
    if _k.casefold() not in _CANONICAL_DOMAIN_MAP:
        _CANONICAL_DOMAIN_MAP[_k.casefold()] = _v
for _cap_id, _cap_def in _ATOMIC_CAPABILITIES.items():
    _CANONICAL_DOMAIN_MAP[_cap_id.casefold()] = (_cap_id, *_cap_def.default_dependencies)
    short_stem = _cap_id.split(".")[-1]
    if short_stem not in _CANONICAL_DOMAIN_MAP:
        _CANONICAL_DOMAIN_MAP[short_stem] = (_cap_id,)

@dataclass(frozen=True)
class CapabilityResolutionNode:
    capability_id: str
    source_span: str
    origin: str  # "explicit" | "archetype_inferred" | "dependency_required" | "unresolved_concept"
    confidence: float = 1.0
    is_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "source_span": self.source_span,
            "origin": self.origin,
            "confidence": self.confidence,
            "is_required": self.is_required,
        }


@dataclass(frozen=True)
class CapabilityResolution:
    nodes: tuple[CapabilityResolutionNode, ...]
    edges: tuple[tuple[str, str], ...]  # (parent, required_dependency)
    unresolved_spans: tuple[str, ...] = ()

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(node.capability_id for node in self.nodes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/capability-resolution-v1",
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [{"from": u, "to": v} for u, v in self.edges],
            "unresolved_spans": list(self.unresolved_spans),
        }



_CLAUSE_SPLIT = re.compile(
    r"\s*(?:,|;|→|->|=>|/|\||•|\u2022|\u25b6|\u25cf|\u2013|\u2014|\n|\r)\s*",
    re.UNICODE,
)


def _extract_canonical_tokens(word: str) -> tuple[str, ...]:
    """Extract canonical domain keys without language-specific morphology."""
    low = word.casefold().strip()
    if not low:
        return ()
    if low in _CANONICAL_DOMAIN_MAP or low in _THEME_ARCHETYPES or low in _FUNCTIONAL_ARCHETYPES:
        return (low,)

    matched_keys: list[str] = []
    for key in sorted(_CANONICAL_DOMAIN_MAP.keys(), key=lambda item: -len(item)):
        if len(key) >= 2 and key in low and key not in matched_keys:
            matched_keys.append(key)
    for theme_key in sorted(_THEME_ARCHETYPES.keys(), key=lambda item: -len(item)):
        if len(theme_key) >= 2 and theme_key in low and theme_key not in matched_keys:
            matched_keys.append(theme_key)
    return tuple(matched_keys)


def resolve_capabilities_from_phrase_structured(phrase: str) -> CapabilityResolution:
    """Resolve a user phrase to a structured capability graph with provenance and requires-edges."""
    clean = str(phrase or "").strip()
    if not clean:
        default_node = CapabilityResolutionNode(
            capability_id="gameplay.core",
            source_span=clean,
            origin="explicit",
            confidence=0.5,
        )
        return CapabilityResolution(nodes=(default_node,), edges=(), unresolved_spans=())

    # No language-specific tokens are silently dropped.  The ontology is a
    # canonicalization / alias / dependency-expansion layer only — it must not
    # make semantic decisions about which words to ignore.  Tokens absent from
    # the alias map produce unresolved: IDs, which the Semantic Model layer
    # resolves.  An empty ignored set ensures all scripts are treated equally.
    ignored: frozenset[str] = frozenset()

    # Split only on language-neutral structural delimiters (comma, semicolon,
    raw_clauses = [
        clause.strip()
        for clause in _CLAUSE_SPLIT.split(clean)
        if clause.strip()
    ]
    if not raw_clauses:
        raw_clauses = [clean]

    nodes: list[CapabilityResolutionNode] = []
    seen_caps: set[str] = set()
    unresolved_spans: list[str] = []

    for clause in raw_clauses:
        words = re.findall(r"[\w]+", clause, re.UNICODE)
        clause_unresolved_words: list[str] = []

        for word in words:
            low = word.casefold()
            if low in ignored:
                continue

            extracted_keys = _extract_canonical_tokens(word)
            if not extracted_keys:
                clause_unresolved_words.append(word)
                continue

            for canon_key in extracted_keys:
                if canon_key in _THEME_ARCHETYPES:
                    for cap in _THEME_ARCHETYPES[canon_key]:
                        if cap not in seen_caps:
                            seen_caps.add(cap)
                            nodes.append(
                                CapabilityResolutionNode(
                                    capability_id=cap,
                                    source_span=word,
                                    origin="archetype_inferred",
                                    confidence=0.85,
                                    is_required=False,
                                )
                            )
                elif canon_key in _FUNCTIONAL_ARCHETYPES or canon_key in _CANONICAL_DOMAIN_MAP:
                    caps = _CANONICAL_DOMAIN_MAP.get(canon_key, ())
                    for idx, cap in enumerate(caps):
                        if cap in seen_caps:
                            # A capability may first occur as a dependency of an
                            # earlier noun (for example, a shop) and later be
                            # explicitly requested by an action (trade).  Preserve
                            # the stronger authored binding instead of leaving it
                            # permanently classified as implementation plumbing.
                            if idx == 0:
                                for node_index, existing in enumerate(nodes):
                                    if (
                                        existing.capability_id == cap
                                        and existing.origin == "dependency_required"
                                    ):
                                        nodes[node_index] = replace(
                                            existing,
                                            source_span=word,
                                            origin="explicit",
                                            confidence=0.95,
                                            is_required=True,
                                        )
                                        break
                            continue
                        else:
                            seen_caps.add(cap)
                            nodes.append(
                                CapabilityResolutionNode(
                                    capability_id=cap,
                                    source_span=word,
                                    origin="explicit" if idx == 0 else "dependency_required",
                                    confidence=0.95 if idx == 0 else 0.85,
                                    is_required=True if idx == 0 else False,
                                )
                            )

        if clause_unresolved_words:
            clause_unresolved = " ".join(clause_unresolved_words)
            slug = re.sub(r"[^a-z0-9_]+", "_", clause_unresolved.casefold()).strip("_")
            slug = re.sub(r"_+", "_", slug)
            if not slug:
                slug = hashlib.sha256(clause_unresolved.encode("utf-8")).hexdigest()[:12]
            if slug:
                cap_id = f"unresolved:{slug[:48]}"
                if cap_id not in seen_caps:
                    seen_caps.add(cap_id)
                    nodes.append(
                        CapabilityResolutionNode(
                            capability_id=cap_id,
                            source_span=clause_unresolved,
                            origin="unresolved_concept",
                            confidence=0.70,
                            is_required=True,
                        )
                    )
                    unresolved_spans.append(clause_unresolved)

    if not nodes:
        nodes.append(
            CapabilityResolutionNode(
                capability_id="gameplay.core",
                source_span=clean,
                origin="explicit",
                confidence=0.5,
            )
        )

    # Establish explicit directed requires edges from ontology definitions
    edges: list[tuple[str, str]] = []
    for node in nodes:
        cap_def = _ATOMIC_CAPABILITIES.get(node.capability_id)
        if cap_def and cap_def.default_dependencies:
            for dep in cap_def.default_dependencies:
                if dep in seen_caps and (node.capability_id, dep) not in edges:
                    edges.append((node.capability_id, dep))

    return CapabilityResolution(
        nodes=tuple(nodes),
        edges=tuple(edges),
        unresolved_spans=tuple(unresolved_spans),
    )


def resolve_capabilities_from_phrase(phrase: str) -> tuple[str, ...]:
    """Resolve a user phrase to canonical capabilities via ontology or dynamic slugification."""
    res = resolve_capabilities_from_phrase_structured(phrase)
    return res.capability_ids


def search_queries_for_capability(capability: str) -> tuple[str, ...]:
    """Return targeted English search queries for a canonical or dynamic capability."""
    clean_cap = capability.removeprefix("unresolved:").removeprefix("provisional:")
    if clean_cap in _ATOMIC_CAPABILITIES:
        return _ATOMIC_CAPABILITIES[clean_cap].search_queries
    # For composite or dynamic capabilities, construct clean Minecraft mod queries
    tokens = clean_cap.replace(".", " ").replace("_", " ").split()
    joined = " ".join(tokens)
    return (
        f"{joined} mod",
        f"minecraft {joined}",
        f"{joined} fabric mod",
    )


def canonical_domain_map() -> Mapping[str, tuple[str, ...]]:
    """Return an immutable view of the unified domain mapping."""
    return dict(_CANONICAL_DOMAIN_MAP)


def atomic_capability_definitions() -> Mapping[str, CapabilityDefinition]:
    """Return the registry of atomic capability definitions."""
    return dict(_ATOMIC_CAPABILITIES)


@dataclass(frozen=True)
class CapabilityRequirementContract:
    requirement_id: str
    capability_id: str
    description: str
    acceptance_pattern: str


_CAPABILITY_REQUIREMENT_CONTRACTS: dict[str, tuple[CapabilityRequirementContract, ...]] = {
    "boss.entity": (
        CapabilityRequirementContract("REQ-BOSS-001", "boss.entity", "Boss entity spawn and initialization", "spawn|init|entity"),
        CapabilityRequirementContract("REQ-BOSS-002", "boss.entity", "Boss health and state persistence", "health|state|persist"),
        CapabilityRequirementContract("REQ-BOSS-003", "boss.entity", "Boss phase transition mechanics", "phase|transition|ai|goal"),
        CapabilityRequirementContract("REQ-BOSS-004", "boss.entity", "Boss death rewards and drop table", "death|loot|drop|reward"),
    ),
    "combat.boss": (
        CapabilityRequirementContract("REQ-BOSS-001", "combat.boss", "Boss attack phase orchestration", "phase|attack|combat"),
        CapabilityRequirementContract("REQ-BOSS-002", "combat.boss", "Boss damage validation and immunity", "damage|immunity|hit"),
    ),
    "item.equipment": (
        CapabilityRequirementContract("REQ-ITEM-001", "item.equipment", "Item registry and equipment attributes", "item|equip|attr|registry"),
        CapabilityRequirementContract("REQ-ITEM-002", "item.equipment", "Durability and usage behavior", "durability|use|usage|tier"),
    ),
    "combat.damage": (
        CapabilityRequirementContract("REQ-COMBAT-001", "combat.damage", "Damage source calculation and attributes", "damage|calc|source"),
        CapabilityRequirementContract("REQ-COMBAT-002", "combat.damage", "Knockback and hit reactions", "knockback|hit|reaction"),
    ),
    "worldgen.ore": (
        CapabilityRequirementContract("REQ-WORLD-001", "worldgen.ore", "Ore feature registry and placement modifier", "ore|feature|world|placement"),
    ),
    "magic.spell": (
        CapabilityRequirementContract("REQ-MAGIC-001", "magic.spell", "Spell casting invocation and mana consumption", "spell|cast|mana"),
        CapabilityRequirementContract("REQ-MAGIC-002", "magic.spell", "Spell projectile effect execution", "projectile|effect|drain"),
    ),
}


def capability_requirement_contracts(capability_id: str) -> tuple[CapabilityRequirementContract, ...]:
    """Return formal requirement contracts for a capability."""
    clean = capability_id.removeprefix("unresolved:").removeprefix("provisional:")
    if clean in _CAPABILITY_REQUIREMENT_CONTRACTS:
        return _CAPABILITY_REQUIREMENT_CONTRACTS[clean]
    dom = clean.split(".")[0].upper() if "." in clean else "CAP"
    return (
        CapabilityRequirementContract(
            requirement_id=f"REQ-{dom}-001",
            capability_id=clean,
            description=f"Core behavioral acceptance contract for {clean}",
            acceptance_pattern=clean.split(".")[-1].replace("_", "|"),
        ),
    )
