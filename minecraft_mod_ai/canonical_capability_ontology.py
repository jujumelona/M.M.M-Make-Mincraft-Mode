from __future__ import annotations

"""Unified Canonical Capability Ontology and Subsystem Archetype Registry.

Single source of truth for:
1. Canonical atomic Minecraft mod capabilities.
2. High-level theme & composite concept archetype expansions (e.g. medieval, sci-fi, nuclear, magic, farming, rpg).
3. Functional mechanics archetype decompositions (e.g. machine/generator, vehicle/space, weapon/gun, portal/dimension).
4. Standard English search query templates for GitHub, Modrinth, and CurseForge.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# 1. Atomic Canonical Capabilities
# ---------------------------------------------------------------------------

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
    "중세": (
        "trade.shop_registry", "economy.currency", "item.equipment", "item.weapon",
        "item.armor", "combat.damage", "quest.state", "crafting.upgrade",
        "worldgen.structure", "loot.drop_table", "ui.menu",
    ),
    "fantasy": (
        "skill.ability", "skill.magic", "item.equipment", "boss.entity",
        "mob.spawning", "worldgen.dungeon", "loot.drop_table", "progression.level",
    ),
    "판타지": (
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
    "메이플": (
        "boss.entity", "mob.spawning", "item.equipment", "progression.level",
        "item.upgrade", "loot.drop_table", "skill.ability", "ui.menu",
    ),
    "메이플스토리": (
        "boss.entity", "mob.spawning", "item.equipment", "progression.level",
        "item.upgrade", "loot.drop_table", "skill.ability", "ui.menu",
    ),

    # Sci-Fi / Space / Cybernetics
    "space": (
        "entity.vehicle", "worldgen.dimension", "energy.generator",
        "energy.storage", "block_entity.tick", "ui.container", "item.equipment",
    ),
    "우주": (
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
    "사이버": (
        "stat.growth", "item.equipment", "item.weapon", "energy.storage",
        "skill.ability", "ui.menu", "combat.damage",
    ),

    # Tech / Nuclear / Energy / Industry
    "nuclear": (
        "energy.generator", "energy.production", "energy.storage",
        "block_entity.tick", "ui.container", "automation.machine",
    ),
    "핵융합": (
        "energy.generator", "energy.production", "energy.storage",
        "block_entity.tick", "ui.container", "automation.machine",
    ),
    "원자로": (
        "energy.generator", "energy.production", "energy.storage",
        "block_entity.tick", "ui.container", "automation.machine",
    ),
    "industry": (
        "energy.generator", "energy.storage", "automation.machine",
        "block_entity.tick", "inventory.transfer", "ui.container",
    ),
    "산업": (
        "energy.generator", "energy.storage", "automation.machine",
        "block_entity.tick", "inventory.transfer", "ui.container",
    ),
}

_FUNCTIONAL_ARCHETYPES: dict[str, tuple[str, ...]] = {
    # Generators / Machines / Reactors
    "generator": ("energy.generator", "energy.production", "block_entity.tick", "energy.storage", "ui.container"),
    "발전기": ("energy.generator", "energy.production", "block_entity.tick", "energy.storage", "ui.container"),
    "machine": ("automation.machine", "block_entity.tick", "inventory.transfer", "ui.container"),
    "기계": ("automation.machine", "block_entity.tick", "inventory.transfer", "ui.container"),
    "reactor": ("energy.generator", "energy.production", "energy.storage", "block_entity.tick"),
    "battery": ("energy.storage", "block_entity.tick", "network.action_sync"),
    "배터리": ("energy.storage", "block_entity.tick", "network.action_sync"),

    # Weapons / Combat / Guns
    "weapon": ("item.weapon", "combat.damage"),
    "무기": ("item.weapon", "combat.damage"),
    "gun": ("item.weapon", "combat.damage", "network.action_sync"),
    "총기": ("item.weapon", "combat.damage", "network.action_sync"),
    "sword": ("item.weapon", "combat.damage"),
    "검": ("item.weapon", "combat.damage"),
    "armor": ("item.armor", "item.equipment"),
    "방어구": ("item.armor", "item.equipment"),
    "equipment": ("item.equipment", "inventory.transfer"),
    "장비": ("item.equipment", "inventory.transfer"),

    # Bosses / Mobs / Entities
    "boss": ("boss.entity", "combat.boss", "loot.drop_table"),
    "보스": ("boss.entity", "combat.boss", "loot.drop_table"),
    "mob": ("mob.spawning", "entity.lifecycle"),
    "잡몹": ("mob.spawning", "entity.lifecycle"),
    "몬스터": ("mob.spawning", "entity.lifecycle"),
    "monster": ("mob.spawning", "entity.lifecycle"),

    # Progression / Leveling / Upgrades
    "level": ("progression.level", "stat.growth"),
    "레벨": ("progression.level", "stat.growth"),
    "성장": ("progression.level", "stat.growth"),
    "upgrade": ("item.upgrade", "crafting.upgrade"),
    "강화": ("item.upgrade", "crafting.upgrade"),
    "제련": ("item.upgrade", "crafting.upgrade"),
    "loot": ("loot.drop_table", "inventory.transfer"),
    "드롭": ("loot.drop_table", "inventory.transfer"),
    "드랍": ("loot.drop_table", "inventory.transfer"),

    # Vehicles / Space / Portals / Dimensions
    "spaceship": ("entity.vehicle", "network.action_sync"),
    "우주선": ("entity.vehicle", "network.action_sync"),
    "vehicle": ("entity.vehicle", "network.action_sync"),
    "탈것": ("entity.vehicle", "network.action_sync"),
    "portal": ("teleport.portal", "worldgen.dimension"),
    "포탈": ("teleport.portal", "worldgen.dimension"),
    "게이트": ("teleport.portal", "worldgen.dimension"),
    "dimension": ("worldgen.dimension", "teleport.portal"),
    "디멘션": ("worldgen.dimension", "teleport.portal"),

    # Economy / Shops / Quests
    "trade": ("trade.transaction", "trade.validation", "trade.offer_model", "trade.shop_registry", "economy.currency"),
    "거래": ("trade.transaction", "trade.validation", "trade.offer_model", "trade.shop_registry", "economy.currency"),
    "shop": ("ui.shop_menu", "trade.shop_registry", "ui.menu", "economy.currency"),
    "상점": ("ui.shop_menu", "trade.shop_registry", "ui.menu", "economy.currency"),
    "quest": ("quest.state", "quest.progression", "quest.reward"),
    "퀘스트": ("quest.state", "quest.progression", "quest.reward"),
    "dungeon": ("worldgen.dungeon", "worldgen.structure", "loot.drop_table"),
    "던전": ("worldgen.dungeon", "worldgen.structure", "loot.drop_table"),
    "skill": ("skill.ability", "skill.magic", "combat.damage"),
    "스킬": ("skill.ability", "skill.magic", "combat.damage"),
}

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

# ---------------------------------------------------------------------------
# 3. Universal Romanization & Token Helpers
# ---------------------------------------------------------------------------

_CHOSUNG = (
    "g", "gg", "n", "d", "dd", "r", "m", "b", "bb", "s",
    "ss", "", "j", "jj", "c", "k", "t", "p", "h"
)
_JUNGSUNG = (
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa",
    "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"
)
_JONGSUNG = (
    "", "g", "gg", "gs", "n", "nj", "nh", "d", "l", "lg",
    "lm", "lb", "ls", "lt", "lp", "lh", "m", "b", "bs", "s",
    "ss", "ng", "j", "c", "k", "t", "p", "h"
)


def romanize_korean_universal(text: str) -> str:
    """Universal dependency-free Romanization of Korean hangul and Unicode normalization."""
    result = []
    for char in str(text or ""):
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            cho = offset // 588
            jung = (offset % 588) // 28
            jong = offset % 28
            result.append(_CHOSUNG[cho] + _JUNGSUNG[jung] + _JONGSUNG[jong])
        else:
            result.append(char)
    return "".join(result)


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


_KOREAN_PARTICLES = (
    "부터", "까지", "에게", "한테", "에서", "으로", "로",
    "에는", "에도", "에", "은", "는", "이", "가", "을", "를",
    "와", "과", "도", "만", "의", "등", "및",
    "시스템", "모드", "기능", "구현", "해줘", "만들어줘", "넣어줘",
    "해야해", "하게해줘",
)

_CLAUSE_SPLIT = re.compile(
    r"\s*(?:,|;|\band\b|\bthen\b|그리고|\s및\s|\s다음\s|→|->|=>|/|\||•|\u2022|\u25b6|\u25cf|\u2013|\u2014|\.|\n|\r)\s*",
    re.IGNORECASE | re.UNICODE,
)


def _extract_canonical_tokens(word: str) -> tuple[str, ...]:
    """Extract canonical domain keywords by prioritizing exact match, particle stripping, and compound segmentation."""
    low = word.casefold().strip()
    if not low:
        return ()

    # 1. Exact match
    if low in _CANONICAL_DOMAIN_MAP or low in _THEME_ARCHETYPES or low in _FUNCTIONAL_ARCHETYPES:
        return (low,)

    # 2. Particle / suffix stripping (e.g. 잡몹부터 -> 잡몹, 보스까지 -> 보스, 레벨도 -> 레벨)
    stripped = low
    changed = True
    while changed and len(stripped) >= 2:
        changed = False
        for p in _KOREAN_PARTICLES:
            if stripped.endswith(p) and len(stripped) > len(p):
                stripped = stripped[:-len(p)]
                changed = True
                break
    if stripped in _CANONICAL_DOMAIN_MAP or stripped in _THEME_ARCHETYPES or stripped in _FUNCTIONAL_ARCHETYPES:
        return (stripped,)

    # 3. Compound segmentation and longest-key substring matching
    matched_keys: list[str] = []
    for key in sorted(_CANONICAL_DOMAIN_MAP.keys(), key=lambda k: -len(k)):
        if len(key) >= 2 and key in low:
            if key not in matched_keys:
                matched_keys.append(key)
    for theme_key in sorted(_THEME_ARCHETYPES.keys(), key=lambda k: -len(k)):
        if len(theme_key) >= 2 and theme_key in low:
            if theme_key not in matched_keys:
                matched_keys.append(theme_key)
    if matched_keys:
        return tuple(matched_keys)

    return ()


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

    ignored = {
        "a", "an", "the", "add", "create", "make", "build", "implement", "keep",
        "minecraft", "mod", "with", "to", "for", "that", "and", "or", "all", "every",
        "그리고", "추가", "만들어", "만들기", "구현", "모드", "시스템", "해줘", "넣어줘",
        "모두", "점점", "등", "해야해", "하게해줘", "부터", "까지",
    }

    # Clause-based separation
    raw_clauses = [c.strip() for c in _CLAUSE_SPLIT.split(clean) if c.strip()]
    if not raw_clauses:
        raw_clauses = [clean]

    nodes: list[CapabilityResolutionNode] = []
    seen_caps: set[str] = set()
    unresolved_spans: list[str] = []

    for clause in raw_clauses:
        words = re.findall(r"[A-Za-z0-9_]+|[\u3131-\u318e\uac00-\ud7a3]+", clause)
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
                        if cap not in seen_caps:
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
            raw_slug = romanize_korean_universal(clause_unresolved)
            slug = re.sub(r"[^a-z0-9_]+", "_", raw_slug.casefold()).strip("_")
            slug = re.sub(r"_+", "_", slug)
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

