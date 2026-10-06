"""Authoritative model/host contract for design content capabilities."""

from __future__ import annotations

from types import MappingProxyType

from .prompt_fact_types import FactType


# This is the single semantic bridge between model-authored content kinds and the
# atomic FactType consumed by content lowering. Model schemas are bound from this
# table at generation time; host validation consumes the same values.
CONTENT_KIND_TO_FACT_TYPE = MappingProxyType(
    {
        "item": FactType.ITEM_EXISTS,
        "block": FactType.BLOCK_EXISTS,
        "entity": FactType.ENTITY_EXISTS,
        "gui": FactType.GUI_EXISTS,
        "network_packet": FactType.NETWORK_PACKET,
        "block_entity": FactType.BLOCK_ENTITY_EXISTS,
        "data_component": FactType.DATA_COMPONENT,
        "worldgen_feature": FactType.WORLDGEN_FEATURE,
        "dimension": FactType.DIMENSION,
        "biome": FactType.BIOME,
        "status_effect": FactType.STATUS_EFFECT,
        "sound_event": FactType.SOUND_EVENT,
        "particle_type": FactType.PARTICLE_TYPE,
        "entity_loot": FactType.ENTITY_LOOT,
        "advancement": FactType.ADVANCEMENT,
        "equipment_armor": FactType.EQUIPMENT_ARMOR,
        "custom_item_behavior": FactType.CUSTOM_ITEM_BEHAVIOR,
        "custom_block_behavior": FactType.CUSTOM_BLOCK_BEHAVIOR,
        "crafting_recipe": FactType.CRAFTING_RECIPE,
        "smelting_recipe": FactType.SMELTING_RECIPE,
        "registry_tag": FactType.REGISTRY_TAG,
    }
)

CONTENT_KINDS = tuple(CONTENT_KIND_TO_FACT_TYPE)

# Resource definitions are graph-level structures, not primary content identities.
# The small model must never invent them during ordinary content-owner discovery.
# They are materialized deterministically from authored data-resource records only
# after concrete target entities already exist.
RESOURCE_DEFINITION_KINDS = (
    "crafting_recipe",
    "smelting_recipe",
    "registry_tag",
)
PRIMARY_CONTENT_KINDS = tuple(
    kind for kind in CONTENT_KINDS
    if kind not in RESOURCE_DEFINITION_KINDS
)
SUPPORTED_CONTENT_FACT_TYPES = tuple(CONTENT_KIND_TO_FACT_TYPE.values())


def resource_definition_kind_for_data_resource(value: str) -> str | None:
    normalized = "_".join(
        part
        for part in "".join(
            char.lower() if char.isalnum() else "_"
            for char in str(value or "")
        ).split("_")
        if part
    )
    aliases = {
        "tag": "registry_tag",
        "tags": "registry_tag",
        "registry_tag": "registry_tag",
        "item_tag": "registry_tag",
        "block_tag": "registry_tag",
        "entity_tag": "registry_tag",
        "entity_type_tag": "registry_tag",
        "recipe": "crafting_recipe",
        "crafting": "crafting_recipe",
        "crafting_recipe": "crafting_recipe",
        "shaped_recipe": "crafting_recipe",
        "shapeless_recipe": "crafting_recipe",
        "smelting": "smelting_recipe",
        "smelt": "smelting_recipe",
        "smelting_recipe": "smelting_recipe",
        "blasting": "smelting_recipe",
        "blasting_recipe": "smelting_recipe",
        "cooking_recipe": "smelting_recipe",
    }
    return aliases.get(normalized)

# ProductionModule.kind is an execution-owner classification, not a second copy
# of the semantic FactType vocabulary. Facts implemented exclusively through the
# canonical artifact graph use the generic integration owner so they cannot leak
# non-executable semantic labels (for example "structure") into CompleteProposal.
CONTENT_FACT_TO_PRODUCTION_KIND = MappingProxyType(
    {
        FactType.ITEM_EXISTS: "item",
        FactType.BLOCK_EXISTS: "block",
        FactType.ENTITY_EXISTS: "entity",
        FactType.GUI_EXISTS: "gui",
        FactType.NETWORK_PACKET: "networking",
        FactType.BLOCK_ENTITY_EXISTS: "block_entity",
        FactType.DATA_COMPONENT: "integration",
        FactType.WORLDGEN_FEATURE: "integration",
        FactType.DIMENSION: "integration",
        FactType.BIOME: "integration",
        FactType.STATUS_EFFECT: "effect",
        FactType.SOUND_EVENT: "integration",
        FactType.PARTICLE_TYPE: "integration",
        FactType.ENTITY_LOOT: "loot",
        FactType.ADVANCEMENT: "advancement",
        FactType.EQUIPMENT_ARMOR: "armor",
        FactType.CUSTOM_ITEM_BEHAVIOR: "item",
        FactType.CUSTOM_BLOCK_BEHAVIOR: "block",
        FactType.CRAFTING_RECIPE: "recipe",
        FactType.SMELTING_RECIPE: "recipe",
        FactType.REGISTRY_TAG: "tag",
    }
)


REGISTRY_TAG_KIND_TO_TARGET_FACT_TYPE = MappingProxyType(
    {
        "item": FactType.ITEM_EXISTS,
        "block": FactType.BLOCK_EXISTS,
        "entity_type": FactType.ENTITY_EXISTS,
    }
)

# Finite semantic property vocabularies belong to the host contract. The model
# chooses meaning only inside these closed sets; lowering must never discover a
# new categorical value after generation.
CONTENT_PROPERTY_VALUE_ENUMS = MappingProxyType(
    {
        "category": ("monster", "creature", "ambient", "water_creature", "misc"),
        "archetype": ("biped", "quadruped", "flying", "serpentine", "construct"),
        "behavior": ("hostile_melee", "neutral_melee", "passive", "npc"),
        "recipe_kind": ("shaped", "shapeless"),
        "cooking_type": ("smelting", "blasting"),
        "registry_kind": tuple(REGISTRY_TAG_KIND_TO_TARGET_FACT_TYPE),
        "count": tuple(str(value) for value in range(1, 65)),
        "stack_limit": tuple(str(value) for value in range(1, 65)),
    }
)

CONTENT_PROPERTY_VALUE_PATTERNS = MappingProxyType(
    {
        "main_color": r"^#[0-9A-Fa-f]{6}$",
        "health": r"^(?:[1-9][0-9]*(?:\.[0-9]+)?|0\.[0-9]*[1-9][0-9]*)$",
        "speed": r"^(?:[1-9][0-9]*(?:\.[0-9]+)?|0\.[0-9]*[1-9][0-9]*)$",
        "tracking_range": r"^(?:[1-9][0-9]*(?:\.[0-9]+)?|0\.[0-9]*[1-9][0-9]*)$",
        "width": r"^(?:[1-9][0-9]*(?:\.[0-9]+)?|0\.[0-9]*[1-9][0-9]*)$",
        "height": r"^(?:[1-9][0-9]*(?:\.[0-9]+)?|0\.[0-9]*[1-9][0-9]*)$",
        "attack_damage": r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$",
        "experience": r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$",
        "cookingtime": r"^[1-9][0-9]*$",
        "container_size": r"^[1-9][0-9]*$",
        "defense": r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$",
        "toughness": r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$",
        "cooldown": r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$",
    }
)

# Concerns describe obligations of content, not the identity of that content.
# For example, displayed state can belong to a block, item, entity, or screen.
# Narrowing it to GUI forces discovery to invent a screen or misclassify its owner.
# The authored requirement and existing entity identities determine the kind.
CONTENT_CONCERN_KINDS = MappingProxyType({
    "registries": PRIMARY_CONTENT_KINDS,
    "data_resources": PRIMARY_CONTENT_KINDS,
    "assets": PRIMARY_CONTENT_KINDS,
    "interactions": PRIMARY_CONTENT_KINDS,
    "displayed_state": PRIMARY_CONTENT_KINDS,
})


def fact_type_for_content_kind(kind: str) -> FactType:
    try:
        return CONTENT_KIND_TO_FACT_TYPE[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported content kind: {kind!r}") from exc



def relation_type_supported_for_content_pair(
    relation_type: str,
    source_kind: str,
    target_kind: str,
) -> bool:
    """Return whether lowering can represent this relation for the fixed kind pair."""

    source = fact_type_for_content_kind(source_kind)
    target = fact_type_for_content_kind(target_kind)
    key_relation = (
        relation_type.startswith("key_")
        and len(relation_type) == 5
        and relation_type[-1].isalnum()
    )

    # Resource-definition nodes have closed relation vocabularies. They are
    # lowered by dedicated recipe/tag code and must never inherit the generic
    # gameplay relations accepted for ordinary content nodes.
    if source == FactType.CRAFTING_RECIPE:
        return target == FactType.ITEM_EXISTS and (
            relation_type in {"consumes", "produces"} or key_relation
        )
    if source == FactType.SMELTING_RECIPE:
        return target == FactType.ITEM_EXISTS and relation_type in {
            "consumes",
            "produces",
        }
    if source == FactType.REGISTRY_TAG:
        return (
            target in set(REGISTRY_TAG_KIND_TO_TARGET_FACT_TYPE.values())
            and relation_type == "contains"
        )

    # These relation families are reserved for the dedicated sources above.
    if relation_type in {"consumes", "produces", "contains"} or key_relation:
        return False

    if relation_type in {"requires", "upgrades", "unlocks"}:
        return True
    if relation_type == "drops":
        return (
            source == FactType.BLOCK_EXISTS and target == FactType.ITEM_EXISTS
        ) or (
            source == FactType.ENTITY_EXISTS
            and target in {FactType.ITEM_EXISTS, FactType.ENTITY_LOOT}
        )
    if relation_type == "opens":
        return target == FactType.GUI_EXISTS
    if relation_type == "controls":
        return target in {FactType.BLOCK_ENTITY_EXISTS, FactType.ENTITY_EXISTS}
    if relation_type == "spawns":
        return target == FactType.ENTITY_EXISTS
    if relation_type == "transports_to":
        return target in {FactType.DIMENSION, FactType.BIOME}
    if relation_type == "displays":
        return source == FactType.GUI_EXISTS
    if relation_type == "synchronizes":
        return target == FactType.NETWORK_PACKET
    return False


__all__ = [
    "CONTENT_CONCERN_KINDS",
    "CONTENT_KIND_TO_FACT_TYPE",
    "CONTENT_FACT_TO_PRODUCTION_KIND",
    "CONTENT_PROPERTY_VALUE_ENUMS",
    "CONTENT_PROPERTY_VALUE_PATTERNS",
    "CONTENT_KINDS",
    "PRIMARY_CONTENT_KINDS",
    "RESOURCE_DEFINITION_KINDS",
    "SUPPORTED_CONTENT_FACT_TYPES",
    "REGISTRY_TAG_KIND_TO_TARGET_FACT_TYPE",
    "fact_type_for_content_kind",
    "relation_type_supported_for_content_pair",
    "resource_definition_kind_for_data_resource",
]
