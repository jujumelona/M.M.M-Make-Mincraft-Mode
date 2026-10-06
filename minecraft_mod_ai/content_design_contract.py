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
SUPPORTED_CONTENT_FACT_TYPES = tuple(CONTENT_KIND_TO_FACT_TYPE.values())

# Concerns describe obligations of content, not the identity of that content.
# For example, displayed state can belong to a block, item, entity, or screen.
# Narrowing it to GUI forces discovery to invent a screen or misclassify its owner.
# The authored requirement and existing entity identities determine the kind.
CONTENT_CONCERN_KINDS = MappingProxyType({
    "registries": CONTENT_KINDS,
    "data_resources": CONTENT_KINDS,
    "assets": CONTENT_KINDS,
    "interactions": CONTENT_KINDS,
    "displayed_state": CONTENT_KINDS,
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
            target in {FactType.ITEM_EXISTS, FactType.BLOCK_EXISTS}
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
    "CONTENT_KINDS",
    "SUPPORTED_CONTENT_FACT_TYPES",
    "fact_type_for_content_kind",
    "relation_type_supported_for_content_pair",
]
