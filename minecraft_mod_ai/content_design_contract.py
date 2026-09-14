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


def fact_type_for_content_kind(kind: str) -> FactType:
    try:
        return CONTENT_KIND_TO_FACT_TYPE[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported content kind: {kind!r}") from exc


__all__ = [
    "CONTENT_KIND_TO_FACT_TYPE",
    "CONTENT_KINDS",
    "SUPPORTED_CONTENT_FACT_TYPES",
    "fact_type_for_content_kind",
]
