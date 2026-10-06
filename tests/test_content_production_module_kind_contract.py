from __future__ import annotations

from minecraft_mod_ai.complete_spec import MODULE_KINDS
from minecraft_mod_ai.content_design_contract import (
    CONTENT_FACT_TO_PRODUCTION_KIND,
    CONTENT_KIND_TO_FACT_TYPE,
)
from minecraft_mod_ai.prompt_fact_types import FactType


def test_all_content_facts_lower_to_supported_production_module_kinds() -> None:
    assert set(CONTENT_KIND_TO_FACT_TYPE.values()) <= set(
        CONTENT_FACT_TO_PRODUCTION_KIND
    )
    assert set(CONTENT_FACT_TO_PRODUCTION_KIND.values()) <= set(MODULE_KINDS)


def test_artifact_only_semantics_use_integration_owner_kind() -> None:
    for fact_type in (
        FactType.DATA_COMPONENT,
        FactType.WORLDGEN_FEATURE,
        FactType.DIMENSION,
        FactType.BIOME,
        FactType.SOUND_EVENT,
        FactType.PARTICLE_TYPE,
    ):
        assert CONTENT_FACT_TO_PRODUCTION_KIND[fact_type] == "integration"


def test_resource_facts_keep_supported_native_owner_kinds() -> None:
    assert CONTENT_FACT_TO_PRODUCTION_KIND[FactType.CRAFTING_RECIPE] == "recipe"
    assert CONTENT_FACT_TO_PRODUCTION_KIND[FactType.SMELTING_RECIPE] == "recipe"
    assert CONTENT_FACT_TO_PRODUCTION_KIND[FactType.REGISTRY_TAG] == "tag"
