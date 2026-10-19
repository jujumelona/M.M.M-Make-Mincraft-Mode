from pathlib import Path

from minecraft_mod_ai.artifact_expansion import (
    FACT_TO_CANONICAL_LEAVES,
)
from minecraft_mod_ai.prompt_fact_types import FactType


def test_artifact_and_generator_routes_are_explicit():
    assert FACT_TO_CANONICAL_LEAVES[FactType.ITEM_EXISTS][0] == "minecraft/item/registry"
    assert FACT_TO_CANONICAL_LEAVES[FactType.CRAFTING_RECIPE] == ("minecraft/recipe/serializer",)
    assert FACT_TO_CANONICAL_LEAVES[FactType.ENTITY_EXISTS] == (
        "minecraft/entity/type", "minecraft/entity/registry",
    )


def test_geckolib_has_no_magic_blue_entity_texture():
    source = Path("minecraft_mod_ai/geckolib_generator.py").read_text(encoding="utf-8")
    assert "#5ba6d8" not in source
    assert "texture_color must be an explicit #RRGGBB entity design value" in source


def test_entity_orchestrator_requires_atomic_semantics():
    from minecraft_mod_ai.geckolib_generation_contract import (
        GeckoLibGenerationContractError,
        geckolib_entity_inputs_from_module_config,
    )
    import pytest

    source = Path("minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")
    assert "geckolib_entity_inputs_from_module_config(" in source
    with pytest.raises(GeckoLibGenerationContractError, match="explicit main_color"):
        geckolib_entity_inputs_from_module_config("entity", {})
    with pytest.raises(GeckoLibGenerationContractError, match="positive finite"):
        geckolib_entity_inputs_from_module_config(
            "entity", {"main_color": "#AA55CC", "max_health": 0}
        )
