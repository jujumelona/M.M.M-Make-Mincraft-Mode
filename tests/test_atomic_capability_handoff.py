from pathlib import Path

from minecraft_mod_ai.artifact_expansion import (
    DECLARED_GENERATOR_HANDOFFS,
    implementation_route,
)
from minecraft_mod_ai.prompt_fact_types import FactType


def test_artifact_and_generator_routes_are_explicit():
    assert implementation_route(FactType.ITEM_EXISTS) == "artifact"
    assert implementation_route(FactType.CRAFTING_RECIPE) == "artifact"
    assert implementation_route(FactType.ENTITY_EXISTS) == "generator"
    assert FactType.CONTENT_RELATION in DECLARED_GENERATOR_HANDOFFS


def test_geckolib_has_no_magic_blue_entity_texture():
    source = Path("minecraft_mod_ai/geckolib_generator.py").read_text(encoding="utf-8")
    assert "#5ba6d8" not in source
    assert "texture_color must be an explicit #RRGGBB entity design value" in source


def test_entity_orchestrator_requires_atomic_semantics():
    source = Path("minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")
    assert "ENTITY_DESIGN_UNRESOLVED" in source
    assert 'max_health=float(config["max_health"])' in source
    assert 'movement_speed=float(config["movement_speed"])' in source
