from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.capabilities import capability_names
from minecraft_mod_ai.capability_plugins import buildable_plugin_ids
from minecraft_mod_ai.mod_development_methods import resolve_mod_development_methods
from minecraft_mod_ai.mod_scope_contract import install
from minecraft_mod_ai.skill_catalog import (
    REVIEWED_TOOL_STAGES,
    compile_skill_contract,
)
from minecraft_mod_ai.spec import SpecValidationError


def test_baseline_mod_methods_exclude_standalone_map() -> None:
    result = resolve_mod_development_methods(
        "간단한 음식 아이템을 추가하는 모드를 만들어줘"
    )

    assert result["standalone_map_generation"] is False
    assert "fabric_project_contract" in result["method_ids"]
    assert "registry_and_datagen" in result["method_ids"]
    assert "content_registry" in result["method_ids"]
    assert "fabric_worldgen" not in result["method_ids"]


def test_mod_method_contract_is_json_native() -> None:
    result = resolve_mod_development_methods(
        "퀘스트와 GUI가 있는 멀티플레이 모드를 만들어줘"
    )

    assert json.loads(json.dumps(result, ensure_ascii=False)) == result
    for method in result["methods"]:
        assert isinstance(method["outputs"], list)
        assert isinstance(method["required_evidence"], list)
        assert isinstance(method["release_gates"], list)


def test_worldgen_is_mod_owned_and_explicit() -> None:
    result = resolve_mod_development_methods(
        "새 바이옴과 구조물을 추가하는 Fabric 모드를 만들어줘"
    )

    assert "fabric_worldgen" in result["method_ids"]
    assert result["standalone_map_generation"] is False


def test_requested_arena_selects_mod_worldgen() -> None:
    result = resolve_mod_development_methods(
        "보스와 함께 생성되는 아레나를 추가하는 모드"
    )

    assert "entity_rendering_animation" in result["method_ids"]
    assert "fabric_worldgen" in result["method_ids"]


def test_standalone_map_request_is_flagged_not_selected() -> None:
    result = resolve_mod_development_methods(
        "월드 ZIP과 litematic 맵 파일을 만들어줘"
    )

    assert result["standalone_map_requested"] is True
    assert result["standalone_map_generation"] is False
    assert "fabric_worldgen" not in result["method_ids"]


def test_capabilities_have_no_map_compiler() -> None:
    names = capability_names()

    assert "world.ir.generate" not in names
    assert "world.compile" not in names
    assert "fabric.worldgen.generate" in names


def test_plugins_have_no_standalone_map_builder() -> None:
    plugin_ids = buildable_plugin_ids()

    assert "worldgen-map" not in plugin_ids
    assert "worldgen-arena" not in plugin_ids
    assert "fabric-worldgen" in plugin_ids
    assert "mod-development-methods" in plugin_ids


def test_skill_policy_has_no_standalone_map_tools() -> None:
    assert "generate_world_ir" not in REVIEWED_TOOL_STAGES
    assert "compile_world_ir" not in REVIEWED_TOOL_STAGES

    plan = compile_skill_contract("plan-game-design")
    worldgen = compile_skill_contract("generate-worldgen")

    assert "generate_world_ir" not in plan.allowed_tools
    assert "compile_world_ir" not in worldgen.allowed_tools
    assert "generate_fabric_project" in worldgen.allowed_tools
    assert "run_gametest" in worldgen.allowed_tools


def _fake_contract_modules():
    class FakeCompleteSpec:
        @staticmethod
        def complete_proposal_from_parts(**kwargs):
            return kwargs

    class FakeCompletePlanner:
        @staticmethod
        def _implementation_prompt(prompt, game_design):
            return f"{prompt}:{sorted(game_design)}"

        complete_proposal_from_parts = (
            FakeCompleteSpec.complete_proposal_from_parts
        )

    install(FakeCompleteSpec, FakeCompletePlanner)
    return FakeCompleteSpec, FakeCompletePlanner


def test_complete_scope_prompt_does_not_mutate_caller_design() -> None:
    _, fake_planner = _fake_contract_modules()
    design = {"title": "ZIP", "payload": "z" * 50_000}
    original = dict(design)

    rendered = fake_planner._implementation_prompt(
        "음식 아이템 모드를 만들어줘",
        design,
    )

    assert design == original
    assert "_mod_development_methods" in rendered


def test_complete_scope_attaches_methods_to_proposal() -> None:
    fake_spec, _ = _fake_contract_modules()
    item_module = SimpleNamespace(
        module_id="food_registry",
        kind="food",
        config={},
    )

    result = fake_spec.complete_proposal_from_parts(
        requested_prompt="음식 아이템 모드를 만들어줘",
        base_proposal=SimpleNamespace(arena=None),
        game_design={"title": "food"},
        modules=(item_module,),
        acceptance_tests=("food can be crafted",),
    )

    methods = result["game_design"]["_mod_development_methods"]
    assert methods["standalone_map_generation"] is False
    assert "content_registry" in methods["method_ids"]
    assert result["game_design"]["_product_scope"][
        "standalone_map_generation"
    ] is False


def test_complete_scope_rejects_unrequested_worldgen() -> None:
    fake_spec, _ = _fake_contract_modules()
    structure_module = SimpleNamespace(
        module_id="unexpected_structure",
        kind="structure",
        config={},
    )

    with pytest.raises(SpecValidationError):
        fake_spec.complete_proposal_from_parts(
            requested_prompt="음식 아이템 모드를 만들어줘",
            base_proposal=SimpleNamespace(arena=None),
            game_design={"title": "food"},
            modules=(structure_module,),
            acceptance_tests=("food can be crafted",),
        )


def test_complete_scope_rejects_standalone_map_request() -> None:
    fake_spec, _ = _fake_contract_modules()

    with pytest.raises(SpecValidationError):
        fake_spec.complete_proposal_from_parts(
            requested_prompt="월드 ZIP 맵 파일을 만들어줘",
            base_proposal=SimpleNamespace(arena=None),
            game_design={"title": "map"},
            modules=(),
            acceptance_tests=("map exists",),
        )
