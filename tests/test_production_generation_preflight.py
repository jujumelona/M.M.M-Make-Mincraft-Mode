from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.complete_spec import CompleteProposal
from minecraft_mod_ai.geckolib_generation_contract import (
    GeckoLibGenerationContractError,
    geckolib_entity_inputs_from_module_config,
)
from minecraft_mod_ai.production_generation_preflight import (
    ProductionGenerationPreflightError,
    validate_production_generation_modules,
    validate_production_generation_project,
)


def _module(
    module_id: str,
    kind: str,
    config: dict[str, object],
) -> SimpleNamespace:
    return SimpleNamespace(
        module_id=module_id,
        kind=kind,
        config=config,
        depends_on=(),
        required_gates=(),
    )


def _module_dict(module: SimpleNamespace) -> dict[str, object]:
    return {
        "module_id": module.module_id,
        "kind": module.kind,
        "config": module.config,
        "depends_on": list(module.depends_on),
        "required_gates": list(module.required_gates),
    }


def _write_system_pack(
    root: Path,
    pack_id: str,
    modules: list[SimpleNamespace],
) -> None:
    path = root / f"src/main/resources/data/example/mmm_systems/{pack_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pack_id": pack_id,
        "module_count": len(modules),
        "modules": [_module_dict(module) for module in modules],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_entity_module_set_is_rejected_before_any_dispatch() -> None:
    modules = (
        _module("valid_entity", "entity", {"max_health": 20}),
        _module("invalid_entity", "entity", {"movement_speed": "not-a-number"}),
    )

    with pytest.raises(
        ProductionGenerationPreflightError,
        match="invalid_entity.*numeric fields cannot be converted",
    ):
        validate_production_generation_modules(modules)


def test_entity_preflight_preserves_orchestrator_coercion_semantics() -> None:
    inputs = geckolib_entity_inputs_from_module_config(
        "npc",
        {
            "texture_width": "128",
            "texture_height": 64.9,
            "max_health": "40",
            "attack_damage": 3,
            "movement_speed": "0.2",
            "follow_range": 24,
            "entity_width": "0.7",
            "entity_height": 1.9,
        },
    )

    assert inputs.texture_width == 128
    assert inputs.texture_height == 64
    assert inputs.max_health == 40.0
    assert inputs.behavior == "npc"
    assert inputs.spawn_group == "creature"

    with pytest.raises(
        GeckoLibGenerationContractError,
        match="positive finite number",
    ):
        geckolib_entity_inputs_from_module_config(
            "entity",
            {"follow_range": "nan"},
        )


def test_system_cross_module_contract_is_rejected_before_generation() -> None:
    modules = (
        _module("mage", "class", {"display_name": "Mage"}),
        _module(
            "blink",
            "skill",
            {
                "required_class": "missing_class",
                "effect": "minecraft:speed",
            },
        ),
    )

    with pytest.raises(
        ProductionGenerationPreflightError,
        match="class-skill-system.*missing class",
    ):
        validate_production_generation_modules(modules)


def test_cross_catalog_duplicates_are_rejected_before_parallel_pack_writes() -> None:
    modules = (
        _module(
            "shop_a",
            "shop",
            {
                "entries": [
                    {"id": "shared", "item": "minecraft:stone", "price": 1}
                ]
            },
        ),
        _module(
            "shop_b",
            "shop",
            {
                "entries": [
                    {"id": "shared", "item": "minecraft:dirt", "price": 2}
                ]
            },
        ),
    )

    with pytest.raises(
        ProductionGenerationPreflightError,
        match="economy-shop.*duplicated across catalogs",
    ):
        validate_production_generation_modules(modules)


def test_imported_system_reference_is_resolved_before_dispatch(tmp_path: Path) -> None:
    existing_class = _module("mage", "class", {"display_name": "Mage"})
    new_skill = _module(
        "blink",
        "skill",
        {
            "required_class": "mage",
            "effect": "minecraft:speed",
        },
    )
    _write_system_pack(
        tmp_path,
        "class-skill-system",
        [existing_class],
    )

    validate_production_generation_modules(
        [new_skill],
        validate_system_packs=False,
    )
    validate_production_generation_project(
        tmp_path,
        [new_skill],
        mod_id="example",
        package_name="com.example",
    )


def test_imported_system_collision_is_rejected_before_dispatch(tmp_path: Path) -> None:
    existing_shop = _module(
        "shop_a",
        "shop",
        {
            "entries": [
                {"id": "shared", "item": "minecraft:stone", "price": 1}
            ]
        },
    )
    new_shop = _module(
        "shop_b",
        "shop",
        {
            "entries": [
                {"id": "shared", "item": "minecraft:dirt", "price": 2}
            ]
        },
    )
    _write_system_pack(tmp_path, "economy-shop", [existing_shop])

    with pytest.raises(
        ProductionGenerationPreflightError,
        match="economy-shop.*duplicated across catalogs",
    ):
        validate_production_generation_project(
            tmp_path,
            [new_shop],
            mod_id="example",
            package_name="com.example",
        )


def test_custom_routed_modules_bypass_builtin_preflight() -> None:
    modules = (
        _module(
            "custom_entity",
            "entity",
            {
                "implementation": "custom",
                "movement_speed": "not-a-number",
                "archetype": "unsupported",
            },
        ),
        _module(
            "custom_skill",
            "skill",
            {
                "implementation": "custom",
                "required_class": "missing_class",
                "effect": "not-namespaced",
            },
        ),
    )

    validate_production_generation_modules(modules)


def test_runtime_owns_proposal_and_dispatch_preflight() -> None:
    assert getattr(
        CompleteProposal.validate,
        "_mmm_production_generation_preflight",
        False,
    )
    assert getattr(
        CompleteProductionOrchestrator._execute_generation_work,
        "_mmm_project_generation_preflight",
        False,
    )
