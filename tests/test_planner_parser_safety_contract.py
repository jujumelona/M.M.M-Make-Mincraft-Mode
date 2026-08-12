from __future__ import annotations

import pytest

from minecraft_mod_ai import complete_planner as planner
from minecraft_mod_ai.planner_module_identity_contract import install as install_identity
from minecraft_mod_ai.planner_parser_safety_contract import install
from minecraft_mod_ai.spec import SpecValidationError


def test_module_does_not_silently_drop_invalid_dependency_shape() -> None:
    install(planner)
    with pytest.raises(SpecValidationError, match="depends_on must be a list"):
        planner._module(
            {
                "module_id": "core_module",
                "kind": "custom_java",
                "config": {},
                "depends_on": "other_module",
                "required_gates": [],
            }
        )


def test_module_identity_is_not_silently_normalized() -> None:
    install(planner)
    install_identity(planner)
    with pytest.raises(SpecValidationError, match="already be lowercase snake_case"):
        planner._module(
            {
                "module_id": "Boss System",
                "kind": "custom_java",
                "config": {},
                "depends_on": [],
                "required_gates": [],
            }
        )


def test_dependency_identity_is_not_silently_normalized() -> None:
    install(planner)
    install_identity(planner)
    with pytest.raises(SpecValidationError, match="invalid dependency ids"):
        planner._module(
            {
                "module_id": "boss_system",
                "kind": "custom_java",
                "config": {},
                "depends_on": ["Core System"],
                "required_gates": [],
            }
        )


def test_asset_does_not_invent_invalid_texture_kind() -> None:
    install(planner)
    with pytest.raises(SpecValidationError, match="Unsupported asset kind"):
        planner._asset(
            {
                "asset_id": "boss_texture",
                "kind": "texture",
                "prompt": "boss texture",
                "target_path": "assets/test/textures/entity/boss.png",
            }
        )


def test_audio_string_false_is_not_coerced_to_true() -> None:
    install(planner)
    with pytest.raises(SpecValidationError, match="loop must be boolean"):
        planner._audio(
            {
                "sound_id": "boss_hit",
                "kind": "effect",
                "duration_seconds": 1.0,
                "loop": "false",
            }
        )


def test_batch_does_not_fabricate_missing_deliverable() -> None:
    install(planner)
    with pytest.raises(SpecValidationError, match="at least one deliverable"):
        planner._production_batch(
            {
                "batch_id": "core",
                "scope": "core gameplay",
                "depends_on_batches": [],
                "deliverables": [],
                "exports": [],
            }
        )


def test_unknown_batch_dependency_is_not_fuzzy_rewritten_or_dropped() -> None:
    install(planner)
    first = planner._ProductionBatch(
        batch_id="core_system",
        scope="core",
        depends_on_batches=(),
        deliverables=("core",),
        exports=(),
    )
    second = planner._ProductionBatch(
        batch_id="boss_system",
        scope="boss",
        depends_on_batches=("core",),
        deliverables=("boss",),
        exports=(),
    )
    with pytest.raises(SpecValidationError, match="unknown dependencies"):
        planner._topological_production_batches((first, second))
