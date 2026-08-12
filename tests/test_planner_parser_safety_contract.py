from __future__ import annotations

import pytest

from minecraft_mod_ai import complete_planner as planner
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
