from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.geckolib_generator import _entity_java
from minecraft_mod_ai.system_templates_quest import _quest_java


ROOT = Path(__file__).resolve().parents[1]


def test_geckolib_attack_controller_returns_play_state() -> None:
    source = _entity_java(
        "ai.minecraft.generated.example",
        "example",
        "guard",
        "GuardEntity",
        40.0,
        5.0,
        0.25,
        24.0,
        "hostile_melee",
    )

    assert "import software.bernie.geckolib.core.object.PlayState;" in source
    assert "PlayState.STOP" in source
    assert "state.stop()" not in source


def test_quest_block_break_requires_server_player() -> None:
    source = _quest_java(
        "ai.minecraft.generated.example",
        "GeneratedQuestSystem",
        "example:systems/quests.json",
    )

    assert "player instanceof ServerPlayerEntity serverPlayer" in source
    assert "progress(\n                    serverPlayer," in source
    assert "progress(\n                player," not in source


def test_machine_ticker_uses_yarn_1201_check_type_helper() -> None:
    source = (
        ROOT / "minecraft_mod_ai/extended_content_generator.py"
    ).read_text(encoding="utf-8")

    assert "return checkType(" in source
    assert "validateTicker(" not in source
