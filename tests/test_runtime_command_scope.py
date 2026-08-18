from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai.runtime_command_scope import (
    server_command_scope_violation,
    validate_server_command_scope,
)
from minecraft_mod_ai.runtime_manager import MinecraftRuntimeManager, RuntimePolicyError


@pytest.mark.parametrize(
    "command",
    (
        "gametest run mmm:smoke",
        "test run mmm:all",
        "say integration ready",
        "mymod:probe target",
        "mymod:fill target",
        "execute as @e run mymod:probe target",
        "minecraft:execute positioned 0 64 0 run mymod:probe target",
    ),
)
def test_mod_playtest_commands_remain_in_scope(command: str) -> None:
    assert server_command_scope_violation(command) is None
    validate_server_command_scope(command)


@pytest.mark.parametrize(
    "command",
    (
        "fill 0 0 0 1 1 1 stone",
        "/fill 0 0 0 1 1 1 stone",
        "minecraft:fill 0 0 0 1 1 1 stone",
        "fillbiome 0 0 0 1 1 1 minecraft:plains",
        "setblock 0 64 0 minecraft:stone",
        "minecraft:setblock 0 64 0 minecraft:stone",
        "clone 0 0 0 1 1 1 10 10 10",
        "place structure mmm:frost_temple 0 64 0",
        "function mmm:builder",
        "schedule function mmm:builder 1t",
        "//paste",
        "/worldedit:paste",
        "we:paste",
        "execute positioned 0 64 0 run fill 0 0 0 1 1 1 stone",
        "minecraft:execute as @a run minecraft:setblock 0 64 0 minecraft:stone",
        "execute as @a run execute at @s run clone 0 0 0 1 1 1 2 2 2",
    ),
)
def test_direct_or_wrapped_world_edit_commands_are_out_of_scope(command: str) -> None:
    assert server_command_scope_violation(command) is not None
    with pytest.raises(ValueError, match="outside M.M.M's mod playtest scope"):
        validate_server_command_scope(command)


def _write_permissive_profile(path: Path) -> None:
    path.write_text(
        """\
schema_version: mmm/runtime-profiles-v1
profiles:
  permissive_test:
    minecraft_version: '1.21.1'
    server_java_command: java
    server_memory_mb: 2048
    server_launcher_relative: fabric-server-launch.jar
    client_command_env: MMM_TEST_CLIENT_COMMAND
    allowed_server_commands:
      - '.*'
    startup_ready_patterns:
      - 'Done'
    disposable_only: true
    eula_must_be_explicitly_accepted: true
""",
        encoding="utf-8",
    )


def test_runtime_profile_cannot_override_world_edit_scope(tmp_path: Path) -> None:
    config = tmp_path / "runtime_profiles.yaml"
    _write_permissive_profile(config)
    manager = MinecraftRuntimeManager(
        tmp_path / "workspace",
        profile_name="permissive_test",
        config_path=config,
    )

    with pytest.raises(RuntimePolicyError, match="outside M.M.M's mod playtest scope"):
        manager.send_server_command("fill 0 0 0 1 1 1 minecraft:stone")


def test_in_scope_custom_command_still_reaches_runtime_state_check(tmp_path: Path) -> None:
    config = tmp_path / "runtime_profiles.yaml"
    _write_permissive_profile(config)
    manager = MinecraftRuntimeManager(
        tmp_path / "workspace",
        profile_name="permissive_test",
        config_path=config,
    )

    with pytest.raises(RuntimePolicyError, match="Minecraft server is not running"):
        manager.send_server_command("mymod:fill validation_case")
