"""Regression for Mojang-named, object-payload typed host generation.

Changed 2026-10-08 to prevent Fabric 26.1 from regressing to the
pre-26.1 Yarn networking/persistence API.
"""
from __future__ import annotations

import pytest

from minecraft_mod_ai.typed_network_mojang import render_mojang_network_policy_files
from minecraft_mod_ai.system_templates_common import _persistent_store_java


def test_mojang_network_registers_object_payload_and_bounds_length():
    files = render_mojang_network_policy_files(
        package_name="demo.mod", mod_id="testmod",
        field_types={"score": "int", "level": "string"},
        interval=20, max_bytes=4096,
    )
    assert len(files) == 3
    server = files["src/main/java/demo/mod/AuthoredNetworkSync.java"]
    client = files["src/client/java/demo/mod/AuthoredNetworkClient.java"]
    assert "src/main/java/demo/mod/AuthoredNetworkClient.java" not in files
    payload = files["src/main/java/demo/mod/AuthoredStateSyncPayload.java"]
    assert "PayloadTypeRegistry.clientboundPlay().register(" in server
    assert "ServerPlayNetworking.send(player, new AuthoredStateSyncPayload(encoded))" in server
    assert "ClientPlayNetworking.registerGlobalReceiver(" in client
    assert "payload.json()" in client
    assert "StandardCharsets.UTF_8" in client
    assert "implements CustomPacketPayload" in payload
    assert "StreamCodec.composite" in payload
    assert 'testmod:typed_state_sync' in payload
    for source in (server, client, payload):
        assert "net.minecraft.server.network.ServerPlayerEntity" not in source
        assert "net.minecraft.util.Identifier" not in source
        assert "PacketByteBufs" not in source


def test_mojang_network_rejects_unknown_type_and_invalid_bounds():
    with pytest.raises(ValueError, match="TYPED_NETWORK_STATE_TYPE_INVALID"):
        render_mojang_network_policy_files(
            package_name="demo.mod", mod_id="demo",
            field_types={"x": "unknown"}, interval=20, max_bytes=4096,
        )
    with pytest.raises(ValueError, match="TYPED_MOJANG_NETWORK_CONFIG_INVALID"):
        render_mojang_network_policy_files(
            package_name="demo.mod", mod_id="demo",
            field_types={"x": "int"}, interval=0, max_bytes=4096,
        )


def test_persistent_store_keeps_old_target_and_new_target_separate():
    legacy = _persistent_store_java("demo.mod", "demo", minecraft_version="1.21.5")
    modern = _persistent_store_java("demo.mod", "demo", minecraft_version="26.1.2")
    assert "net.minecraft.util.WorldSavePath" in legacy
    assert "server.getSavePath(WorldSavePath.ROOT)" in legacy
    assert "net.minecraft.world.level.storage.LevelResource" in modern
    assert "server.getWorldPath(LevelResource.ROOT)" in modern
    assert "WorldSavePath" not in modern



def test_mojang_command_binding_does_not_use_yarn_literal_or_integer_permission():
    from minecraft_mod_ai.typed_plan_java import _Renderer

    event = {
        "event": "command",
        "function": "run",
        "config": {"literal": "debug", "permission_level": 2},
    }
    modern = "\n".join(
        _Renderer({}, {}, minecraft_version="26.1.2").event_registration(event)
    )
    legacy = "\n".join(
        _Renderer({}, {}, minecraft_version="1.21.5").event_registration(event)
    )
    assert "net.minecraft.commands.Commands.literal(" in modern
    assert "Commands.hasPermission(" in modern
    assert "Commands.LEVEL_MODERATORS" in modern
    assert "net.minecraft.server.command.CommandManager" not in modern
    assert "hasPermissionLevel(" not in modern
    assert "net.minecraft.server.command.CommandManager.literal(" in legacy
    assert "hasPermissionLevel(2)" in legacy


def test_mojang_network_retirement_only_deletes_owned_stale_main_source(tmp_path):
    from minecraft_mod_ai.typed_plan_production import (
        _retire_host_owned_legacy_network_client,
    )

    package = "demo.mod"
    old = tmp_path / "src/main/java/demo/mod/AuthoredNetworkClient.java"
    new = tmp_path / "src/client/java/demo/mod/AuthoredNetworkClient.java"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text(
        "package demo.mod;\\n// MMM:TYPED_NETWORK_CLIENT_OWNER\\n",
        encoding="utf-8",
    )
    new.write_text(
        "package demo.mod;\\n// MMM:TYPED_NETWORK_CLIENT_OWNER\\n",
        encoding="utf-8",
    )
    retired = _retire_host_owned_legacy_network_client(
        tmp_path, package, "26.2",
    )
    assert retired == "src/main/java/demo/mod/AuthoredNetworkClient.java"
    assert not old.exists()
    assert new.exists()
    assert _retire_host_owned_legacy_network_client(tmp_path, package, "26.2") is None


def test_mojang_network_retirement_refuses_to_delete_user_source(tmp_path):
    from minecraft_mod_ai.typed_plan_production import (
        _retire_host_owned_legacy_network_client,
    )

    old = tmp_path / "src/main/java/demo/mod/AuthoredNetworkClient.java"
    new = tmp_path / "src/client/java/demo/mod/AuthoredNetworkClient.java"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text("package demo.mod; // user-owned", encoding="utf-8")
    new.write_text("// MMM:TYPED_NETWORK_CLIENT_OWNER", encoding="utf-8")
    with pytest.raises(ValueError, match="TYPED_PLAN_OWNERSHIP_CONFLICT"):
        _retire_host_owned_legacy_network_client(tmp_path, "demo.mod", "26.2")
    assert old.exists()
