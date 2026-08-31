from dataclasses import replace
from pathlib import Path

import pytest

from minecraft_mod_ai import platform_specialized_generator_contract as specialized
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.platform_catalog import adapter_from_project
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
from minecraft_mod_ai.system_pack_generator import generate_system_pack
from minecraft_mod_ai.system_pack_validation import validate_system_modules


def _project(root: Path) -> Path:
    spec = ModSpec(
        mod_id="testmod",
        mod_name="Test Mod",
        package_name="ai.minecraft.testmod",
        version="1.0.0",
        summary="test",
        contents=(
            ContentSpec(
                content_id="test_item",
                kind=ContentKind.ITEM,
                display_name_en="Test Item",
                display_name_ko="테스트 아이템",
            ),
        ),
    )
    FabricProjectGenerator().generate(spec, root)
    return root


def _approve_specialized(
    monkeypatch: pytest.MonkeyPatch,
    project: Path,
    *capabilities: str,
) -> None:
    adapter = adapter_from_project(project)
    approved = replace(
        adapter,
        deterministic_module_kinds=frozenset(
            {*adapter.deterministic_module_kinds, *capabilities}
        ),
    )
    monkeypatch.setattr(specialized, "adapter_from_project", lambda _root: approved)


def _system_module(module_id: str, kind: str, config: dict) -> dict:
    return {
        "module_id": module_id,
        "kind": kind,
        "config": config,
        "depends_on": [],
        "required_gates": [],
    }


def test_single_economy_manager_scales_shops_but_routes_extra_currency() -> None:
    modules = [
        _system_module("credits", "economy", {"initial_balance": 0}),
        *[
            _system_module(
                f"shop_{index:04d}",
                "shop",
                {
                    "entries": [
                        {
                            "id": f"entry_{index:04d}",
                            "item": "minecraft:stone",
                            "price": 1,
                        }
                    ]
                },
            )
            for index in range(512)
        ],
    ]
    validate_system_modules("economy-shop", modules)

    with pytest.raises(ValueError, match="explicit instance namespace"):
        validate_system_modules(
            "economy-shop",
            [modules[0], _system_module("tokens", "economy", {"initial_balance": 0})],
        )


def test_single_party_manager_has_unbounded_groups_but_routes_extra_manager() -> None:
    with pytest.raises(ValueError, match="any number of runtime party groups"):
        validate_system_modules(
            "party-guild",
            [
                _system_module(
                    "adventure_parties",
                    "party",
                    {"display_name": "Adventure Party"},
                ),
                _system_module(
                    "raid_parties",
                    "party",
                    {"display_name": "Raid Party"},
                ),
            ],
        )


def test_system_pack_generates_real_fabric_binding_and_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    _approve_specialized(monkeypatch, project, "quest", "system-pack:quest-system")
    result = generate_system_pack(
        project_root=project,
        pack_id="quest-system",
        mod_id="testmod",
        package_name="ai.minecraft.testmod",
        config={
            "modules": [
                {
                    "module_id": "start",
                    "kind": "quest",
                    "config": {"objective": "manual", "required": 1},
                    "depends_on": [],
                    "required_gates": [],
                }
            ]
        },
    )
    assert result["status"] == "fabric_binding_generated"
    paths = [Path(path) for path in result["files"]]
    assert all(path.is_file() for path in paths)
    quest_java = next(path for path in paths if path.name == "QuestSystem.java")
    text = quest_java.read_text(encoding="utf-8")
    assert "CommandRegistrationCallback.EVENT.register" in text
    assert "MmmPersistentStore" in text
    assert 'if (!definition.id().equals(target)) continue;' in text
    main = project / "src/main/java/ai/minecraft/testmod/TestmodMod.java"
    main_text = main.read_text(encoding="utf-8")
    assert "QuestSystem.register();" in main_text


def test_party_pack_generates_persistent_owner_controlled_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "party-project")
    _approve_specialized(
        monkeypatch,
        project,
        "party",
        "guild",
        "system-pack:party-guild",
    )
    result = generate_system_pack(
        project_root=project,
        pack_id="party-guild",
        mod_id="testmod",
        package_name="ai.minecraft.testmod",
        config={
            "modules": [
                {
                    "module_id": "main_party",
                    "kind": "party",
                    "config": {"display_name": "Party"},
                    "depends_on": [],
                    "required_gates": [],
                }
            ]
        },
    )
    party_java = next(
        Path(path)
        for path in result["files"]
        if Path(path).name == "PartyGuildSystem.java"
    )
    text = party_java.read_text(encoding="utf-8")
    assert 'registerGroup("mmmparty", "parties")' in text
    assert 'registerGroup("mmmguild", "guilds")' in text
    assert 'if (guildEnabled) registerGroup("mmmguild", "guilds");' in text
    assert 'CommandManager.literal("kick")' in text
    assert 'CommandManager.literal("disband")' in text
    assert "ServerPlayConnectionEvents.DISCONNECT" not in text
