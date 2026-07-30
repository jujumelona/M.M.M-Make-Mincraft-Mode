from pathlib import Path

from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
from minecraft_mod_ai.system_pack_generator import generate_system_pack


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


def test_system_pack_generates_real_fabric_binding_and_contract(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
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
                    "config": {
                        "objective": "manual",
                        "required": 1,
                    },
                    "depends_on": [],
                    "required_gates": [],
                }
            ]
        },
    )
    assert result["status"] == "fabric_binding_generated"
    paths = [Path(path) for path in result["files"]]
    assert all(path.is_file() for path in paths)
    quest_java = next(
        path for path in paths if path.name == "QuestSystem.java"
    )
    text = quest_java.read_text(encoding="utf-8")
    assert "CommandRegistrationCallback.EVENT.register" in text
    assert "MmmPersistentStore" in text
    assert 'if (!definition.id().equals(target)) continue;' in text
    main = project / "src/main/java/ai/minecraft/testmod/TestmodMod.java"
    main_text = main.read_text(encoding="utf-8")
    assert "QuestSystem.register();" in main_text


def test_party_pack_generates_persistent_owner_controlled_lifecycle(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "party-project")
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
