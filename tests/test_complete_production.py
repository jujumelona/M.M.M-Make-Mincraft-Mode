import json
import zipfile
from pathlib import Path

import pytest

from minecraft_mod_ai.audio_generator import register_existing_ogg
from minecraft_mod_ai.complete_orchestrator import (
    CompleteExecutionOptions,
    CompleteProductionOrchestrator,
)
from minecraft_mod_ai.complete_spec import (
    AssetRequest,
    AudioRequest,
    CompleteProposal,
    CompleteProposalStatus,
    ProductionModule,
    complete_proposal_from_parts,
)
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.geckolib_generator import generate_geckolib_entity_assets
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.source_patch import (
    SourcePatchError,
    TransactionalSourcePatcher,
    sha256_file,
)
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec, SpecValidationError
from minecraft_mod_ai.world_compiler import _build_blocks


def _spec() -> ModSpec:
    return ModSpec(
        mod_id="complete_test",
        mod_name="Complete Test",
        package_name="ai.minecraft.complete_test",
        version="1.0.0",
        summary="complete production test",
        contents=(
            ContentSpec(
                content_id="core_item",
                kind=ContentKind.ITEM,
                display_name_en="Core Item",
                display_name_ko="핵심 아이템",
            ),
        ),
    )


def _project(root: Path) -> Path:
    FabricProjectGenerator().generate(_spec(), root)
    return root


def test_complete_proposal_hash_covers_modules_and_rejects_cycles() -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan("Create one frost item")
    proposal = complete_proposal_from_parts(
        requested_prompt="Create quests and a weapon",
        base_proposal=base,
        game_design={"title": "Complete"},
        modules=(
            ProductionModule("weapon_one", "weapon", {"attack_damage": 5}),
            ProductionModule("quest_one", "quest", {}, ("weapon_one",)),
        ),
        acceptance_tests=("weapon and quest work",),
    )
    assert proposal.approve(proposal.calculate_hash()).status is CompleteProposalStatus.APPROVED
    raw = proposal.to_dict()
    raw["modules"][0]["config"]["attack_damage"] = 99
    with pytest.raises(SpecValidationError):
        CompleteProposal.from_dict(raw)

    with pytest.raises(SpecValidationError):
        complete_proposal_from_parts(
            requested_prompt="cycle",
            base_proposal=base,
            game_design={"title": "Cycle"},
            modules=(
                ProductionModule("left", "custom_java", {}, ("right",)),
                ProductionModule("right", "custom_java", {}, ("left",)),
            ),
            acceptance_tests=("never",),
        )


def test_source_patch_is_hash_guarded_and_transactional(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "file.txt"
    target.write_text("before\n", encoding="utf-8")
    patcher = TransactionalSourcePatcher(root)
    receipt = patcher.apply(
        [
            {
                "operation": "edit",
                "path": "file.txt",
                "expected_sha256": sha256_file(target),
                "replacements": [{"old": "before", "new": "after", "count": 1}],
            },
            {"operation": "create", "path": "new.txt", "content": "new\n"},
        ]
    )
    assert receipt["status"] == "APPLIED"
    assert target.read_text(encoding="utf-8") == "after\n"
    with pytest.raises(SourcePatchError):
        patcher.apply(
            [
                {
                    "operation": "replace",
                    "path": "file.txt",
                    "expected_sha256": "sha256:" + "0" * 64,
                    "content": "corrupt",
                }
            ]
        )
    assert target.read_text(encoding="utf-8") == "after\n"


def test_geckolib_generator_accumulates_real_entity_bindings(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    for entity_id in ("frost_guard", "ember_guard"):
        result = generate_geckolib_entity_assets(
            project_root=project,
            mod_id="complete_test",
            package_name="ai.minecraft.complete_test",
            entity_id=entity_id,
        )
        assert result["status"] == "fabric_binding_generated"
    registrar = project / "src/main/java/ai/minecraft/complete_test/geckolib/GeneratedGeckoEntities.java"
    text = registrar.read_text(encoding="utf-8")
    assert "FROST_GUARD" in text and "EMBER_GUARD" in text
    assert "FabricDefaultAttributeRegistry.register" in text
    entity_java = project / "src/main/java/ai/minecraft/complete_test/entity/FrostGuardEntity.java"
    entity_text = entity_java.read_text(encoding="utf-8")
    assert "animation.complete_test.frost_guard.idle" in entity_text
    assert "animation.complete_test.frost_guard.attack" in entity_text
    client = project / "src/main/java/ai/minecraft/complete_test/client/geckolib/GeneratedGeckoClient.java"
    assert client.read_text(encoding="utf-8").count("EntityRendererRegistry.register") == 2


def test_audio_registration_writes_soundevent_and_sounds_json(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    source = tmp_path / "click.ogg"
    source.write_bytes(b"OggS" + b"\0" * 256)
    result = register_existing_ogg(
        project_root=project,
        mod_id="complete_test",
        package_name="ai.minecraft.complete_test",
        sound_id="menu_click",
        ogg_path=source,
        kind="ui",
        subtitle_en="Menu click",
        subtitle_ko="메뉴 클릭",
    )
    assert result["status"] == "REGISTERED"
    sounds = json.loads(
        (project / "src/main/resources/assets/complete_test/sounds.json").read_text(encoding="utf-8")
    )
    assert sounds["menu_click"]["sounds"][0]["name"] == "complete_test:menu_click"
    root_java = project / "src/main/java/ai/minecraft/complete_test/sound/GeneratedSounds.java"
    assert "GeneratedSoundShard" in root_java.read_text(encoding="utf-8")
    shards = sorted(project.rglob("GeneratedSoundShard*.java"))
    assert shards
    assert any("SoundEvent.of" in path.read_text(encoding="utf-8") for path in shards)


def test_world_architecture_is_not_only_a_hollow_box() -> None:
    structure = {"kind": "village"}
    size = (9, 7, 9)
    palette = ["minecraft:stone_bricks", "minecraft:oak_planks", "minecraft:air"]
    blocks = dict(_build_blocks(structure, size, palette))
    air = 2
    assert blocks[(size[0] // 2, 2, size[2] // 2 + 2)] == 1
    assert blocks[(size[0] // 2, 1, 0)] == air
    assert any(position[1] >= 5 and state != air for position, state in blocks.items())


def test_complete_orchestrator_source_only_connects_all_generators(tmp_path: Path) -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan("Create one frost item")
    world_ir = {
        "schema_version": "mmm/world-ir-v1",
        "regions": [{"id": "spawn", "purpose": "start"}],
        "routes": [],
        "structures": [
            {
                "id": "spawn_hall",
                "region_id": "spawn",
                "kind": "village",
                "brief": "starter hall",
                "size": [7, 5, 7],
                "palette": ["minecraft:stone_bricks", "minecraft:oak_planks", "minecraft:air"],
                "biomes": ["minecraft:plains"],
            }
        ],
        "quests": [],
        "constraints": [],
    }
    proposal = complete_proposal_from_parts(
        requested_prompt="weapon, quest, animated entity and village",
        base_proposal=base,
        game_design={"title": "Integrated"},
        modules=(
            ProductionModule("frost_blade", "weapon", {"attack_damage": 6}),
            ProductionModule("first_quest", "quest", {}, ("frost_blade",)),
            ProductionModule("frost_guard", "entity", {"max_health": 60}),
            ProductionModule("spawn_hall", "structure", {}),
        ),
        world_ir=world_ir,
        acceptance_tests=("all generated systems are present",),
    )
    result = CompleteProductionOrchestrator(workspace_root=tmp_path / "out").execute(
        proposal,
        approval_hash=proposal.calculate_hash(),
        run_name="integrated",
        options=CompleteExecutionOptions(
            source_only=True,
            run_jdt=False,
            run_blockbench=False,
            run_runtime=False,
            run_client=False,
            run_mineflayer=False,
            run_visual_review=False,
        ),
    )
    assert result.status == "SOURCE_READY"
    project = Path(result.project_root)
    package_path = Path(*base.spec.package_name.split("."))
    assert (project / "src/main/java" / package_path / "extended/GeneratedExtendedContent.java").is_file()
    assert any(path.name == "QuestSystem.java" for path in project.rglob("QuestSystem.java"))
    assert any(path.name == "GeneratedGeckoEntities.java" for path in project.rglob("GeneratedGeckoEntities.java"))
    assert (project / f"src/main/resources/data/{base.spec.mod_id}/structures/spawn_hall.nbt").is_file()
    with zipfile.ZipFile(result.release_zip) as archive:
        assert any(name.endswith("GeneratedExtendedContent.java") for name in archive.namelist())
