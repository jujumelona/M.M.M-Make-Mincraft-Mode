from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.complete_spec import ProductionModule, complete_proposal_from_parts
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.mineflayer_bridge import MineflayerBridge
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.project_edit import ensure_main_initializer_call, inspect_fabric_project
from minecraft_mod_ai.project_index import ProjectIndex
from minecraft_mod_ai.scalable_generator import ScalableFabricProjectGenerator
from minecraft_mod_ai.scalable_validator import ScalableProjectValidator
from minecraft_mod_ai.scalable_world_compiler import compile_scalable_world_ir
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
from minecraft_mod_ai.system_pack_generator import generate_system_pack


def _spec(*, count: int = 1) -> ModSpec:
    return ModSpec(
        mod_id="scale_test",
        mod_name="Scale Test",
        package_name="ai.minecraft.scale_test",
        version="1.0.0",
        summary="scalable generation test",
        contents=tuple(
            ContentSpec(
                content_id=f"item_{index:04d}",
                kind=ContentKind.ITEM,
                display_name_en=f"Item {index}",
                display_name_ko=f"아이템 {index}",
                recipe=True,
            )
            for index in range(count)
        ),
    )


def test_complete_spec_accepts_large_iterative_dependency_graph() -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan(
        "Create one bootstrap frost item"
    )
    modules = tuple(
        ProductionModule(
            module_id=f"feature_{index:04d}",
            kind="custom_java",
            config={"index": index},
            depends_on=(f"feature_{index - 1:04d}",) if index else (),
        )
        for index in range(1500)
    )
    proposal = complete_proposal_from_parts(
        requested_prompt="Create a large complete mod",
        base_proposal=base,
        game_design={"title": "Large"},
        modules=modules,
        acceptance_tests=("all features are generated",),
    )
    assert len(proposal.modules) == 1500
    assert proposal.approve(proposal.calculate_hash()).status.value == "approved"


def test_scalable_generator_shards_bootstrap_content_and_gametests(
    tmp_path: Path,
) -> None:
    policy = ScalePolicy(java_shard_size=16)
    spec = _spec(count=130)
    project = tmp_path / "project"
    result = ScalableFabricProjectGenerator(policy=policy).generate(spec, project)
    assert result.root == project.resolve()

    shard_files = sorted(project.rglob("GeneratedContentShard*.java"))
    assert len(shard_files) == 9
    main_java = project / "src/main/java/ai/minecraft/scale_test/ScaleTestMod.java"
    assert "ITEM_0129" not in main_java.read_text(encoding="utf-8")

    metadata = json.loads(
        (project / "src/main/resources/fabric.mod.json").read_text(encoding="utf-8")
    )
    scale_tests = [
        value
        for value in metadata["entrypoints"]["fabric-gametest"]
        if "ScalableContentGameTest" in value
    ]
    assert len(scale_tests) == 9
    assert ScalableProjectValidator(policy=policy).validate(project, spec).passed


def test_project_index_retrieves_relevant_file_not_first_directory_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    source = root / "src/main/java/example"
    source.mkdir(parents=True)
    for index in range(200):
        (source / f"A{index:04d}.java").write_text(
            f"package example; final class A{index:04d} {{}}\n",
            encoding="utf-8",
        )
    important = source / "ZImportantMachine.java"
    important.write_text(
        "package example; final class ZImportantMachine { void quantumAssemblerFault() {} }\n",
        encoding="utf-8",
    )
    context = ProjectIndex(
        root,
        policy=ScalePolicy(model_context_bytes=32_000),
    ).select(query="quantumAssemblerFault")
    selected = {item["path"] for item in context["files"]}
    assert important.relative_to(root).as_posix() in selected
    assert context["indexed_file_count"] == 201


def test_large_logical_world_is_partitioned_instead_of_rejected(
    tmp_path: Path,
) -> None:
    ir = {
        "schema_version": "mmm/world-ir-v1",
        "regions": [{"id": "capital", "purpose": "large city"}],
        "routes": [],
        "structures": [
            {
                "id": "capital_city",
                "region_id": "capital",
                "kind": "village",
                "brief": "large connected capital",
                "size": [96, 40, 96],
                "palette": [
                    "minecraft:stone_bricks",
                    "minecraft:oak_planks",
                    "minecraft:air",
                ],
                "biomes": ["minecraft:plains"],
            }
        ],
        "quests": [],
        "constraints": [],
    }
    result = compile_scalable_world_ir(
        ir,
        mod_id="scale_test",
        output_root=tmp_path / "world",
        policy=ScalePolicy(nbt_piece_axis=32, nbt_piece_volume=32768),
    )
    assert result["partitioned_structures"] == ["capital_city"]
    assert result["physical_template_count"] == 18
    assert (
        tmp_path
        / "world/data/scale_test/functions/build_capital_city.mcfunction"
    ).is_file()


def test_shop_uses_server_owned_catalog_and_not_player_supplied_price(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    FabricProjectGenerator().generate(_spec(count=1), project)
    generate_system_pack(
        project_root=project,
        pack_id="economy-shop",
        mod_id="scale_test",
        package_name="ai.minecraft.scale_test",
        config={
            "modules": [
                {
                    "module_id": "currency",
                    "kind": "economy",
                    "config": {"initial_balance": 100},
                    "depends_on": [],
                    "required_gates": [],
                },
                {
                    "module_id": "general_shop",
                    "kind": "shop",
                    "config": {
                        "entries": [
                            {
                                "id": "bread",
                                "item": "minecraft:bread",
                                "count": 1,
                                "price": 4,
                            }
                        ]
                    },
                    "depends_on": ["currency"],
                    "required_gates": [],
                },
            ]
        },
    )
    java = next(project.rglob("EconomyShopSystem.java")).read_text(
        encoding="utf-8"
    )
    assert "CATALOG.get(id)" in java
    assert 'argument("price"' not in java
    assert 'getInteger(context, "price")' not in java


def test_nonstandard_initializer_falls_back_to_generated_entrypoint(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    FabricProjectGenerator().generate(_spec(count=1), project)
    info = inspect_fabric_project(project)
    info.main_java.write_text(
        "package ai.minecraft.scale_test; public final class ScaleTestMod {}\n",
        encoding="utf-8",
    )
    receipt = ensure_main_initializer_call(
        inspect_fabric_project(project),
        import_line="import ai.minecraft.scale_test.extended.GeneratedExtendedContent",
        call_line="GeneratedExtendedContent.register()",
        marker="test:fallback",
    )
    assert receipt["status"] == "APPLIED"
    metadata = json.loads(info.fabric_mod_json.read_text(encoding="utf-8"))
    assert any(
        "MmmGeneratedInitializer" in value
        for value in metadata["entrypoints"]["main"]
    )


def test_mineflayer_playtest_surface_covers_real_interactions() -> None:
    assert {
        "craft",
        "chat",
        "wait_for",
        "open_container",
        "click_slot",
    } <= MineflayerBridge.ACTIONS
