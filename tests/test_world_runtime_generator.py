from __future__ import annotations

import json
import shutil
from pathlib import Path

from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.scalable_world_compiler import (
    compile_scalable_world_ir,
)
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
from minecraft_mod_ai.world_runtime_generator import (
    generate_world_runtime_bridge,
)


def _spec() -> ModSpec:
    return ModSpec(
        mod_id="world_test",
        mod_name="World Test",
        package_name="ai.minecraft.world_test",
        version="1.0.0",
        summary="generated world runtime test",
        contents=(
            ContentSpec(
                content_id="world_anchor",
                kind=ContentKind.ITEM,
                display_name_en="World Anchor",
                display_name_ko="월드 기준점",
            ),
        ),
    )


def _world_ir(structure_count: int) -> dict:
    return {
        "schema_version": "mmm/world-ir-v1",
        "regions": [
            {
                "id": "surface",
                "purpose": "generated settlements",
                "min_y": 0,
                "max_y": 240,
            }
        ],
        "routes": [
            {
                "from": "surface",
                "to": "surface",
                "travel_mode": "road",
            }
        ],
        "structures": [
            {
                "id": f"settlement_{index:04d}",
                "region_id": "surface",
                "kind": "village",
                "size": [33, 2, 2],
                "palette": [
                    "minecraft:stone_bricks",
                    "minecraft:air",
                ],
                "biomes": ["minecraft:plains"],
                "spacing": 48,
                "separation": 12,
                "salt": 700_000 + index,
            }
            for index in range(structure_count)
        ],
        "quests": [
            {
                "id": "visit_surface",
                "region_id": "surface",
                "objective": "visit a generated settlement",
            }
        ],
        "constraints": [
            {
                "id": "overworld_surface",
                "dimensions": ["minecraft:overworld"],
                "min_y": 0,
            }
        ],
    }


def _install_world(
    tmp_path: Path,
    *,
    structure_count: int,
    suffix: str,
) -> tuple[Path, Path, dict]:
    compiled = tmp_path / f"compiled-{suffix}"
    result = compile_scalable_world_ir(
        _world_ir(structure_count),
        mod_id="world_test",
        output_root=compiled,
        package_world_zip=False,
        policy=ScalePolicy(
            nbt_piece_axis=32,
            nbt_piece_volume=32768,
            world_placements_per_tick=2,
        ),
    )
    project = tmp_path / f"project-{suffix}"
    FabricProjectGenerator().generate(_spec(), project)
    shutil.copytree(
        compiled / "data/world_test",
        project / "src/main/resources/data/world_test",
        dirs_exist_ok=True,
    )
    receipt = generate_world_runtime_bridge(
        project_root=project,
        mod_id="world_test",
        package_name="ai.minecraft.world_test",
    )
    return compiled, project, receipt


def test_runtime_bridge_preserves_anchor_and_persists_concurrent_jobs(
    tmp_path: Path,
) -> None:
    compiled, project, receipt = _install_world(
        tmp_path,
        structure_count=2,
        suffix="runtime",
    )

    assert receipt["runtime_structure_count"] == 2
    runtime = (
        project
        / "src/main/java/ai/minecraft/world_test/world/"
        "GeneratedWorldRuntime.java"
    ).read_text(encoding="utf-8")
    state = (
        project
        / "src/main/java/ai/minecraft/world_test/world/"
        "GeneratedWorldState.java"
    ).read_text(encoding="utf-8")
    assert "ServerChunkEvents.CHUNK_LOAD" in runtime
    assert "ServerLifecycleEvents.SERVER_STARTING" in runtime
    assert "ServerLifecycleEvents.END_DATA_PACK_RELOAD" in runtime
    assert "CONTRACT_CHECKS_PER_TICK = 64" in runtime
    assert "GeneratedWorldState state = state(world)" in runtime
    assert ".withPosition(Vec3d.of(job.anchor()))" in runtime
    assert "executeOnePlacementShard" in runtime
    assert "appliesTo(constraint, structureId, regionId)" in runtime
    assert "cursor = (cursor + 1) % jobs.size()" in state
    assert 'root.put("GeneratedAnchors", anchors)' in state
    assert 'root.put("Jobs", savedJobs)' in state
    assert "settlement_0000" not in runtime

    functions = sorted(
        (
            compiled
            / "data/world_test/functions/generated/settlement_0000"
        ).glob("*.mcfunction")
    )
    assert functions
    assert all(
        "schedule function" not in path.read_text(encoding="utf-8")
        for path in functions
    )
    main = (
        project
        / "src/main/java/ai/minecraft/world_test/WorldTestMod.java"
    ).read_text(encoding="utf-8")
    assert "GeneratedWorldRuntime.register()" in main


def test_runtime_code_and_root_manifest_do_not_grow_with_structure_count(
    tmp_path: Path,
) -> None:
    small_root, small_project, _ = _install_world(
        tmp_path,
        structure_count=1,
        suffix="small",
    )
    large_root, large_project, large_receipt = _install_world(
        tmp_path,
        structure_count=130,
        suffix="large",
    )

    small_manifest = (
        small_root / "mmm-world-manifest.json"
    ).read_text(encoding="utf-8")
    large_manifest = (
        large_root / "mmm-world-manifest.json"
    ).read_text(encoding="utf-8")
    assert len(large_manifest) - len(small_manifest) < 32
    assert "settlement_0000" not in large_manifest
    assert "logical_structures" not in large_manifest
    assert '"files"' not in large_manifest

    relative_java = Path(
        "src/main/java/ai/minecraft/world_test/world/"
        "GeneratedWorldRuntime.java"
    )
    assert (
        small_project / relative_java
    ).read_bytes() == (
        large_project / relative_java
    ).read_bytes()
    assert large_receipt["runtime_structure_count"] == 130
    contracts = list(
        (
            large_root
            / "data/world_test/mmm_world/contracts/structures"
        ).glob("*.json")
    )
    assert len(contracts) == 130

    manifest = json.loads(large_manifest)
    assert manifest["logical_structure_count"] == 130
    assert manifest["partitioned_structure_count"] == 130
