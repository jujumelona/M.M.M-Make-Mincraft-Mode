import gzip
import hashlib
import json
import zipfile
from pathlib import Path

from minecraft_mod_ai.world_compiler import compile_world_ir


def _ir() -> dict:
    return {
        "schema_version": "mmm/world-ir-v1",
        "regions": [
            {"id": "spawn", "purpose": "start"},
            {"id": "dungeon", "purpose": "combat"},
        ],
        "routes": [{"from": "spawn", "to": "dungeon", "travel_mode": "road"}],
        "structures": [
            {
                "id": "spawn_hall",
                "region_id": "spawn",
                "kind": "village",
                "brief": "starter hall",
                "size": [7, 5, 7],
                "palette": ["minecraft:stone_bricks", "minecraft:air"],
                "biomes": ["minecraft:plains"],
            }
        ],
        "quests": [
            {
                "id": "first_steps",
                "region_id": "spawn",
                "objective": "reach the starter hall",
            }
        ],
        "constraints": [
            {
                "id": "surface_only",
                "min_y": 0,
                "max_y": 200,
            }
        ],
    }


def test_world_ir_compiles_real_gzipped_structure_nbt_and_jigsaw(tmp_path: Path) -> None:
    output = tmp_path / "world-pack"
    result = compile_world_ir(
        _ir(),
        mod_id="testmod",
        output_root=output,
    )
    nbt = output / "data/testmod/structures/spawn_hall.nbt"
    assert nbt.is_file()
    raw = gzip.decompress(nbt.read_bytes())
    assert raw[0] == 10
    pool = json.loads(
        (
            output
            / "data/testmod/worldgen/template_pool/spawn_hall.json"
        ).read_text(encoding="utf-8")
    )
    assert pool["elements"][0]["element"]["location"] == "testmod:spawn_hall"
    archive = Path(result["world_zip"])
    assert archive.is_file()
    with zipfile.ZipFile(archive) as zipped:
        assert "data/testmod/structures/spawn_hall.nbt" in zipped.namelist()

    contracts = output / "data/testmod/mmm_world/contracts"
    assert (
        contracts / "regions/spawn.json"
    ).is_file()
    route_contract = next((contracts / "routes").glob("*.json"))
    route = json.loads(route_contract.read_text(encoding="utf-8"))
    assert route["operation"] == "connect_regions"
    quest = json.loads(
        (contracts / "quests/first_steps.json").read_text(
            encoding="utf-8"
        )
    )
    assert quest["operation"] == "register_quest"
    constraint = json.loads(
        (contracts / "constraints/surface_only.json").read_text(
            encoding="utf-8"
        )
    )
    assert constraint["operation"] == "enforce_constraint"

    manifest = json.loads(
        (output / "mmm-world-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == "mmm/world-compile-manifest-v3"
    assert "files" not in manifest
    assert "logical_structures" not in manifest


def test_world_zip_is_reproducible_across_clean_output_roots(
    tmp_path: Path,
) -> None:
    first = compile_world_ir(
        _ir(),
        mod_id="testmod",
        output_root=tmp_path / "first",
    )
    second = compile_world_ir(
        _ir(),
        mod_id="testmod",
        output_root=tmp_path / "second",
    )

    first_digest = hashlib.sha256(
        Path(first["world_zip"]).read_bytes()
    ).hexdigest()
    second_digest = hashlib.sha256(
        Path(second["world_zip"]).read_bytes()
    ).hexdigest()
    assert first_digest == second_digest


def test_world_compile_is_idempotent_and_preserves_changed_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "world-pack"
    first = compile_world_ir(
        _ir(),
        mod_id="testmod",
        output_root=output,
        package_world_zip=False,
    )
    second = compile_world_ir(
        _ir(),
        mod_id="testmod",
        output_root=output,
        package_world_zip=False,
    )

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert not list(tmp_path.glob(".world-pack.replaced-*"))

    changed = json.loads(json.dumps(_ir()))
    changed["regions"][0]["purpose"] = "revised start"
    replaced = compile_world_ir(
        changed,
        mod_id="testmod",
        output_root=output,
        package_world_zip=False,
    )

    assert replaced["manifest_sha256"] != first["manifest_sha256"]
    preserved = list(tmp_path.glob(".world-pack.replaced-*"))
    assert len(preserved) == 1
    assert (
        json.loads(
            (preserved[0] / "mmm-world-manifest.json").read_text(
                encoding="utf-8"
            )
        )["world_ir_sha256"]
        != json.loads(
            (output / "mmm-world-manifest.json").read_text(
                encoding="utf-8"
            )
        )["world_ir_sha256"]
    )


def test_duplicate_optional_contract_ids_do_not_overwrite_shards(
    tmp_path: Path,
) -> None:
    ir = _ir()
    ir["quests"].append(
        {
            "id": "first_steps",
            "region_id": "dungeon",
            "objective": "enter the dungeon",
        }
    )
    output = tmp_path / "world-pack"

    compile_world_ir(
        ir,
        mod_id="testmod",
        output_root=output,
        package_world_zip=False,
    )

    quest_files = sorted(
        (
            output
            / "data/testmod/mmm_world/contracts/quests"
        ).glob("*.json")
    )
    assert len(quest_files) == 2
    contracts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in quest_files
    ]
    assert len({contract["id"] for contract in contracts}) == 2
    assert {
        contract["payload"]["objective"]
        for contract in contracts
    } == {"reach the starter hall", "enter the dungeon"}


def test_complete_staging_tree_is_promoted_after_interruption(
    tmp_path: Path,
) -> None:
    output = tmp_path / "world-pack"
    stage = tmp_path / ".world-pack.mmm-stage"
    compile_world_ir(
        _ir(),
        mod_id="testmod",
        output_root=stage,
        package_world_zip=False,
    )
    assert stage.is_dir()

    result = compile_world_ir(
        _ir(),
        mod_id="testmod",
        output_root=output,
        package_world_zip=False,
    )

    assert Path(result["output_root"]) == output.resolve()
    assert output.is_dir()
    assert not stage.exists()
