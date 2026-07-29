import gzip
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
        "quests": [],
        "constraints": [],
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
