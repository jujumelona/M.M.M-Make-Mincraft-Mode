from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai.mod_output_scope import (
    ModOutputScopeError,
    validate_mod_output_path,
)
from minecraft_mod_ai.source_patch import SourcePatchError, TransactionalSourcePatcher


@pytest.mark.parametrize(
    "path",
    (
        "src/main/java/dev/mmm/world/FrostBiome.java",
        "src/main/resources/data/mmm/worldgen/biome/frost.json",
        "src/main/resources/data/mmm/worldgen/configured_feature/frost_ore.json",
        "src/main/resources/data/mmm/dimension/frost.json",
        "src/main/resources/data/mmm/structure/frost_temple.nbt",
        "src/main/resources/data/mmm/advancement/explore_frost.json",
        "dist/mmm-mod-release.zip",
    ),
)
def test_mod_native_outputs_remain_allowed(path: str) -> None:
    validate_mod_output_path(path)


@pytest.mark.parametrize(
    "path",
    (
        "castle.schem",
        "exports/castle.schematic",
        "exports/castle.litematic",
        "exports/world.mcworld",
        "region/r.0.0.mca",
        "saves/demo/region/r.0.0.mca",
        "world/playerdata/00000000-0000-0000-0000-000000000000.dat",
        "level.dat",
        "artifacts/buildspec.json",
        "artifacts/block_delta.json",
        "artifacts/blocks.npz",
    ),
)
def test_standalone_world_and_builder_outputs_are_rejected(path: str) -> None:
    with pytest.raises(ModOutputScopeError, match="outside M.M.M's mod-project scope"):
        validate_mod_output_path(path)


def test_transactional_patcher_rejects_standalone_output_before_write(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(SourcePatchError, match="mod-project scope"):
        TransactionalSourcePatcher(root).apply(
            [{"operation": "create", "path": "exports/castle.schem", "content": "x"}]
        )

    assert not (root / "exports" / "castle.schem").exists()


def test_transactional_patcher_keeps_mod_worldgen_writable(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    relative = "src/main/resources/data/mmm/worldgen/biome/frost.json"

    receipt = TransactionalSourcePatcher(root).apply(
        [{"operation": "create", "path": relative, "content": "{}\n"}]
    )

    assert receipt["status"] == "APPLIED"
    assert (root / relative).read_text(encoding="utf-8") == "{}\n"
