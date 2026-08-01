from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.production_hardener import _gametest_files
from minecraft_mod_ai.production_hardener import harden_generated_project
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec


def _definitions(count: int) -> list[dict[str, str]]:
    return [
        {
            "registry": "ITEM",
            "id": f"generated_item_{index:05d}",
        }
        for index in range(count)
    ]


def test_production_gametest_entrypoint_is_constant_while_units_shard() -> None:
    small_files, small_entries, small_shards = _gametest_files(
        package_name="ai.minecraft.scale",
        mod_id="scale",
        definitions=_definitions(1),
        shard_size=32,
    )
    large_files, large_entries, large_shards = _gametest_files(
        package_name="ai.minecraft.scale",
        mod_id="scale",
        definitions=_definitions(4097),
        shard_size=32,
    )

    root_path = (
        "src/main/java/ai/minecraft/scale/gametest/"
        "GeneratedRegistryGameTest.java"
    )
    assert small_entries == large_entries == [
        "ai.minecraft.scale.gametest.GeneratedRegistryGameTest"
    ]
    assert small_files[root_path] == large_files[root_path]
    assert "Files.list(directory)" in large_files[root_path]
    assert "generated_item_04096" not in large_files[root_path]
    assert small_shards == 1
    assert large_shards == 129
    assert len(large_files) == large_shards + 1
    assert all(
        source.count("containsId") <= 32
        for path, source in large_files.items()
        if "GeneratedRegistryGameTestUnit" in path
    )


def test_hardener_removes_obsolete_generated_tests_when_catalog_is_empty(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    FabricProjectGenerator().generate(
        ModSpec(
            mod_id="scale",
            mod_name="Scale",
            package_name="ai.minecraft.scale",
            version="1.0.0",
            summary="migration fixture",
            contents=(
                ContentSpec(
                    content_id="anchor",
                    kind=ContentKind.ITEM,
                    display_name_en="Anchor",
                    display_name_ko="Anchor",
                ),
            ),
        ),
        project,
    )
    stale_directory = (
        project
        / "src/main/java/ai/minecraft/scale/gametest"
    )
    stale_directory.mkdir(parents=True)
    stale = stale_directory / "GeneratedRegistryGameTest0000.java"
    stale.write_text(
        "package ai.minecraft.scale.gametest;\n",
        encoding="utf-8",
    )
    metadata_path = project / "src/main/resources/fabric.mod.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["entrypoints"]["fabric-gametest"].extend(
        [
            (
                "ai.minecraft.scale.gametest."
                "GeneratedRegistryGameTest0000"
            ),
            "third.party.KeepThisGameTest",
        ]
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    first = harden_generated_project(project)
    assert first["status"] == "HARDENED"
    assert not stale.exists()
    updated = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = updated["entrypoints"]["fabric-gametest"]
    assert "third.party.KeepThisGameTest" in entries
    assert not any(
        "GeneratedRegistryGameTest" in str(entry)
        for entry in entries
    )
    assert harden_generated_project(project)["status"] == "UNCHANGED"
