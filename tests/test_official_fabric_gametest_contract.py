from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import fabric_official_template_provider as provider


def test_official_scaffold_installs_host_gametest_contract(tmp_path: Path) -> None:
    root = tmp_path / "project"
    resources = root / "src/main/resources"
    resources.mkdir(parents=True)
    (root / "build.gradle").write_text(
        "plugins { id 'fabric-loom' }\n",
        encoding="utf-8",
    )
    metadata = resources / "fabric.mod.json"
    metadata.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "mmm_debug_fixture",
                "version": "1.0.0",
                "entrypoints": {"main": ["dev.mmm.debugfixture.MmmDebugFixtureMod"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = SimpleNamespace(
        mod_id="mmm_debug_fixture",
        package_name="dev.mmm.debugfixture",
    )

    receipt = provider._install_host_gametest_contract(root, spec)

    build = (root / "build.gradle").read_text(encoding="utf-8")
    assert "configureTests" in build
    assert "enableGameTests = true" in build
    assert "enableClientGameTests = false" in build
    assert "fabric-api.gametest.report-file" in build
    assert receipt["task"] == "runGameTest"
    assert receipt["report"] == "build/gametest-report.xml"

    updated = json.loads(metadata.read_text(encoding="utf-8"))
    assert updated["entrypoints"]["fabric-gametest"] == [
        "dev.mmm.debugfixture.MmmDebugFixtureModGameTests"
    ]
    source = root / receipt["source"]
    text = source.read_text(encoding="utf-8")
    assert "net.fabricmc.fabric.api.gametest.v1.GameTest" in text
    assert "net.minecraft.gametest.framework.GameTestHelper" in text
    assert "generatedRegistriesAreLive" in text
    assert 'isModLoaded("mmm_debug_fixture")' in text

    provider._install_host_gametest_contract(root, spec)
    repeated = (root / "build.gradle").read_text(encoding="utf-8")
    assert repeated.count("configureTests") == 1
    assert repeated.count("fabric-api.gametest.report-file") == 1
    repeated_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    assert repeated_metadata["entrypoints"]["fabric-gametest"].count(
        "dev.mmm.debugfixture.MmmDebugFixtureModGameTests"
    ) == 1
