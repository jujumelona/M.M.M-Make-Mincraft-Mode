from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import fabric_official_template_provider as provider


def test_official_scaffold_is_gametest_ready_before_first_build(tmp_path: Path) -> None:
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
                "entrypoints": {
                    "main": ["dev.mmm.debugfixture.MmmDebugFixtureMod"]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = SimpleNamespace(
        mod_id="mmm_debug_fixture",
        package_name="dev.mmm.debugfixture",
    )
    adapter = SimpleNamespace(minecraft_version="1.21.8")

    receipt = provider._install_host_gametest_contract(root, spec, adapter)

    build = (root / "build.gradle").read_text(encoding="utf-8")
    assert "configureTests" in build
    assert "createSourceSet = true" in build
    assert 'modId = "mmm_debug_fixture_gametest"' in build
    assert "enableGameTests = true" in build
    assert "enableClientGameTests = false" in build
    assert "fabric-api.gametest.report-file" in build
    assert (
        'property "fabric-api.gametest.report-file", '
        "file('build/gametest-report.xml').absolutePath"
    ) in build
    assert 'vmArg "-Dfabric-api.gametest.report-file=' not in build
    assert "gameTest {" in build
    assert "gameTestServer" not in build
    assert receipt == {
        "task": "runGameTest",
        "report": "build/gametest-report.xml",
        "entrypoint": "dev.mmm.debugfixture.MmmDebugFixtureModGameTests",
        "source": (
            "src/gametest/java/dev/mmm/debugfixture/"
            "MmmDebugFixtureModGameTests.java"
        ),
        "metadata": "src/gametest/resources/fabric.mod.json",
        "mod_id": "mmm_debug_fixture_gametest",
        "api_generation": "fabric-gametest-v2",
        "minecraft_version": "1.21.8",
    }

    updated = json.loads(metadata.read_text(encoding="utf-8"))
    assert "fabric-gametest" not in updated["entrypoints"]
    test_metadata = json.loads(
        (root / receipt["metadata"]).read_text(encoding="utf-8")
    )
    assert test_metadata["id"] == "mmm_debug_fixture_gametest"
    assert test_metadata["depends"]["mmm_debug_fixture"] == "*"
    assert test_metadata["entrypoints"]["fabric-gametest"] == [
        "dev.mmm.debugfixture.MmmDebugFixtureModGameTests"
    ]

    source = (root / receipt["source"]).read_text(encoding="utf-8")
    assert "net.fabricmc.fabric.api.gametest.v1.GameTest" in source
    assert "FabricGameTest" not in source
    assert "net.minecraft.gametest.framework.GameTestHelper" in source
    assert "generatedRegistriesAreLive" in source
    assert 'isModLoaded("mmm_debug_fixture")' in source
    assert "context.succeed();" in source

    provider._install_host_gametest_contract(root, spec, adapter)
    repeated = (root / "build.gradle").read_text(encoding="utf-8")
    assert repeated.count("configureTests") == 1
    assert repeated.count("fabric-api.gametest.report-file") == 1
    repeated_metadata = json.loads(
        (root / receipt["metadata"]).read_text(encoding="utf-8")
    )
    assert repeated_metadata["entrypoints"]["fabric-gametest"].count(
        "dev.mmm.debugfixture.MmmDebugFixtureModGameTests"
    ) == 1
