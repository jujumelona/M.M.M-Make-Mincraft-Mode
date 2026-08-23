from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from minecraft_mod_ai.project_inventory import (
    ProjectInventory,
    ProjectInventoryError,
    inspect_existing_archive_inventory,
    inspect_project_inventory,
    validate_project_inventory_payload,
)


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _fabric_fixture(root: Path) -> None:
    _write(root / "settings.gradle.kts", 'rootProject.name = "inventory-demo"\n')
    _write(
        root / "gradle.properties",
        "minecraft_version=1.21.1\nloader_version=0.16.10\nfabric_version=0.115.1+1.21.1\n",
    )
    _write(
        root / "build.gradle.kts",
        """
plugins { id("fabric-loom") version "1.9.2" }
java { toolchain.languageVersion.set(JavaLanguageVersion.of(21)) }
sourceSets { main { resources.srcDir("src/main/generated") } }
dependencies {
    minecraft("com.mojang:minecraft:${minecraft_version}")
    modImplementation("net.fabricmc:fabric-loader:${loader_version}")
    modImplementation("net.fabricmc.fabric-api:fabric-api:${fabric_version}")
}
""".strip()
        + "\n",
    )
    _write(
        root / "gradle/wrapper/gradle-wrapper.properties",
        "distributionUrl=https\\://services.gradle.org/distributions/gradle-8.12-bin.zip\n",
    )
    _write(
        root / "src/main/resources/fabric.mod.json",
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "inventory_demo",
                "name": "Inventory Demo",
                "version": "1.0.0",
                "environment": "*",
                "entrypoints": {"main": ["example.inventory.InventoryMod"]},
                "depends": {"minecraft": "~1.21.1", "fabricloader": ">=0.16.10"},
            }
        ),
    )
    _write(
        root / "src/main/java/example/inventory/InventoryMod.java",
        """package example.inventory;
import net.fabricmc.api.ModInitializer;
public final class InventoryMod implements ModInitializer {
    public static final String MODEL = "inventory_demo:item/widget";
    public void onInitialize() {}
}
""",
    )
    _write(
        root / "src/test/java/example/inventory/InventoryModTest.java",
        """package example.inventory;
public final class InventoryModTest {
    public void createsWidget() {}
}
""",
    )
    _write(
        root / "src/main/resources/assets/inventory_demo/models/item/widget.json",
        '{"parent":"minecraft:item/generated","textures":{"layer0":"inventory_demo:item/widget"}}\n',
    )
    _write(
        root / "src/main/resources/assets/inventory_demo/textures/item/widget.png",
        b"not-executed-image-bytes",
    )
    _write(root / "src/main/generated/data/inventory_demo/recipes/widget.json", "{}\n")
    _write(root / ".github/workflows/release.yml", "name: release\n")
    _write(root / "LICENSE", "fixture license\n")


def test_fabric_inventory_is_complete_relocatable_and_hash_bound(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _fabric_fixture(first_root)
    shutil.copytree(first_root, second_root)

    first = inspect_project_inventory(first_root)
    second = inspect_project_inventory(second_root)

    assert first.schema_version == "mmm/project-inventory-v1"
    assert first.root_name == "inventory-demo"
    assert first.project_snapshot_sha256 == second.project_snapshot_sha256
    assert first.inventory_sha256 == second.inventory_sha256
    assert first.target.minecraft_versions == ("1.21.1", "~1.21.1")
    assert first.target.loaders == ("fabric",)
    assert "21" in first.target.java_versions
    assert first.target.gradle_versions == ("8.12",)
    assert any(item.coordinate == "com.mojang:minecraft:1.21.1" for item in first.dependencies)
    assert first.metadata[0].mod_id == "inventory_demo"
    assert first.entrypoints[0].value == "example.inventory.InventoryMod"
    assert "example.inventory" in first.namespaces
    assert "model:inventory_demo:item/widget" in first.logical_resource_ids
    assert "texture:inventory_demo:item/widget" in first.logical_resource_ids
    assert "minecraft:item/generated" in first.logical_resource_references
    root_module = next(item for item in first.modules if item.module_id == ":")
    assert root_module.source_sets == ("main", "test")
    assert "src/main/generated" in root_module.generated_resource_roots
    assert "src/test/java" in root_module.test_roots
    provides = {value for component in first.components for value in component.provides}
    assert "symbol:example.inventory.InventoryMod" in provides
    assert "capability:inventory_mod" in provides
    assert "capability:on_initialize" in provides
    assert "capability:inventory_mod_on_initialize" in provides
    assert "capability:widget" in provides
    assert "test:example.inventory.InventoryModTest" in provides
    assert "release_config:.github/workflows/release.yml" in provides
    assert all(component.evidence for component in first.components)
    assert all(
        component.content_sha256 in {item.sha256 for item in component.evidence}
        for component in first.components
    )
    first.validate()
    assert ProjectInventory.from_dict(first.to_dict()) == first
    assert validate_project_inventory_payload(first.to_dict()) == first.to_dict()


def test_payload_validator_rejects_unknown_fields_and_stale_component_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    _fabric_fixture(root)
    inventory = inspect_project_inventory(root)

    unknown = inventory.to_dict()
    unknown["model_summary"] = "trust me"
    with pytest.raises(ProjectInventoryError, match="unknown fields"):
        validate_project_inventory_payload(unknown)

    stale = inventory.to_dict()
    stale["component_catalog"]["components"][0]["content_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ProjectInventoryError, match="content hash|catalog hash"):
        validate_project_inventory_payload(stale)


def test_multiloader_kotlin_topology_and_project_edges(tmp_path: Path) -> None:
    root = tmp_path / "multi"
    _write(
        root / "settings.gradle.kts",
        'rootProject.name = "multi-demo"\ninclude("common", "fabric", "neoforge")\n'
        'project(":fabric").projectDir = file("loaders/fabric")\n',
    )
    _write(root / "gradle.properties", "minecraft_version=1.21.1\njava_version=21\n")
    _write(root / "common/build.gradle.kts", "plugins { kotlin(\"jvm\") version \"2.1.0\" }\n")
    _write(
        root / "common/src/main/kotlin/example/common/CommonHooks.kt",
        """package example.common
object CommonHooks {
    fun initialize() = Unit
}
""",
    )
    _write(
        root / "loaders/fabric/build.gradle.kts",
        """plugins { id("fabric-loom") version "1.9.2" }
dependencies {
    implementation(project(":common"))
    minecraft("com.mojang:minecraft:${minecraft_version}")
    modImplementation("net.fabricmc:fabric-loader:0.16.10")
}
""",
    )
    _write(
        root / "loaders/fabric/src/main/resources/fabric.mod.json",
        '{"schemaVersion":1,"id":"multi_fabric","version":"1.0.0","entrypoints":{"main":["example.fabric.FabricEntry"]},"depends":{"minecraft":"1.21.1"}}',
    )
    _write(
        root / "loaders/fabric/src/main/java/example/fabric/FabricEntry.java",
        "package example.fabric; public final class FabricEntry {}\n",
    )
    _write(
        root / "neoforge/build.gradle.kts",
        'plugins { id("net.neoforged.moddev") version "2.0.80" }\n'
        'dependencies { implementation(project(":common")); implementation("net.neoforged:neoforge:21.1.120") }\n',
    )
    _write(
        root / "neoforge/src/main/resources/META-INF/neoforge.mods.toml",
        'modLoader="javafml"\nloaderVersion="[4,)"\nlicense="MIT"\n[[mods]]\nmodId="multi_neoforge"\nversion="1.0.0"\ndisplayName="Multi NeoForge"\n',
    )

    inventory = inspect_project_inventory(root)

    modules = {item.module_id: item for item in inventory.modules}
    assert set(modules) == {":", ":common", ":fabric", ":neoforge"}
    assert modules[":fabric"].path == "loaders/fabric"
    assert modules[":fabric"].depends_on_modules == (":common",)
    assert modules[":neoforge"].depends_on_modules == (":common",)
    assert inventory.target.loaders == ("fabric", "neoforge")
    provides = {value for component in inventory.components for value in component.provides}
    assert "symbol:example.common.CommonHooks" in provides
    assert "symbol:example.common.CommonHooks#initialize" in provides
    assert "metadata:fabric:multi_fabric" in provides
    assert "metadata:neoforge:multi_neoforge" in provides


def test_archive_inventory_uses_safe_importer_and_never_executes_build_script(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "must-not-exist.txt"
    archive = tmp_path / "existing.zip"
    build = f'new File("{marker.as_posix()}").text = "executed"\n'
    metadata = {
        "schemaVersion": 1,
        "id": "archived_mod",
        "version": "1.0.0",
        "entrypoints": {"main": ["example.ArchivedMod"]},
        "depends": {"minecraft": "1.20.1"},
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("project/settings.gradle", "rootProject.name='archived'\n")
        bundle.writestr("project/build.gradle", build)
        bundle.writestr("project/src/main/resources/fabric.mod.json", json.dumps(metadata))
        bundle.writestr("project/src/main/java/example/ArchivedMod.java", "package example; public class ArchivedMod {}\n")

    inventory = inspect_existing_archive_inventory(archive)

    assert not marker.exists()
    assert inventory.source_kind == "archive"
    assert inventory.source_sha256.startswith("sha256:")
    assert inventory.imported_source_snapshot_sha256.startswith("sha256:")
    assert inventory.root_name == "archived"
    assert inventory.target.minecraft_versions == ("1.20.1",)
    assert inventory.target.loaders == ("fabric",)
    assert inventory.metadata[0].mod_id == "archived_mod"
    inventory.validate()
