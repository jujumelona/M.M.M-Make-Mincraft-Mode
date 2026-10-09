"""No-network regression for the authored donor->project identity boundary."""
from __future__ import annotations

import pytest

from minecraft_mod_ai import source_transplant
from minecraft_mod_ai.authored_reuse_bridge import materialize_verified_authored_sources


def _source_receipt(tmp_path):
    java = tmp_path / "donor-ExampleItem.java"
    java.write_text(
        'package org.donor;\n'
        'import net.minecraft.util.Identifier;\n'
        'public final class ExampleItem {\n'
        '  static final Identifier KEY = Identifier.of("donor", "fuel");\n'
        '}\n',
        encoding="utf-8",
    )
    resource = tmp_path / "donor-en_us.json"
    resource.write_text('{"item.donor.fuel":"Fuel"}', encoding="utf-8")
    texture = tmp_path / "donor-fuel.png"
    texture.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n" + b"\\xff" * 24)
    license_file = tmp_path / "donor-license.txt"
    license_file.write_text("MIT License\nTest attribution fixture", encoding="utf-8")
    donor = {
        "capability": "fuel",
        "repository": "example/donor",
        "commit_sha": "a" * 40,
        "license_id": "MIT",
        "required_dependencies": [],
        "files": [
            {
                "source_path": "src/main/java/org/donor/ExampleItem.java",
                "path": str(java),
            },
            {
                "source_path": "src/main/resources/assets/donor/lang/en_us.json",
                "path": str(resource),
            },
            {
                "source_path": "src/main/resources/assets/donor/textures/item/fuel.png",
                "path": str(texture),
            },
            {"source_path": "LICENSE", "path": str(license_file)},
        ],
    }
    return {
        "schema_version": "mmm/reuse-materialization-v1",
        "donors": [donor],
        "count": 1,
    }


def _plan():
    return {
        "schema_version": "mmm/grounded-repository-reuse-plan-v2",
        "bound_target": {"minecraft_version": "1.20.1", "loader": "fabric"},
        "capabilities": [{"capability": "fuel", "mode": "source_transplant"}],
    }


def test_authored_transplant_requires_real_namespace_before_any_download(tmp_path, monkeypatch):
    def must_not_download(*args):
        raise AssertionError("transplant downloader must not run without mod identity")

    monkeypatch.setattr(source_transplant, "materialize_source_slices", must_not_download)
    with pytest.raises(ValueError, match="SOURCE_REUSE_TARGET_IDENTITY_REQUIRED"):
        materialize_verified_authored_sources(
            str(tmp_path / "output"), _plan(),
            minecraft_version="1.20.1", loader="fabric",
        )


def test_authored_transplant_uses_actual_mod_package_and_mod_id(tmp_path, monkeypatch):
    receipt = _source_receipt(tmp_path)
    monkeypatch.setattr(
        source_transplant, "materialize_source_slices",
        lambda root, reuse_plan: receipt,
    )
    out = tmp_path / "output"
    result = materialize_verified_authored_sources(
        str(out), _plan(),
        minecraft_version="1.20.1", loader="fabric",
        package_name="dev.example.spacemod", mod_id="spacemod",
    )
    assert result["donor_count"] == 1
    java_path = out / "src/main/java/dev/example/spacemod/ExampleItem.java"
    assert java_path.is_file()
    assert not (out / "src/main/java/org/donor/ExampleItem.java").exists()
    java = java_path.read_text(encoding="utf-8")
    assert "package dev.example.spacemod;" in java
    assert 'Identifier.of("spacemod", "fuel")' in java
    assert (out / "src/main/resources/assets/spacemod/lang/en_us.json").is_file()
    texture_path = out / "src/main/resources/assets/spacemod/textures/item/fuel.png"
    assert texture_path.read_bytes().startswith(b"\\x89PNG")
    assert not (out / "src/main/resources/assets/donor/textures/item/fuel.png").exists()
    assert (out / "src/main/resources/META-INF/mmm-third-party").is_dir()
    assert (out / ".minecraft_ai/reuse/source_provenance.json").is_file()


def test_java_package_relocation_does_not_overwrite_another_donor_source():
    from minecraft_mod_ai.reuse_adapters import PackageRelocationAdapter

    files = {
        "src/main/java/org/donor/Example.java": (
            "package org.donor; public final class Example {}"
        ),
        "src/main/java/dev/example/spacemod/Example.java": (
            "package dev.example.spacemod; public final class Example { static final int EXISTING = 1; }"
        ),
    }
    with pytest.raises(ValueError, match="SOURCE_REUSE_JAVA_PACKAGE_PATH_COLLISION"):
        PackageRelocationAdapter().apply(
            files, {"target_package": "dev.example.spacemod"}
        )


def test_resource_namespace_rewrite_rejects_target_collision():
    from minecraft_mod_ai.reuse_adapters import ModIdRewriteAdapter

    files = {
        "src/main/resources/assets/donor/lang/en_us.json": '{"item":"donor"}',
        "src/main/resources/assets/spacemod/lang/en_us.json": '{"item":"target"}',
    }
    original = files.copy()
    with pytest.raises(ValueError, match="SOURCE_REUSE_RESOURCE_NAMESPACE_COLLISION"):
        ModIdRewriteAdapter().apply(
            files, {"donor_modid": "donor", "target_modid": "spacemod"}
        )
    assert files["src/main/resources/assets/spacemod/lang/en_us.json"] == original[
        "src/main/resources/assets/spacemod/lang/en_us.json"
    ]


def test_fabric_api_migration_keeps_java_package_before_import():
    from minecraft_mod_ai.reuse_adapters import FabricApiMigrationAdapter

    files = {
        "src/main/java/example/ModItems.java": (
            "package example;\n"
            "import net.minecraft.registry.Registry;\n"
            "final class ModItems { Object items = Registry.ITEM; }\n"
        ),
    }
    migration = FabricApiMigrationAdapter()
    assert migration.can_apply(files, {"mappings": "yarn"})
    receipt = migration.apply(files, {"mappings": "yarn"})
    source = files["src/main/java/example/ModItems.java"]
    assert receipt.applied
    assert source.startswith("package example;\n")
    assert source.index("package example;") < source.index(
        "import net.minecraft.registry.Registries;"
    ) < source.index("final class ModItems")
    assert "Registries.ITEM" in source
