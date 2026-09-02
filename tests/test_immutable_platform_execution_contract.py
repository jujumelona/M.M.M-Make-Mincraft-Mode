from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from minecraft_mod_ai import platform_catalog, platform_resolver
from minecraft_mod_ai.fabric_immutable_rebind_contract import _rebind_scaffold
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.platform_catalog import PlatformAdapter
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec, PlatformLock, SpecValidationError


def _adapter() -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id="fabric_live_27_0_immutable_test",
        edition="java",
        loader="fabric",
        minecraft_version="27.0",
        java_version="25",
        yarn_mappings="mojang",
        mappings_kind="mojang",
        mappings_version="mojang",
        fabric_loader="0.20.0",
        fabric_api="0.200.0+27.0",
        fabric_loom="1.20-SNAPSHOT",
        gradle="9.7",
        gradle_sha256="a" * 64,
        data_pack_version="100.0",
        resource_pack_version="100.0",
        resource_pack_format=100,
        release_metadata_url="https://www.minecraft.net/en-us/article/minecraft-java-edition-27-0",
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset({"item"}),
    )


def test_complete_execution_lock_reconstructs_without_live_provider(monkeypatch, tmp_path) -> None:
    adapter = _adapter()
    lock = platform_resolver.lock_from_adapter(adapter)
    assert lock.adapter_id == adapter.adapter_id
    assert lock.gradle_sha256 == adapter.gradle_sha256
    assert lock.gradle_distribution_url.endswith("gradle-9.7-bin.zip")
    assert lock.resource_pack_format == 100

    def explode(*_args, **_kwargs):
        raise AssertionError("live target provider must not run after approval")

    monkeypatch.setattr(platform_catalog, "adapter_for_target", explode)

    restored = platform_catalog.adapter_for_lock_values(lock)
    assert restored == adapter

    project = tmp_path / "project"
    lock_path = project / ".minecraft_ai" / "platform-lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps(asdict(lock)), encoding="utf-8")
    from_project = platform_catalog.adapter_from_project(project)
    assert from_project == adapter


def test_generator_persists_full_approval_bound_platform_receipt(monkeypatch, tmp_path) -> None:
    adapter = _adapter()
    lock = platform_resolver.lock_from_adapter(adapter)

    def explode(*_args, **_kwargs):
        raise AssertionError("generator attempted live target re-resolution")

    monkeypatch.setattr(platform_catalog, "adapter_for_target", explode)

    spec = ModSpec(
        mod_id="immutable_probe",
        mod_name="Immutable Probe",
        package_name="example.immutableprobe",
        version="1.0.0",
        summary="receipt test",
        contents=(
            ContentSpec(
                content_id="probe_item",
                kind=ContentKind.ITEM,
                display_name_en="Probe Item",
                display_name_ko="프로브 아이템",
            ),
        ),
        platform=lock,
    )
    root = tmp_path / "generated"
    FabricProjectGenerator().generate(spec, root)

    raw = json.loads(
        (root / ".minecraft_ai" / "platform-lock.json").read_text(encoding="utf-8")
    )
    assert raw["adapter_id"] == adapter.adapter_id
    assert raw["gradle_sha256"] == adapter.gradle_sha256
    assert raw["release_metadata_url"] == adapter.release_metadata_url
    assert raw["resource_pack_format"] == adapter.resource_pack_format


def test_incomplete_execution_lock_fails_closed_without_live_rediscovery(monkeypatch) -> None:
    legacy = PlatformLock(
        edition="java",
        loader="fabric",
        minecraft_version="27.0",
        java_version="25",
        yarn_mappings="mojang",
        fabric_loader="0.20.0",
        fabric_api="0.200.0+27.0",
        fabric_loom="1.20-SNAPSHOT",
        gradle="9.7",
    )

    def explode(*_args, **_kwargs):
        raise AssertionError("legacy execution lock attempted live provider discovery")

    monkeypatch.setattr(platform_catalog, "adapter_for_target", explode)
    with pytest.raises(SpecValidationError, match="incomplete"):
        platform_catalog.adapter_for_lock_values(legacy)


def test_official_fabric_scaffold_is_rebound_to_approved_receipt(tmp_path) -> None:
    from minecraft_mod_ai import fabric_official_template_provider as provider

    adapter = _adapter()
    root = tmp_path / "fabric"
    (root / "gradle/wrapper").mkdir(parents=True)
    (root / "gradle.properties").write_text(
        "\n".join(
            (
                "minecraft_version=27.0",
                "loader_version=0.19.9",
                "fabric_version=0.199.0+27.0",
                "loom_version=1.19-SNAPSHOT",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "build.gradle").write_text(
        """plugins {
    id 'fabric-loom' version '1.19-SNAPSHOT'
}

dependencies {
    modImplementation \"net.fabricmc:fabric-loader:0.19.9\"
    modImplementation \"net.fabricmc.fabric-api:fabric-api:0.199.0+27.0\"
}

java {
    toolchain.languageVersion = JavaLanguageVersion.of(21)
}
""",
        encoding="utf-8",
    )
    (root / "gradle/wrapper/gradle-wrapper.properties").write_text(
        "distributionUrl=https\\://services.gradle.org/distributions/gradle-9.6-bin.zip\n",
        encoding="utf-8",
    )

    verified = _rebind_scaffold(provider, root, adapter)

    assert verified == {
        "minecraft_version": "27.0",
        "loader_version": "0.20.0",
        "fabric_api": "0.200.0+27.0",
        "loom": "1.20-SNAPSHOT",
        "gradle": "9.7",
        "java": "25",
    }
    properties = provider._read_properties(root / "gradle.properties")
    assert properties["loader_version"] == adapter.fabric_loader
    assert properties["fabric_version"] == adapter.fabric_api
    assert properties["loom_version"] == adapter.fabric_loom
    build = (root / "build.gradle").read_text(encoding="utf-8")
    assert "0.19.9" not in build
    assert "0.199.0+27.0" not in build
    assert "1.19-SNAPSHOT" not in build
    assert "JavaLanguageVersion.of(25)" in build
    wrapper = (root / "gradle/wrapper/gradle-wrapper.properties").read_text(encoding="utf-8")
    assert "gradle-9.7-bin.zip" in wrapper
    assert f"distributionSha256Sum={adapter.gradle_sha256}" in wrapper
