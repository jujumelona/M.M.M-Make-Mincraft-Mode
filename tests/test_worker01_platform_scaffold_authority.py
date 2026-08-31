from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from minecraft_mod_ai.platform_catalog import PlatformAdapter
from minecraft_mod_ai import verified_scaffold_registry as scaffold


def _adapter(
    *,
    version: str = "26.2",
    loader: str = "fabric",
    gradle: str = "9.5.1",
    gradle_sha256: str = "a" * 64,
    mappings_kind: str = "mojang",
    mappings_version: str = "mojang",
) -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id=f"{loader}-{version}-receipt",
        edition="java",
        loader=loader,
        minecraft_version=version,
        java_version="21",
        yarn_mappings=mappings_version,
        mappings_kind=mappings_kind,
        mappings_version=mappings_version,
        fabric_loader="0.18.0",
        fabric_api=f"0.140.0+{version}",
        fabric_loom="1.17-SNAPSHOT",
        gradle=gradle,
        gradle_sha256=gradle_sha256,
        data_pack_version="100",
        resource_pack_version="80",
        resource_pack_format=80,
        release_metadata_url="https://piston-meta.mojang.com/v1/packages/test.json",
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


def test_adapter_path_preserves_exact_receipt_without_catalog_reresolution(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(
        scaffold,
        "adapter_for_target",
        lambda *_args, **_kwargs: pytest.fail("adapter path must not re-resolve target"),
    )

    template = scaffold.get_verified_scaffold_template_for_adapter(adapter)

    assert template.adapter_id == adapter.adapter_id
    assert template.loader == adapter.loader
    assert template.minecraft_version == adapter.minecraft_version
    assert template.gradle_version == adapter.gradle
    assert template.distribution_sha256 == adapter.gradle_sha256


def test_compatibility_wrapper_resolves_once_then_delegates(monkeypatch):
    adapter = _adapter()
    calls: list[tuple[str, str]] = []

    def resolve(version: str, loader: str) -> PlatformAdapter:
        calls.append((version, loader))
        return adapter

    monkeypatch.setattr(scaffold, "adapter_for_target", resolve)

    template = scaffold.get_verified_scaffold_template("fabric", "26.2")

    assert calls == [("26.2", "fabric")]
    assert template.adapter_id == adapter.adapter_id


def test_invalid_provider_metadata_is_rejected_before_scaffold_acceptance():
    adapter = replace(_adapter(), gradle_sha256="not-a-sha")

    with pytest.raises(ValueError):
        scaffold.validate_scaffold_buildability(adapter)


def test_unsupported_loader_cannot_bypass_executable_provider(monkeypatch):
    adapter = _adapter(loader="neoforge")
    monkeypatch.setattr(scaffold, "executable_loaders", lambda: ("fabric",))

    with pytest.raises(scaffold.UnsupportedTargetSpecificationError):
        scaffold.validate_scaffold_buildability(adapter)


def test_is_target_supported_fails_closed_when_provider_resolution_fails(monkeypatch):
    def fail(_version: str, _loader: str) -> PlatformAdapter:
        raise ValueError("offline")

    monkeypatch.setattr(scaffold, "adapter_for_target", fail)

    assert scaffold.is_target_supported("fabric", "26.2") is False


def test_loom_plugin_boundary_matches_official_fabric_template():
    unobfuscated = scaffold.get_verified_scaffold_template_for_adapter(_adapter(version="26.1"))
    remapped = scaffold.get_verified_scaffold_template_for_adapter(_adapter(version="1.21.11"))

    assert unobfuscated.loom_plugin_id == "net.fabricmc.fabric-loom"
    assert "officialMojangMappings" not in unobfuscated.build_gradle
    assert "implementation 'net.fabricmc:fabric-loader:" in unobfuscated.build_gradle

    assert remapped.loom_plugin_id == "net.fabricmc.fabric-loom-remap"
    assert "mappings loom.officialMojangMappings()" in remapped.build_gradle
    assert "modImplementation 'net.fabricmc:fabric-loader:" in remapped.build_gradle


def test_distribution_integrity_is_owned_by_adapter_not_static_version_table():
    adapter = _adapter(gradle="99.123", gradle_sha256="c" * 64)

    template = scaffold.get_verified_scaffold_template_for_adapter(adapter)

    assert template.gradle_version == "99.123"
    assert template.distribution_sha256 == "c" * 64
    assert template.distribution_url.endswith("/gradle-99.123-bin.zip")


def test_embedded_adapter_context_does_not_reresolve(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(
        scaffold,
        "adapter_for_target",
        lambda *_args, **_kwargs: pytest.fail("embedded adapter must be reused"),
    )

    result = scaffold._adapter_from_target_context(
        {
            "loader": "fabric",
            "minecraft_version": "26.2",
            "platform_adapter": adapter,
        }
    )

    assert result is adapter


def test_embedded_adapter_identity_mismatch_fails_closed():
    adapter = _adapter()

    with pytest.raises(scaffold.UnsupportedTargetSpecificationError):
        scaffold._adapter_from_target_context(
            {
                "loader": "fabric",
                "minecraft_version": "1.21.11",
                "platform_adapter": adapter,
            }
        )


def test_no_static_minecraft_target_support_matrix_remains():
    source = Path(scaffold.__file__).read_text(encoding="utf-8")

    assert "SUPPORTED_TARGET_SPECS" not in source
    assert "(\"neoforge\"," not in source
    assert "(\"forge\"," not in source
