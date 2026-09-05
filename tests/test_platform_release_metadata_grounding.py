from __future__ import annotations

import pytest

from minecraft_mod_ai import platform_live_discovery as live
from minecraft_mod_ai.platform_catalog import PlatformAdapter


def _adapter(release_metadata_url: str) -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id="fabric_live_test",
        edition="java",
        loader="fabric",
        minecraft_version="26.1.2",
        java_version="25",
        yarn_mappings="",
        mappings_kind="",
        mappings_version="",
        fabric_loader="0.18.4",
        fabric_api="0.140.2+26.1",
        fabric_loom="1.14.10",
        gradle="9.2.1",
        gradle_sha256="a" * 64,
        data_pack_version="101.1",
        resource_pack_version="84.0",
        resource_pack_format=84,
        release_metadata_url=release_metadata_url,
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


@pytest.mark.parametrize(
    "url",
    (
        "https://piston-meta.mojang.com/v1/packages/deadbeef/26.1.2.json",
        "https://launcher.mojang.com/v1/objects/deadbeef/26.1.2.json",
        "https://feedback.minecraft.net/hc/en-us/articles/123-release",
        "https://www.minecraft.net/en-us/article/minecraft-java-edition-26-1-2",
    ),
)
def test_platform_adapter_accepts_only_supported_official_metadata_hosts(url: str) -> None:
    _adapter(url).validate()


def test_platform_adapter_rejects_unofficial_metadata_host() -> None:
    with pytest.raises(ValueError, match="official Minecraft/Mojang metadata URL"):
        _adapter("https://example.invalid/piston-meta.mojang.com/26.1.2.json").validate()


def test_mojang_pack_metadata_is_primary_and_skips_human_article_network(monkeypatch) -> None:
    live._official_pack_versions.cache_clear()
    monkeypatch.setattr(
        live,
        "_mojang_pack_versions",
        lambda version: ("101.1", "84.0"),
    )
    monkeypatch.setattr(
        live,
        "_mojang_target_url",
        lambda version: "https://piston-meta.mojang.com/v1/packages/hash/26.1.2.json",
    )

    def forbidden(*args, **kwargs):
        pytest.fail("human release article path must not run when Mojang metadata succeeds")

    monkeypatch.setattr(live, "_feedback_pack_versions", forbidden)
    monkeypatch.setattr(live, "_fetch", forbidden)
    try:
        assert live._official_pack_versions("26.1.2") == (
            "101.1",
            "84.0",
            "https://piston-meta.mojang.com/v1/packages/hash/26.1.2.json",
        )
    finally:
        live._official_pack_versions.cache_clear()


def test_feedback_is_fallback_only_after_mojang_pack_failure(monkeypatch) -> None:
    live._official_pack_versions.cache_clear()

    def fail_mojang(version: str):
        raise live.PlatformDiscoveryError("simulated Mojang outage")

    monkeypatch.setattr(live, "_mojang_pack_versions", fail_mojang)
    monkeypatch.setattr(
        live,
        "_feedback_pack_versions",
        lambda version: (
            "61",
            "46",
            "https://feedback.minecraft.net/hc/en-us/articles/release",
        ),
    )
    try:
        assert live._official_pack_versions("1.21.4") == (
            "61",
            "46",
            "https://feedback.minecraft.net/hc/en-us/articles/release",
        )
    finally:
        live._official_pack_versions.cache_clear()
