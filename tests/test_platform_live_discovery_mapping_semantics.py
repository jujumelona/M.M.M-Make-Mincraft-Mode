from __future__ import annotations

import pytest

import minecraft_mod_ai.platform_live_discovery as discovery


def _discover_with_stubbed_official_metadata(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> discovery.LiveFabricTarget:
    api_version = f"0.0.1+{version}"
    monkeypatch.setattr(
        discovery,
        "discover_game_versions",
        lambda: ({"version": version, "stable": True},),
    )
    monkeypatch.setattr(
        discovery,
        "_common_platform_metadata",
        lambda: (
            "0.17.0",
            (api_version,),
            "1.11.0",
            "9.1.0",
            "a" * 64,
            ((version, "https://piston-meta.mojang.com/version.json"),),
        ),
    )
    monkeypatch.setattr(
        discovery,
        "_official_pack_versions",
        lambda _version: (
            "80",
            "70",
            "https://piston-meta.mojang.com/version.json",
        ),
    )
    monkeypatch.setattr(
        discovery,
        "_stable_java_versions",
        lambda: ((version, "25" if version.startswith("26.") else "21"),),
    )
    discovery.discover_fabric_target.cache_clear()
    try:
        return discovery.discover_fabric_target(version)
    finally:
        discovery.discover_fabric_target.cache_clear()


def test_native_target_discovery_omits_legacy_mapping_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _discover_with_stubbed_official_metadata(monkeypatch, "26.2")

    assert target.mappings_kind == ""
    assert target.mappings_version == ""


def test_pre_cutover_target_discovery_keeps_explicit_mapping_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _discover_with_stubbed_official_metadata(monkeypatch, "1.21.4")

    assert target.mappings_kind == "mojang"
    assert target.mappings_version == "mojang"
