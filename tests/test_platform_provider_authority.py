from __future__ import annotations

import minecraft_mod_ai.platform_selection_pipeline as selection_pipeline
from minecraft_mod_ai.host_version_catalog import host_target
from minecraft_mod_ai.platform_catalog import PlatformProvider, provider_for_loader


def test_provider_id_alone_does_not_grant_host_catalog_authority(monkeypatch):
    spoofed = PlatformProvider(
        loader="fabric",
        provider_id="host-coherent-version-catalog-v1",
        discover_versions=lambda _limit: ("1.21.11",),
        resolve=lambda version: host_target(version),
    )
    monkeypatch.setattr(selection_pipeline, "provider_for_loader", lambda _loader: spoofed)
    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "off")

    selection = selection_pipeline.resolve_platform_fail_closed("Create a Fabric mod")

    assert selection.adapter.minecraft_version == "1.21.11"
    assert selection.source == "provider_receipt_only"


def test_builtin_fabric_provider_declares_host_catalog_authority():
    assert provider_for_loader("fabric").host_authoritative is True
