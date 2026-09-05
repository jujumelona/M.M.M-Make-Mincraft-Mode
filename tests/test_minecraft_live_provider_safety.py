from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import platform_catalog


def test_live_fabric_provider_never_advertises_unreviewed_deterministic_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered = SimpleNamespace(
        discovery_sha256="sha256:" + "a" * 64,
        resource_pack_version="70.0",
        data_pack_version="88.0",
        minecraft_version="26.2",
        java_version="25",
        mappings_version="mojang",
        mappings_kind="mojang",
        loader_version="0.17.2",
        fabric_api_version="0.140.0+26.2",
        loom_version="1.12.0",
        gradle_version="8.14.3",
        gradle_sha256="b" * 64,
        release_metadata_url="https://www.minecraft.net/en-us/article/test-release",
    )
    monkeypatch.setattr(platform_catalog, "discover_fabric_target", lambda _version: discovered)

    adapter = platform_catalog._fabric_adapter("26.2")

    assert adapter.source_api_family == "fabric_live_ai"
    assert adapter.deterministic_module_kinds == frozenset()
