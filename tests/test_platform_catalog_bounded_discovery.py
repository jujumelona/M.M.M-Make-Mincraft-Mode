from __future__ import annotations

import pytest

import minecraft_mod_ai.platform_catalog as catalog


def _adapter(loader: str, version: str) -> catalog.PlatformAdapter:
    return catalog.PlatformAdapter(
        adapter_id=f"test_{loader}_{version}",
        edition="java",
        loader=loader,
        minecraft_version=version,
        java_version="21",
        yarn_mappings="mojang",
        mappings_kind="mojang",
        mappings_version="mojang",
        fabric_loader="test-loader",
        fabric_api="test-api",
        fabric_loom="test-loom",
        gradle="8.10",
        gradle_sha256="a" * 64,
        data_pack_version="48",
        resource_pack_version="48",
        resource_pack_format=48,
        release_metadata_url="https://piston-meta.mojang.com/v1/packages/test.json",
        source_api_family="test",
        deterministic_module_kinds=frozenset(),
    )


def _install_provider(monkeypatch: pytest.MonkeyPatch, provider: catalog.PlatformProvider) -> None:
    monkeypatch.setitem(catalog._PROVIDERS, provider.loader, provider)


def test_exact_target_bypasses_version_enumeration(monkeypatch: pytest.MonkeyPatch) -> None:
    def discover_versions(_limit: int) -> tuple[str, ...]:
        raise AssertionError("exact target resolution must not enumerate the catalogue")

    provider = catalog.PlatformProvider(
        loader="test-exact",
        provider_id="test",
        discover_versions=discover_versions,
        resolve=lambda version: _adapter("test-exact", version),
    )
    _install_provider(monkeypatch, provider)

    assert catalog.discover_target_keys(
        loader="test-exact",
        minecraft_version="1.21.1",
    ) == (("test-exact", "1.21.1"),)


def test_automatic_discovery_is_bounded_and_filters_unresolvable_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def discover_versions(limit: int) -> tuple[str, ...]:
        calls.append(limit)
        # Deliberately return more than requested to verify the registry enforces its bound.
        return ("new-bad", "new-good", "old-good", "ancient-good")

    def resolve(version: str) -> catalog.PlatformAdapter:
        if version == "new-bad":
            raise ValueError("incomplete toolchain")
        return _adapter("test-bounded", version)

    provider = catalog.PlatformProvider(
        loader="test-bounded",
        provider_id="test",
        discover_versions=discover_versions,
        resolve=resolve,
    )
    _install_provider(monkeypatch, provider)

    diagnostics: list[str] = []
    assert catalog.discover_target_keys(
        loader="test-bounded",
        limit_per_loader=2,
        diagnostics=diagnostics,
    ) == (("test-bounded", "new-good"),)
    assert calls == [2]
    assert any("new-bad" in message for message in diagnostics)
    assert all("old-good" not in message for message in diagnostics)


def test_expected_target_failure_has_no_traceback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[tuple[str, dict[str, object]]] = []

    provider = catalog.PlatformProvider(
        loader="test-log",
        provider_id="test",
        discover_versions=lambda limit: ("bad",)[:limit],
        resolve=lambda _version: (_ for _ in ()).throw(ValueError("not executable")),
    )
    _install_provider(monkeypatch, provider)
    monkeypatch.setattr(
        catalog,
        "_emit_discovery_log",
        lambda message, **kwargs: emitted.append((message, kwargs)),
    )

    with pytest.raises(ValueError, match="not executable"):
        catalog.adapter_for_target("bad", "test-log")

    assert emitted
    assert all(not kwargs.get("exc_info") for _message, kwargs in emitted)


def test_newest_adapter_skips_incomplete_newest_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve(version: str) -> catalog.PlatformAdapter:
        if version == "broken-newest":
            raise ValueError("missing dependency")
        return _adapter("test-newest", version)

    provider = catalog.PlatformProvider(
        loader="test-newest",
        provider_id="test",
        discover_versions=lambda limit: ("broken-newest", "working-next")[:limit],
        resolve=resolve,
    )
    _install_provider(monkeypatch, provider)

    adapter = catalog.newest_adapter(loader="test-newest")
    assert adapter.minecraft_version == "working-next"
