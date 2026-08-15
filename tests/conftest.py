from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest


_TEST_MINECRAFT_VERSION = "mmm-test-target"
_TEST_LOADER = "fabric"


def _synthetic_test_adapter(version: str = _TEST_MINECRAFT_VERSION):
    """Return a deterministic non-release receipt used only by unit-test scaffolds.

    The synthetic version is deliberately not a real Minecraft release. This keeps
    tests that exercise source/catalog mechanics independent from a historical
    production target while production discovery remains authoritative.
    """

    from minecraft_mod_ai.platform_catalog import PlatformAdapter

    normalized = str(version).strip() or _TEST_MINECRAFT_VERSION
    return PlatformAdapter(
        adapter_id="fabric_unit_test_receipt",
        edition="java",
        loader=_TEST_LOADER,
        minecraft_version=normalized,
        java_version="21",
        yarn_mappings=f"{normalized}+test-mappings",
        fabric_loader="test-loader",
        fabric_api="test-api",
        fabric_loom="test-loom",
        gradle="test-gradle",
        gradle_sha256="sha256:" + "0" * 64,
        resource_pack_format=0,
        source_api_family="fabric_unit_test",
        deterministic_module_kinds=frozenset(),
    )


def _platform_lock_from_adapter(adapter):
    from minecraft_mod_ai.spec import PlatformLock

    return PlatformLock(
        edition=adapter.edition,
        loader=adapter.loader,
        minecraft_version=adapter.minecraft_version,
        java_version=adapter.java_version,
        yarn_mappings=adapter.yarn_mappings,
        fabric_loader=adapter.fabric_loader,
        fabric_api=adapter.fabric_api,
        fabric_loom=adapter.fabric_loom,
        gradle=adapter.gradle,
    )


def _fabric_1201_target() -> dict[str, str]:
    """Compatibility fixture for legacy tests that explicitly exercise that target."""

    from minecraft_mod_ai.platform_catalog import adapter_for_target

    adapter = adapter_for_target("1.20.1", "fabric")
    return {
        "edition": adapter.edition,
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "mappings": adapter.yarn_mappings,
        "java_version": adapter.java_version,
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
    }


@pytest.fixture(autouse=True)
def _isolate_test_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Keep unit tests deterministic while production caches remain durable."""

    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "off")
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path / "planner-checkpoints"))
    monkeypatch.setenv("MMM_PLANNER_TRACE_DIR", str(tmp_path / "planner-traces"))
    monkeypatch.setenv(
        "MMM_RESEARCH_CHECKPOINT_ROOT",
        str(tmp_path / "research-checkpoints"),
    )
    monkeypatch.setenv(
        "MMM_RESEARCH_DOCUMENT_DIR",
        str(tmp_path / "research-evidence"),
    )

    # Source/catalog tests must not acquire a historical Minecraft default merely
    # because they need a generated project directory. Install one synthetic target
    # that is never advertised by discovery and inject it only when the caller left
    # the platform intentionally unresolved.
    from minecraft_mod_ai import platform_catalog
    from minecraft_mod_ai.generator import FabricProjectGenerator
    from minecraft_mod_ai.platform_catalog import PlatformProvider

    production_provider = platform_catalog.provider_for_loader(_TEST_LOADER)
    synthetic_adapter = _synthetic_test_adapter()

    def resolve_test_or_production(version: str):
        if str(version).strip() == _TEST_MINECRAFT_VERSION:
            return synthetic_adapter
        return production_provider.resolve(version)

    monkeypatch.setitem(
        platform_catalog._PROVIDERS,
        _TEST_LOADER,
        PlatformProvider(
            loader=_TEST_LOADER,
            provider_id=production_provider.provider_id,
            discover_versions=production_provider.discover_versions,
            resolve=resolve_test_or_production,
        ),
    )

    original_generate = FabricProjectGenerator.generate

    def generate_with_explicit_test_target(self, spec, root):
        if spec.platform.is_unresolved():
            spec = replace(spec, platform=_platform_lock_from_adapter(synthetic_adapter))
        return original_generate(self, spec, root)

    monkeypatch.setattr(
        FabricProjectGenerator,
        "generate",
        generate_with_explicit_test_target,
    )

    # These unit tests exercise technology semantics directly. Supply an explicit
    # executable fixture target instead of depending on a production-wide default.
    if request.module.__name__ == "test_technology_radar":
        target = _fabric_1201_target()
        original = request.module.build_technology_radar

        def build_with_explicit_test_target(*args, **kwargs):
            kwargs.setdefault("target", target)
            return original(*args, **kwargs)

        monkeypatch.setattr(
            request.module,
            "build_technology_radar",
            build_with_explicit_test_target,
        )

    # This test module intentionally replaces GameDesignPlanner.plan, which bypasses
    # the production platform-selection owner. Bind an explicit target only at that
    # mocked boundary.
    if request.module.__name__ == "test_complete_planner_technology_sidecar":
        import minecraft_mod_ai.complete_planner as planner_module

        target = _fabric_1201_target()
        original_collect = planner_module.collect_technology_radar

        def collect_with_explicit_test_target(*args, **kwargs):
            kwargs.setdefault("target", target)
            return original_collect(*args, **kwargs)

        monkeypatch.setattr(
            planner_module,
            "collect_technology_radar",
            collect_with_explicit_test_target,
        )

    # Legacy central-research tests assert old provider receipts. Keep that fixture
    # local to the legacy test module; production retrieval continues to use live
    # provider receipts and has no historical mapping fallback.
    if request.module.__name__ == "test_central_research":
        import minecraft_mod_ai.central_research as central_research

        original_adapter_for_target = central_research.adapter_for_target

        def legacy_research_adapter(version: str, loader: str):
            normalized = str(version).strip()
            if str(loader).strip().casefold() != "fabric":
                return original_adapter_for_target(version, loader)
            adapter = _synthetic_test_adapter(normalized)
            mapping = {
                "1.20.1": "1.20.1+build.1",
                "1.21.1": "1.21.1+build.3",
            }.get(normalized, adapter.yarn_mappings)
            return replace(adapter, yarn_mappings=mapping)

        monkeypatch.setattr(
            central_research,
            "adapter_for_target",
            legacy_research_adapter,
        )

    # Legacy ecosystem unit tests are exact-target tests. Keep their intent explicit
    # while product defaults become platform-neutral. Dedicated dynamic-target tests
    # exercise the new targetless path without this fixture.
    if request.module.__name__ == "test_ecosystem_discovery":
        from minecraft_mod_ai import ecosystem_discovery as ecosystem

        original_search = ecosystem.EcosystemDiscoveryClient.search
        original_inspect = ecosystem.EcosystemDiscoveryClient.inspect_modrinth_project

        def search_with_explicit_test_target(self, *args, **kwargs):
            if str(kwargs.get("target_profile", "minecraft_mod")) == "minecraft_mod":
                kwargs.setdefault("minecraft_version", "1.20.1")
                kwargs.setdefault("loader", "fabric")
            return original_search(self, *args, **kwargs)

        def inspect_with_explicit_test_target(self, *args, **kwargs):
            kwargs.setdefault("minecraft_version", "1.20.1")
            kwargs.setdefault("loader", "fabric")
            return original_inspect(self, *args, **kwargs)

        monkeypatch.setattr(
            ecosystem.EcosystemDiscoveryClient,
            "search",
            search_with_explicit_test_target,
        )
        monkeypatch.setattr(
            ecosystem.EcosystemDiscoveryClient,
            "inspect_modrinth_project",
            inspect_with_explicit_test_target,
        )
