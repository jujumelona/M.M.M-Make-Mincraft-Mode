from __future__ import annotations

from pathlib import Path

import pytest


def _fabric_1201_target() -> dict[str, str]:
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
