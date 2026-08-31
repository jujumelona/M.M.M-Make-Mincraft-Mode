from __future__ import annotations

import sys
from dataclasses import replace
from functools import wraps
from pathlib import Path

import pytest

_TEST_MINECRAFT_VERSION = "mmm-test-target"
_TEST_LOADER = "fabric"


def _synthetic_test_adapter(version: str = _TEST_MINECRAFT_VERSION):
    """Return a deterministic non-release receipt used only by unit-test scaffolds.

    The synthetic version is deliberately not a real Minecraft release. This keeps
    tests that exercise source/catalog mechanics independent from a historical
    production target while production discovery remains authoritative. The test
    receipt advertises deterministic module capabilities directly so unit-only
    generators exercise capability routing without impersonating a historical target.
    """

    from minecraft_mod_ai.complete_spec import MODULE_KINDS
    from minecraft_mod_ai.platform_catalog import PlatformAdapter

    normalized = str(version).strip() or _TEST_MINECRAFT_VERSION
    mappings_version = f"{normalized}+test-mappings"
    return PlatformAdapter(
        adapter_id="fabric_unit_test_" + "_".join(part for part in normalized.replace("-", "_").split(".")),
        edition="java",
        loader=_TEST_LOADER,
        minecraft_version=normalized,
        java_version="21",
        yarn_mappings=mappings_version,
        mappings_kind="yarn",
        mappings_version=mappings_version,
        fabric_loader="test-loader",
        fabric_api="test-api",
        fabric_loom="test-loom",
        gradle="8.10.2",
        gradle_sha256="0" * 64,
        data_pack_version="1",
        resource_pack_version="1",
        resource_pack_format=1,
        release_metadata_url="https://www.minecraft.net/test-fixture/mmm-test-target",
        source_api_family="fabric_reviewed_test_template",
        deterministic_module_kinds=frozenset((
            *MODULE_KINDS,
            "system-pack:quest-system",
            "system-pack:class-skill-system",
            "system-pack:economy-shop",
            "system-pack:gui-networking",
            "system-pack:party-guild",
            "geckolib:entity",
            "geckolib:version:4.8.2",
        )),
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


def _complete_partial_test_target(target):
    """Model the host provider receipt for legacy unit tests of unrelated contracts.

    Production now requires a complete executable target receipt before semantic
    implementation planning. A small set of older tests intentionally construct the
    planning layer directly and therefore bypass the normal provider-resolution owner.
    For those modules only, fill missing receipt metadata with deterministic synthetic
    evidence while preserving every explicitly supplied coordinate, including invalid
    values that a test may be exercising.
    """

    if not isinstance(target, dict):
        return target
    version = str(target.get("minecraft_version") or "").strip()
    loader = str(target.get("loader") or "").strip()
    if not version or not loader:
        return target
    completed = dict(target)
    mappings_version = str(
        completed.get("mappings_version")
        or completed.get("yarn_mappings")
        or f"{version}+test-mappings"
    )
    completed.setdefault("java_version", 21)
    completed.setdefault("mappings_kind", "yarn")
    completed.setdefault("mappings_version", mappings_version)
    completed.setdefault("yarn_mappings", mappings_version)
    completed.setdefault("fabric_loader", "test-loader")
    completed.setdefault("fabric_api", "test-api")
    completed.setdefault("fabric_loom", "test-loom")
    completed.setdefault("gradle", "8.10.2")
    completed.setdefault("gradle_sha256", "0" * 64)
    completed.setdefault("data_pack_version", "1")
    completed.setdefault("resource_pack_version", "1")
    completed.setdefault("resource_pack_format", 1)
    completed.setdefault(
        "release_metadata_url",
        "https://www.minecraft.net/test-fixture/mmm-test-target",
    )
    return completed


@pytest.fixture
def synthetic_platform_lock():
    """Resolved non-release target for version-independent generation tests."""

    return _platform_lock_from_adapter(_synthetic_test_adapter())


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
        normalized = str(version).strip()
        if not normalized:
            raise ValueError("Test platform target must be explicit.")
        if normalized == _TEST_MINECRAFT_VERSION:
            return synthetic_adapter
        return _synthetic_test_adapter(normalized)

    monkeypatch.setitem(
        platform_catalog._PROVIDERS,
        _TEST_LOADER,
        PlatformProvider(
            loader=_TEST_LOADER,
            provider_id=production_provider.provider_id,
            discover_versions=lambda limit=32: (_TEST_MINECRAFT_VERSION,)[: max(1, int(limit))],
            resolve=resolve_test_or_production,
        ),
    )

    # Runtime-manager unit tests receive a synthetic run-scoped profile instead of
    # reviving a repository-level production default target.
    import json

    from minecraft_mod_ai import runtime_manager

    # Runtime helpers must live outside a project root that generator tests expect empty.
    fake_java = tmp_path.parent / f"{tmp_path.name}-fake-java-21"
    fake_java.write_text(
        "#!" + sys.executable + "\n"
        "import sys\n"
        "print('openjdk version \"21.0.0\"', file=sys.stderr)\n",
        encoding="utf-8",
    )
    fake_java.chmod(0o755)

    runtime_config = tmp_path.parent / f"{tmp_path.name}-runtime-profiles.yaml"
    runtime_config.write_text(
        json.dumps(
            {
                "schema_version": "mmm/runtime-profiles-v1",
                "profiles": {
                    "fabric_target_disposable": {
                        "minecraft_version": synthetic_adapter.minecraft_version,
                        "loader": synthetic_adapter.loader,
                        "java_project_version": int(synthetic_adapter.java_version),
                        "server_java_command": str(fake_java),
                        "server_memory_mb": 512,
                        "server_launcher_relative": "runtime/server.jar",
                        "client_command_env": "MMM_MINECRAFT_CLIENT_COMMAND_JSON",
                        "allowed_server_commands": [
                            "^list$",
                            "^stop$",
                            "^say [A-Za-z0-9 _.,!?-]{1,120}$",
                            "^gametest runall$",
                        ],
                        "startup_ready_patterns": ["Done"],
                        "disposable_only": True,
                        "eula_must_be_explicitly_accepted": True,
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    original_config_resolver = runtime_manager.resolve_config_path

    def test_config_path(name: str):
        if name == "runtime_profiles.yaml":
            return runtime_config
        return original_config_resolver(name)

    monkeypatch.setattr(runtime_manager, "resolve_config_path", test_config_path)

    original_generate = FabricProjectGenerator.generate

    @wraps(original_generate)
    def generate_with_explicit_test_target(self, spec, root):
        if spec.platform.is_unresolved():
            spec = replace(spec, platform=_platform_lock_from_adapter(synthetic_adapter))
        return original_generate(self, spec, root)

    monkeypatch.setattr(
        FabricProjectGenerator,
        "generate",
        generate_with_explicit_test_target,
    )

    # Direct evidence-planning tests predate the complete target receipt barrier and
    # bypass the host platform resolver entirely. Complete only those legacy fixtures;
    # dedicated target-grounding tests continue to exercise the strict production gate.
    legacy_partial_target_modules = {
        "test_evidence_first_planning",
        "test_evidence_first_session_integration",
        "test_planner_structural_repair_contract",
        "test_resource_asset_backend_contract",
        "test_target_snapshot_hardening",
    }
    if request.module.__name__ in legacy_partial_target_modules:
        from minecraft_mod_ai import evidence_first_planning as planning

        original_target_decision = planning._target_decision

        def target_decision_with_synthetic_receipt(game_design, target_decision=None):
            design = dict(game_design) if isinstance(game_design, dict) else game_design
            decision = dict(target_decision) if isinstance(target_decision, dict) else target_decision
            if isinstance(decision, dict):
                if isinstance(decision.get("coordinates"), dict):
                    decision["coordinates"] = _complete_partial_test_target(decision["coordinates"])
                elif isinstance(decision.get("target"), dict):
                    decision["target"] = _complete_partial_test_target(decision["target"])
                elif "minecraft_version" in decision or "loader" in decision:
                    decision = _complete_partial_test_target(decision)
            elif isinstance(design, dict):
                selection = design.get("_platform_selection")
                if isinstance(selection, dict):
                    selection_copy = dict(selection)
                    target = selection_copy.get("target")
                    if isinstance(target, dict):
                        selection_copy["target"] = _complete_partial_test_target(target)
                        design["_platform_selection"] = selection_copy
            return original_target_decision(design, decision)

        monkeypatch.setattr(
            planning,
            "_target_decision",
            target_decision_with_synthetic_receipt,
        )

    # These unit tests exercise technology semantics directly. Supply an explicit
    # executable fixture target instead of depending on a production-wide default.
    if request.module.__name__ == "test_technology_radar":
        adapter = _synthetic_test_adapter()
        target = {
            "edition": adapter.edition,
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
            "java_version": adapter.java_version,
            "fabric_loader": adapter.fabric_loader,
            "fabric_api": adapter.fabric_api,
        }
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

        adapter = _synthetic_test_adapter()
        target = {
            "edition": adapter.edition,
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
            "java_version": adapter.java_version,
            "fabric_loader": adapter.fabric_loader,
            "fabric_api": adapter.fabric_api,
        }
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
        from minecraft_mod_ai import central_research

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
            return replace(adapter, yarn_mappings=mapping, mappings_version=mapping)

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
