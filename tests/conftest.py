from __future__ import annotations

import sys
from dataclasses import replace
from functools import wraps
from pathlib import Path

import pytest

_TEST_MINECRAFT_VERSION = "1.21.11+mmm-test"
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
        adapter_id="fabric_unit_test_" + "_".join(
            part
            for part in normalized.replace("-", "_").replace("+", "_").split(".")
        ),
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
        release_metadata_url="https://www.minecraft.net/test-fixture/1.21.11+mmm-test",
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
    from minecraft_mod_ai.platform_resolver import lock_from_adapter

    return lock_from_adapter(adapter)


def _uses_native_naming(version: str) -> bool:
    base = str(version).strip().split("+", 1)[0].split("-", 1)[0]
    parts = base.split(".")
    try:
        major = int(parts[0]) if parts else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return False
    return major > 26 or (major == 26 and minor >= 1)


def _complete_partial_test_target(target):
    """Model the host provider receipt for legacy unit tests of unrelated contracts.

    Production now requires a complete executable target receipt before semantic
    implementation planning. A small set of older tests intentionally construct the
    planning layer directly and therefore bypass the normal provider-resolution owner.
    For those modules only, fill missing receipt metadata with deterministic synthetic
    evidence while preserving every explicitly supplied coordinate, including invalid
    values that a test may be exercising. Minecraft 26.1+ uses native names, so missing
    legacy mapping coordinates stay empty and the Java baseline is 25 rather than 21.
    """

    if not isinstance(target, dict):
        return target
    version = str(target.get("minecraft_version") or "").strip()
    loader = str(target.get("loader") or "").strip()
    if not version or not loader:
        return target
    completed = dict(target)
    native_naming = _uses_native_naming(version)
    if native_naming:
        completed.setdefault("java_version", 25)
        completed.setdefault("mappings_kind", "")
        completed.setdefault("mappings_version", "")
        completed.setdefault("yarn_mappings", "")
    else:
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
        "https://www.minecraft.net/test-fixture/1.21.11+mmm-test",
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
    # for intentionally unresolved test scaffolds without replacing real-release
    # provider semantics elsewhere in the suite.
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
        return production_provider.resolve(normalized)

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
            return original_target_decision(design, target_decision=decision)

        monkeypatch.setattr(planning, "_target_decision", target_decision_with_synthetic_receipt)

    yield
