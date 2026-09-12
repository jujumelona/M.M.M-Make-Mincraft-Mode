from dataclasses import replace
import json
from pathlib import Path

import pytest

from minecraft_mod_ai import platform_catalog as catalog
from minecraft_mod_ai.generator import FabricProjectGenerator, GenerationError
from minecraft_mod_ai.knowledge import evidence_for_target
from minecraft_mod_ai.platform_resolver import lock_from_adapter, resolve_platform
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec, PlatformLock, SpecValidationError
from minecraft_mod_ai.resolved_version_context import VersionContextError
from test_resolved_version_context import target_fixture


@pytest.fixture(autouse=True)
def host_catalog(tmp_path, monkeypatch):
    bundles = []
    for version in ("1.21.1", "1.20.1", "27.0"):
        native = version == "27.0"
        target = replace(target_fixture(), adapter_id="host_fixture_" + version,
                         minecraft_version=version, java_version="25" if native else "21",
                         yarn_mappings="" if native else "mojang",
                         mappings_kind="" if native else "mojang",
                         mappings_version="" if native else "mojang")
        bundles.append(target.version_context.to_dict())
    path = tmp_path / "host-catalog.json"
    path.write_text(json.dumps({"schema_version": "mmm/host-version-catalog-v1",
                               "auto_context_id": bundles[0]["context_id"], "bundles": bundles}))
    monkeypatch.setenv("MMM_VERSION_BUNDLE_CATALOG", str(path))
    monkeypatch.setitem(catalog._PROVIDERS, "fabric", catalog.PlatformProvider(
        "fabric", "host-coherent-version-catalog-v1", catalog._fabric_versions,
        catalog._fabric_adapter, host_authoritative=True))


def _fabric_1211():
    return catalog.adapter_for_target("1.21.1", "fabric")


def _simple_spec(adapter):
    return ModSpec(mod_id="target_test", mod_name="Target Test", package_name="generated.targettest",
                   version="1.0.0", summary="Synthetic target test",
                   contents=(ContentSpec(kind=ContentKind.ITEM, content_id="target_crystal",
                                         display_name_en="Target Crystal", display_name_ko="테스트", recipe=True),),
                   platform=lock_from_adapter(adapter))


def test_supported_versions_are_host_catalog_not_source_allowlist():
    assert catalog.supported_minecraft_versions(loader="fabric")[:2] == ("1.21.1", "1.20.1")


def test_future_version_uses_whole_host_bundle():
    selected = catalog.adapter_for_target("27.0", "fabric")
    assert selected.minecraft_version == "27.0"
    assert selected.version_context.facts["host_revision"] == "fixture-a"


def test_public_resolver_never_asks_model_for_coordinates():
    class Router:
        def generate_text(self, *args, **kwargs):
            pytest.fail("model must not select version")
    result = resolve_platform("Create a mod", router=Router())
    assert result.adapter.minecraft_version == "1.21.1"
    assert result.source == "host_coherent_bundle"


def test_pinned_is_exact_not_optimizer_hint():
    selected = resolve_platform("Minecraft 27.0 Fabric item")
    assert selected.adapter.minecraft_version == "27.0"
    assert selected.explicit_version is True
    with pytest.raises(VersionContextError, match="UNSUPPORTED_MINECRAFT_VERSION"):
        resolve_platform("Minecraft 99.99 Fabric item")


def test_revise_preserves_existing_target_without_migration_request():
    selected = resolve_platform("Add an item", existing_version="1.20.1", existing_loader="fabric")
    assert selected.adapter.minecraft_version == "1.20.1"
    assert selected.preserved_existing_target is True


def test_platform_lock_rejects_mixed_version_tuple() -> None:
    mixed = PlatformLock(
        edition="java",
        loader="fabric",
        minecraft_version="1.21.1",
        java_version="17",
        yarn_mappings="1.21.1+build.3",
        fabric_loader="0.19.3",
        fabric_api="0.116.15+1.21.1",
        fabric_loom="1.10.5",
        gradle="8.12",
    )
    with pytest.raises(SpecValidationError):
        mixed.validate()


def test_target_evidence_uses_live_sources_without_historical_javadoc_ids() -> None:
    ids = {
        item.source_id
        for item in evidence_for_target(None, minecraft_version="explicit-test-version")
    }
    assert "fabric-develop-live" in ids
    assert "fabric-meta" in ids
    assert not any("1201" in source_id or "1211" in source_id for source_id in ids)


def test_generator_fails_closed_without_reviewed_deterministic_templates(
    tmp_path: Path,
) -> None:
    adapter = _fabric_1211()
    assert adapter.deterministic_module_kinds == frozenset()
    project_root = tmp_path / "project"

    with pytest.raises(
        GenerationError,
        match="no reviewed deterministic module templates",
    ):
        FabricProjectGenerator().generate(_simple_spec(adapter), project_root)

    assert not project_root.exists()
