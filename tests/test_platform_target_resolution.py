from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.knowledge import evidence_for_target
from minecraft_mod_ai import platform_catalog as catalog
from minecraft_mod_ai import platform_resolver as resolver
from minecraft_mod_ai.platform_catalog import (
    FABRIC_1201,
    FABRIC_1211,
    PlatformAdapter,
    adapter_from_project,
)
from minecraft_mod_ai.platform_live_discovery import LiveFabricTarget
from minecraft_mod_ai.platform_resolver import lock_from_adapter, resolve_platform
from minecraft_mod_ai.spec import (
    ContentKind,
    ContentSpec,
    ModSpec,
    PlatformLock,
    SpecValidationError,
)
from minecraft_mod_ai.validator import ProjectValidator


def _simple_spec(adapter) -> ModSpec:
    return ModSpec(
        mod_id="target_probe",
        mod_name="Target Probe",
        package_name="ai.minecraft.generated.target_probe",
        version="1.0.0",
        summary="Target-aware generator probe",
        contents=(
            ContentSpec(
                content_id="probe_item",
                kind=ContentKind.ITEM,
                display_name_en="Probe Item",
                display_name_ko="프로브 아이템",
                recipe=True,
            ),
        ),
        platform=lock_from_adapter(adapter),
    )


def _future_live(version: str = "27.0") -> LiveFabricTarget:
    return LiveFabricTarget(
        minecraft_version=version,
        stable=True,
        loader_version="0.20.0",
        fabric_api_version=f"0.200.0+{version}",
        loom_version="1.20-SNAPSHOT",
        java_version="25",
        gradle_version="9.7",
        gradle_sha256="a" * 64,
        mappings_kind="mojang",
        mappings_version="mojang",
        discovery_sha256="sha256:" + "b" * 64,
    )


def test_supported_versions_are_live_discovery_not_source_allowlist(monkeypatch) -> None:
    monkeypatch.setattr(
        catalog,
        "latest_stable_versions",
        lambda limit=32: ("27.0", "26.2", "1.21.11")[:limit],
    )
    assert catalog.supported_minecraft_versions(loader="fabric")[:2] == (
        "27.0",
        "26.2",
    )


def test_future_version_needs_no_new_platform_catalog_entry(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "discover_fabric_target", lambda version: _future_live(version))
    selected = catalog.adapter_for_target("27.0", "fabric")
    assert selected.minecraft_version == "27.0"
    assert selected.source_api_family == "fabric_live_ai"
    assert selected.adapter_id.startswith("fabric_live_27_0_")
    assert all(seed.minecraft_version != "27.0" for seed in catalog.PLATFORM_ADAPTERS)


class _ChoiceRouter:
    def __init__(self, selected: str) -> None:
        self.selected = selected
        self.calls = 0

    def generate_text(self, *_args, **_kwargs) -> str:
        self.calls += 1
        return json.dumps(
            {
                "minecraft_version": self.selected,
                "reason": "central compatibility choice",
            }
        )


def _patch_future_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        resolver,
        "supported_minecraft_versions",
        lambda loader="fabric": ("27.0", "26.2"),
    )
    monkeypatch.setattr(catalog, "discover_fabric_target", lambda version: _future_live(version))


def test_central_ai_selects_from_live_discovered_candidates(monkeypatch) -> None:
    _patch_future_candidates(monkeypatch)
    router = _ChoiceRouter("26.2")
    selected = resolve_platform(
        "새 모드를 만들어줘",
        router=router,
    )
    assert router.calls == 1
    assert selected.adapter.minecraft_version == "26.2"
    assert selected.source == "central_ai_over_live_discovery"


def test_central_ai_cannot_invent_undiscovered_version(monkeypatch) -> None:
    _patch_future_candidates(monkeypatch)
    router = _ChoiceRouter("99.99")
    selected = resolve_platform("새 모드를 만들어줘", router=router)
    assert selected.adapter.minecraft_version == "27.0"
    assert "fail-closed" in selected.reason


def test_explicit_future_target_is_hard_constraint_when_officially_discovered(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "discover_fabric_target", lambda version: _future_live(version))
    selected = resolve_platform("Minecraft 27.0 Fabric에 아이템 하나 추가")
    assert selected.adapter.minecraft_version == "27.0"
    assert selected.explicit_version is True

    with pytest.raises(SpecValidationError):
        resolve_platform("Minecraft 27.0 NeoForge에 아이템 하나 추가")


def test_revise_preserves_existing_target_without_migration_request() -> None:
    selected = resolve_platform(
        "기존 모드에 아이템 하나 추가",
        existing_version="1.20.1",
        existing_loader="fabric",
    )
    assert selected.adapter.adapter_id == FABRIC_1201.adapter_id
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


def test_legacy_target_evidence_keeps_exact_yarn_javadocs() -> None:
    ids_1201 = {
        item.source_id for item in evidence_for_target(None, minecraft_version="1.20.1")
    }
    ids_1211 = {
        item.source_id for item in evidence_for_target(None, minecraft_version="1.21.1")
    }
    assert "yarn-1201-javadoc" in ids_1201
    assert "yarn-1211-javadoc" not in ids_1201
    assert "yarn-1211-javadoc" in ids_1211
    assert "yarn-1201-javadoc" not in ids_1211
    assert "fabric-develop-live" in ids_1201
    assert "fabric-develop-live" in ids_1211


def test_1211_generator_writes_1211_source_resource_and_lock(tmp_path: Path) -> None:
    spec = _simple_spec(FABRIC_1211)
    generated = FabricProjectGenerator().generate(spec, tmp_path / "project")
    root = generated.root

    gradle = (root / "build.gradle").read_text(encoding="utf-8")
    java = next((root / "src/main/java").rglob("TargetProbeMod.java")).read_text(
        encoding="utf-8"
    )
    pack = json.loads((root / "src/main/resources/pack.mcmeta").read_text(encoding="utf-8"))
    recipe = json.loads(
        (root / "src/main/resources/data/target_probe/recipe/probe_item.json").read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads((root / ".minecraft_ai/platform-lock.json").read_text(encoding="utf-8"))

    assert "options.release = 21" in gradle
    assert "JavaVersion.VERSION_21" in gradle
    assert "new Item.Settings()" in java
    assert "FabricItemSettings" not in java
    assert "Identifier.of(MOD_ID, name)" in java
    assert pack["pack"]["pack_format"] == 34
    assert recipe["result"]["id"] == "target_probe:probe_item"
    assert lock["adapter_id"] == "fabric_1_21_1"
    assert adapter_from_project(root).adapter_id == FABRIC_1211.adapter_id


def test_1211_static_validator_uses_singular_data_paths(tmp_path: Path) -> None:
    spec = _simple_spec(FABRIC_1211)
    root = FabricProjectGenerator().generate(spec, tmp_path / "project").root
    report = ProjectValidator().validate(root, spec)
    assert report.status == "PASS", [item.__dict__ for item in report.findings]


def test_validator_rejects_project_and_proposal_target_mismatch(tmp_path: Path) -> None:
    root = FabricProjectGenerator().generate(
        _simple_spec(FABRIC_1211), tmp_path / "project"
    ).root
    report = ProjectValidator().validate(root, _simple_spec(FABRIC_1201))
    assert report.status == "FAIL"
    assert any(item.code == "PLATFORM_LOCK_MISMATCH" for item in report.findings)
