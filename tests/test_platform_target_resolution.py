from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai import platform_catalog as catalog
from minecraft_mod_ai import platform_resolver as resolver
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.knowledge import evidence_for_target
from minecraft_mod_ai.platform_catalog import adapter_from_project
from minecraft_mod_ai.platform_evidence_pipeline import PlatformOptimization, TargetEvidence
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


def _fabric_1201():
    return catalog.adapter_for_target("1.20.1", "fabric")


def _fabric_1211():
    return catalog.adapter_for_target("1.21.1", "fabric")


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
        data_pack_version="100.0",
        resource_pack_version="100.0",
        release_metadata_url="https://www.minecraft.net/en-us/article/minecraft-java-edition-27-0",
        discovery_sha256="sha256:" + "b" * 64,
    )


def _optimization(adapter) -> PlatformOptimization:
    evidence = TargetEvidence(
        adapter=adapter,
        requested_capabilities=(),
        covered_capabilities=(),
        exact_projects=(),
        exact_versions=1,
        verified_hash_files=1,
        dependency_edges=0,
        maintenance_signals=1,
        adoption=0,
        freshness=0.0,
        evidence_quality=1.0,
        integration_risk=0.0,
        residual_cost=0,
        dependency_complexity=0,
    )
    return PlatformOptimization(
        selected=adapter,
        evidence=evidence,
        candidates=(evidence,),
        capability_queries=("test capability",),
        discovery_mode="test-host-evidence",
    )


def test_supported_versions_are_provider_discovery_not_source_allowlist(monkeypatch) -> None:
    provider = catalog.provider_for_loader("fabric")
    monkeypatch.setitem(
        catalog._PROVIDERS,
        "fabric",
        catalog.PlatformProvider(
            loader="fabric",
            provider_id=provider.provider_id,
            discover_versions=lambda limit=32: ("future-a", "future-b", "future-c")[:limit],
            resolve=provider.resolve,
        ),
    )
    assert catalog.supported_minecraft_versions(loader="fabric")[:2] == (
        "future-a",
        "future-b",
    )


def test_future_version_needs_no_new_platform_catalog_entry(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "discover_fabric_target", lambda version: _future_live(version))
    provider = catalog.provider_for_loader("fabric")
    monkeypatch.setitem(
        catalog._PROVIDERS,
        "fabric",
        catalog.PlatformProvider(
            loader="fabric",
            provider_id=provider.provider_id,
            discover_versions=lambda limit=32: ("27.0",)[:limit],
            resolve=catalog._fabric_adapter,
        ),
    )
    selected = catalog.adapter_for_target("27.0", "fabric")
    assert selected.minecraft_version == "27.0"
    assert selected.source_api_family == "fabric_live_ai"
    assert selected.adapter_id.startswith("fabric_live_27_0_")


class _ChoiceRouter:
    def __init__(self, selected: str) -> None:
        self.selected = selected
        self.calls = 0

    def generate_text(self, *_args, **_kwargs) -> str:
        self.calls += 1
        return json.dumps(
            {
                "minecraft_version": self.selected,
                "reason": "model coordinate guess",
            }
        )


def test_host_optimizer_is_coordinate_authority(monkeypatch) -> None:
    selected_adapter = _fabric_1211()
    monkeypatch.setattr(
        resolver,
        "_optimize",
        lambda *_args, **_kwargs: _optimization(selected_adapter),
    )
    router = _ChoiceRouter("1.20.1")

    selected = resolve_platform("새 모드를 만들어줘", router=router)

    assert router.calls == 0
    assert selected.adapter.adapter_id == selected_adapter.adapter_id
    assert selected.source == "host_reuse_optimizer"


def test_model_cannot_invent_platform_coordinate(monkeypatch) -> None:
    selected_adapter = _fabric_1201()
    monkeypatch.setattr(
        resolver,
        "_optimize",
        lambda *_args, **_kwargs: _optimization(selected_adapter),
    )
    router = _ChoiceRouter("99.99")

    selected = resolve_platform("새 모드를 만들어줘", router=router)

    assert router.calls == 0
    assert selected.adapter.minecraft_version == "1.20.1"
    assert selected.adapter.minecraft_version != router.selected


def test_explicit_future_version_is_nonbinding_optimizer_hint(monkeypatch) -> None:
    selected_adapter = _fabric_1201()
    monkeypatch.setattr(
        resolver,
        "_optimize",
        lambda *_args, **_kwargs: _optimization(selected_adapter),
    )

    selected = resolve_platform("Minecraft 27.0 Fabric에 아이템 하나 추가")

    assert selected.adapter.adapter_id == selected_adapter.adapter_id
    assert selected.adapter.minecraft_version != "27.0"
    assert selected.explicit_version is True
    assert selected.source == "host_reuse_optimizer_with_version_hint"

    with pytest.raises(SpecValidationError):
        resolve_platform("Minecraft 27.0 NeoForge에 아이템 하나 추가")


def test_revise_preserves_existing_target_without_migration_request() -> None:
    selected = resolve_platform(
        "기존 모드에 아이템 하나 추가",
        existing_version="1.20.1",
        existing_loader="fabric",
    )
    assert selected.adapter.adapter_id == _fabric_1201().adapter_id
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


def test_generator_uses_adapter_toolchain_resource_format_and_lock(tmp_path: Path) -> None:
    adapter = _fabric_1211()
    spec = _simple_spec(adapter)
    generated = FabricProjectGenerator().generate(spec, tmp_path / "project")
    root = generated.root

    gradle = (root / "build.gradle").read_text(encoding="utf-8")
    pack = json.loads((root / "src/main/resources/pack.mcmeta").read_text(encoding="utf-8"))
    recipe = json.loads(
        (root / "src/main/resources/data/target_probe/recipes/probe_item.json").read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads((root / ".minecraft_ai/platform-lock.json").read_text(encoding="utf-8"))

    assert "Integer.parseInt(project.java_version)" in gradle
    assert "JavaVersion.toVersion(project.java_version)" in gradle
    assert pack["pack"]["pack_format"] == adapter.resource_pack_format
    assert recipe["result"]["item"] == "target_probe:probe_item"
    assert lock["adapter_id"] == adapter.adapter_id
    assert adapter_from_project(root).adapter_id == adapter.adapter_id


def test_static_validator_uses_adapter_selected_project_layout(tmp_path: Path) -> None:
    spec = _simple_spec(_fabric_1211())
    root = FabricProjectGenerator().generate(spec, tmp_path / "project").root
    report = ProjectValidator().validate(root, spec)
    assert report.status == "PASS", [item.__dict__ for item in report.findings]


def test_validator_rejects_project_and_proposal_target_mismatch(tmp_path: Path) -> None:
    root = FabricProjectGenerator().generate(
        _simple_spec(_fabric_1211()), tmp_path / "project"
    ).root
    report = ProjectValidator().validate(root, _simple_spec(_fabric_1201()))
    assert report.status == "FAIL"
    assert any(item.code == "PLATFORM_LOCK_MISMATCH" for item in report.findings)
