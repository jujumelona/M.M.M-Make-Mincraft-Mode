from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.knowledge import evidence_for_target
from minecraft_mod_ai.platform_catalog import (
    FABRIC_1201,
    FABRIC_1211,
    adapter_from_project,
    supported_minecraft_versions,
)
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


def test_supported_versions_are_reviewed_catalog_targets() -> None:
    assert supported_minecraft_versions(loader="fabric") == ("1.20.1", "1.21.1")


def test_auto_target_uses_newest_adapter_for_simple_content() -> None:
    selection = resolve_platform("간단한 장식 아이템 하나를 추가해줘")
    assert selection.adapter.adapter_id == FABRIC_1211.adapter_id
    assert selection.source == "host_capability_resolution"
    assert selection.explicit_version is False


def test_auto_target_keeps_mature_adapter_for_advanced_source_family() -> None:
    selection = resolve_platform("새로운 보스 몬스터와 전투 패턴을 추가해줘")
    assert selection.adapter.adapter_id == FABRIC_1201.adapter_id


def test_explicit_target_is_hard_constraint() -> None:
    selected = resolve_platform("Minecraft 1.21.1 Fabric에 아이템 하나 추가")
    assert selected.adapter.adapter_id == FABRIC_1211.adapter_id
    assert selected.explicit_version is True

    with pytest.raises(SpecValidationError):
        resolve_platform("Minecraft 1.22.0 Fabric에 아이템 하나 추가")

    with pytest.raises(SpecValidationError):
        resolve_platform("Minecraft 1.21.1 NeoForge에 아이템 하나 추가")


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


def test_target_evidence_uses_matching_yarn_snapshot() -> None:
    ids_1201 = {item.source_id for item in evidence_for_target("item", minecraft_version="1.20.1")}
    ids_1211 = {item.source_id for item in evidence_for_target("item", minecraft_version="1.21.1")}
    assert "yarn-1201-javadoc" in ids_1201
    assert "yarn-1211-javadoc" not in ids_1201
    assert "yarn-1211-javadoc" in ids_1211
    assert "yarn-1201-javadoc" not in ids_1211


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
