from __future__ import annotations

from dataclasses import replace

import pytest

from minecraft_mod_ai.spec import PlatformLock, SpecValidationError, platform_receipt_sha256


def _seal(lock: PlatformLock) -> PlatformLock:
    return replace(lock, receipt_sha256=platform_receipt_sha256(lock))


def _base(**overrides: object) -> PlatformLock:
    values: dict[str, object] = {
        "edition": "java",
        "loader": "fabric",
        "minecraft_version": "26.2",
        "java_version": "25",
        "yarn_mappings": "",
        "fabric_loader": "0.18.0",
        "fabric_api": "0.140.0+26.2",
        "fabric_loom": "1.14.0",
        "gradle": "9.1.0",
        "adapter_id": "fabric-target-v3",
        "mappings_kind": "",
        "mappings_version": "",
        "gradle_sha256": "0" * 64,
        "gradle_distribution_url": "https://services.gradle.org/distributions/gradle-9.1.0-bin.zip",
        "data_pack_version": "94.1",
        "resource_pack_version": "75.0",
        "resource_pack_format": 75,
        "release_metadata_url": "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json",
        "source_api_family": "fabric-26.2",
        "deterministic_module_kinds": ("metadata",),
        "receipt_sha256": "",
    }
    values.update(overrides)
    return PlatformLock(**values)


def test_native_26_2_platform_lock_accepts_no_legacy_mapping_coordinates() -> None:
    lock = _seal(_base())

    assert lock.has_full_execution_receipt()
    lock.validate()


def test_native_26_2_platform_lock_rejects_fabricated_legacy_mappings() -> None:
    lock = _seal(
        _base(
            yarn_mappings="26.2+build.1",
            mappings_kind="yarn",
            mappings_version="26.2+build.1",
        )
    )

    assert not lock.has_full_execution_receipt()
    with pytest.raises(SpecValidationError, match="naming regime"):
        lock.validate()


def test_pre_26_1_platform_lock_still_requires_real_mapping_coordinates() -> None:
    incomplete = _seal(
        _base(
            minecraft_version="1.21.4",
            java_version="21",
            fabric_api="0.119.4+1.21.4",
            source_api_family="fabric-1.21.4",
        )
    )
    assert not incomplete.has_full_execution_receipt()
    with pytest.raises(SpecValidationError):
        incomplete.validate()

    mapped = _seal(
        _base(
            minecraft_version="1.21.4",
            java_version="21",
            yarn_mappings="1.21.4+build.8",
            mappings_kind="yarn",
            mappings_version="1.21.4+build.8",
            fabric_api="0.119.4+1.21.4",
            source_api_family="fabric-1.21.4",
        )
    )
    assert mapped.has_full_execution_receipt()
    mapped.validate()


def test_native_26_2_platform_lock_rejects_java_below_25() -> None:
    lock = _seal(_base(java_version="24"))

    with pytest.raises(SpecValidationError, match="Java 25"):
        lock.validate()
