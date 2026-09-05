from __future__ import annotations

from minecraft_mod_ai.platform_catalog import PlatformAdapter
from minecraft_mod_ai.platform_resolver import PlatformSelection
from minecraft_mod_ai.target_grounding_contract import (
    _required_target_fields,
    _validate_complete_target,
)


def _mapped_adapter() -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id="fabric_mapped_1_21_11_test",
        edition="java",
        loader="fabric",
        minecraft_version="1.21.11",
        java_version="21",
        yarn_mappings="mojang",
        mappings_kind="mojang",
        mappings_version="mojang",
        fabric_loader="0.18.4",
        fabric_api="0.136.1+1.21.11",
        fabric_loom="1.14.10",
        gradle="9.2.1",
        gradle_sha256="a" * 64,
        data_pack_version="94",
        resource_pack_version="75",
        resource_pack_format=75,
        release_metadata_url=(
            "https://piston-meta.mojang.com/v1/packages/deadbeef/1.21.11.json"
        ),
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


def _native_adapter() -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id="fabric_native_26_1_2_test",
        edition="java",
        loader="fabric",
        minecraft_version="26.1.2",
        java_version="25",
        yarn_mappings="",
        mappings_kind="",
        mappings_version="",
        fabric_loader="0.18.4",
        fabric_api="0.140.2+26.1",
        fabric_loom="1.14.10",
        gradle="9.2.1",
        gradle_sha256="a" * 64,
        data_pack_version="101.1",
        resource_pack_version="84.0",
        resource_pack_format=84,
        release_metadata_url=(
            "https://piston-meta.mojang.com/v1/packages/deadbeef/26.1.2.json"
        ),
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


def _selection(adapter: PlatformAdapter) -> PlatformSelection:
    return PlatformSelection(
        adapter=adapter,
        source="test",
        reason="test",
        explicit_version=False,
        explicit_loader=False,
    )


def test_mapped_platform_selection_preserves_complete_provider_target_receipt() -> None:
    adapter = _mapped_adapter()
    target = _selection(adapter).to_dict()["target"]

    assert not [
        field
        for field in _required_target_fields(target)
        if target.get(field) in (None, "", "unresolved")
    ]
    assert target["mappings_kind"] == adapter.mappings_kind
    assert target["mappings_version"] == adapter.mappings_version
    assert target["yarn_mappings"] == adapter.yarn_mappings
    assert target["gradle_sha256"] == adapter.gradle_sha256
    assert target["data_pack_version"] == adapter.data_pack_version
    assert target["resource_pack_version"] == adapter.resource_pack_version
    assert target["resource_pack_format"] == adapter.resource_pack_format
    assert target["release_metadata_url"] == adapter.release_metadata_url


def test_serialized_mapped_provider_target_passes_grounding_contract() -> None:
    grounded = _validate_complete_target(_selection(_mapped_adapter()).to_dict()["target"])

    assert grounded["mappings"] == {"kind": "mojang", "version": "mojang"}
    assert grounded["naming_regime"]["kind"] == "mapped_obfuscated"
    assert grounded["pack_versions"] == {
        "data": "94",
        "resource": "75",
        "resource_major": 75,
    }
    assert grounded["target_schema_version"] == "3"


def test_native_provider_target_requires_no_legacy_mapping_coordinates() -> None:
    adapter = _native_adapter()
    adapter.validate()
    target = _selection(adapter).to_dict()["target"]

    required = _required_target_fields(target)
    assert "mappings_kind" not in required
    assert "mappings_version" not in required
    grounded = _validate_complete_target(target)
    assert grounded["naming_regime"] == {
        "kind": "native_unobfuscated",
        "mappings_applicable": False,
        "minecraft_version": "26.1.2",
    }
    assert "mappings_kind" not in grounded
    assert "mappings_version" not in grounded
    assert "yarn_mappings" not in grounded
    assert not grounded.get("mappings")
    assert grounded["target_schema_version"] == "3"
