from __future__ import annotations

from minecraft_mod_ai.platform_catalog import PlatformAdapter
from minecraft_mod_ai.platform_resolver import PlatformSelection
from minecraft_mod_ai.target_grounding_contract import (
    _REQUIRED_TARGET_FIELDS,
    _validate_complete_target,
)


def _adapter() -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id="fabric_live_26_1_2_test",
        edition="java",
        loader="fabric",
        minecraft_version="26.1.2",
        java_version="21",
        yarn_mappings="mojang",
        mappings_kind="mojang",
        mappings_version="mojang",
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


def test_platform_selection_preserves_complete_provider_target_receipt() -> None:
    adapter = _adapter()
    selection = PlatformSelection(
        adapter=adapter,
        source="test",
        reason="test",
        explicit_version=False,
        explicit_loader=False,
    )

    target = selection.to_dict()["target"]

    assert not [field for field in _REQUIRED_TARGET_FIELDS if target.get(field) in (None, "", "unresolved")]
    assert target["mappings_kind"] == adapter.mappings_kind
    assert target["mappings_version"] == adapter.mappings_version
    assert target["yarn_mappings"] == adapter.yarn_mappings
    assert target["gradle_sha256"] == adapter.gradle_sha256
    assert target["data_pack_version"] == adapter.data_pack_version
    assert target["resource_pack_version"] == adapter.resource_pack_version
    assert target["resource_pack_format"] == adapter.resource_pack_format
    assert target["release_metadata_url"] == adapter.release_metadata_url


def test_serialized_provider_target_passes_grounding_contract() -> None:
    selection = PlatformSelection(
        adapter=_adapter(),
        source="test",
        reason="test",
        explicit_version=False,
        explicit_loader=False,
    )

    grounded = _validate_complete_target(selection.to_dict()["target"])

    assert grounded["mappings"] == {"kind": "mojang", "version": "mojang"}
    assert grounded["pack_versions"] == {
        "data": "101.1",
        "resource": "84.0",
        "resource_major": 84,
    }
    assert grounded["target_schema_version"] == "2"
