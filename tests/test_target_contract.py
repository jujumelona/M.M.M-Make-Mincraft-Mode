from __future__ import annotations

import pytest

from minecraft_mod_ai.target_contract import (
    TargetContractError,
    target_coordinates_from_mapping,
    validate_target_coordinates,
)


def test_native_26_2_allows_blank_mappings() -> None:
    target = validate_target_coordinates(
        "26.2",
        "fabric",
        "",
        declared_mappings_applicable=False,
    )
    assert target.minecraft_version == "26.2"
    assert target.loader == "fabric"
    assert target.mappings == ""
    assert target.mappings_applicable is False
    assert target.naming_regime == "native_unobfuscated"


def test_native_target_rejects_legacy_mapping_coordinate() -> None:
    with pytest.raises(TargetContractError, match="TARGET_MAPPINGS_INAPPLICABLE"):
        validate_target_coordinates("26.2", "fabric", "legacy-yarn")


def test_mapped_target_requires_mapping_coordinate() -> None:
    with pytest.raises(TargetContractError, match="TARGET_MAPPINGS_REQUIRED"):
        validate_target_coordinates("1.21.4", "fabric", "")


def test_declared_applicability_must_match_version_semantics() -> None:
    with pytest.raises(TargetContractError, match="TARGET_MAPPINGS_APPLICABLE"):
        validate_target_coordinates(
            "26.2",
            "fabric",
            "",
            declared_mappings_applicable=True,
        )


def test_provider_receipt_mapping_object_uses_same_contract() -> None:
    target = target_coordinates_from_mapping(
        {
            "minecraft_version": "1.21.4",
            "loader": "fabric",
            "mappings": {"kind": "yarn", "version": "1.21.4+build.8"},
            "naming_regime": {"mappings_applicable": True},
        }
    )
    assert target.mappings == "1.21.4+build.8"
    assert target.mappings_applicable is True


@pytest.mark.parametrize("version,mapping,java", [("1.21.4", "mojang", "21"), ("26.2", "", "25")])
def test_complete_provider_contract_round_trip(version, mapping, java):
    from minecraft_mod_ai.platform_catalog import PlatformAdapter
    from minecraft_mod_ai.target_contract import TargetContract, target_contract_from_mapping

    target = TargetContract(
        adapter_id="fixture", edition="java", loader="fabric", minecraft_version=version,
        java_version=java, yarn_mappings=mapping, mappings_kind="mojang" if mapping else "",
        mappings_version=mapping, fabric_loader="test-loader", fabric_api="test-api",
        fabric_loom="test-loom", gradle="test-gradle", gradle_sha256="a" * 64,
        data_pack_version="1", resource_pack_version="1", resource_pack_format=1,
        release_metadata_url="https://www.minecraft.net/test", source_api_family="test-family",
        deterministic_module_kinds=frozenset(),
    )
    assert PlatformAdapter is TargetContract
    assert target_contract_from_mapping(target.public_dict()) == target
    receipt = target.public_dict()
    receipt["source_api_family"] = ""
    with pytest.raises(ValueError, match="source_api_family"):
        target_contract_from_mapping(receipt)
    receipt = target.public_dict()
    receipt["pack_versions"]["resource_major"] = 99
    with pytest.raises(TargetContractError, match="contradictory"):
        target_contract_from_mapping(receipt)
