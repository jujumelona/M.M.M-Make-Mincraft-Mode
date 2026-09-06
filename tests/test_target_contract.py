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
