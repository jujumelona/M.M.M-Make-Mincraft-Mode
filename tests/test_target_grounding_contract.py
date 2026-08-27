from __future__ import annotations

import pytest

from minecraft_mod_ai import target_grounding_contract as contract
from minecraft_mod_ai.evidence_first_planning import EvidencePlanError


def _target(**overrides):
    value = {
        "minecraft_version": "26.2",
        "loader": "fabric",
        "java_version": "21",
        "yarn_mappings": "mojang",
        "mappings_kind": "mojang",
        "mappings_version": "mojang",
        "fabric_loader": "0.18.0",
        "fabric_api": "0.158.0+26.2",
        "fabric_loom": "1.11-SNAPSHOT",
        "gradle": "9.0.0",
        "gradle_sha256": "a" * 64,
        "data_pack_version": "99.0",
        "resource_pack_version": "88.0",
        "resource_pack_format": 88,
        "release_metadata_url": "https://www.minecraft.net/en-us/article/minecraft-java-edition-26-2",
    }
    value.update(overrides)
    return value


def test_target_barrier_rejects_version_loader_only():
    with pytest.raises(EvidencePlanError, match="TARGET_GROUNDING_INCOMPLETE"):
        contract._validate_complete_target(
            {"minecraft_version": "26.2", "loader": "fabric"}
        )


def test_target_barrier_rejects_resource_pack_zero():
    with pytest.raises(EvidencePlanError, match="TARGET_RESOURCE_PACK_FORMAT"):
        contract._validate_complete_target(_target(resource_pack_format=0))


def test_target_barrier_rejects_pack_major_mismatch():
    with pytest.raises(EvidencePlanError, match="TARGET_RESOURCE_PACK_FORMAT"):
        contract._validate_complete_target(
            _target(resource_pack_version="88.0", resource_pack_format=87)
        )


def test_target_barrier_rejects_mapping_schema_alias_mismatch():
    with pytest.raises(EvidencePlanError, match="TARGET_MAPPINGS_ALIAS"):
        contract._validate_complete_target(
            _target(yarn_mappings="yarn-foo", mappings_version="mojang")
        )


def test_complete_target_gets_canonical_mapping_and_pack_receipts():
    result = contract._validate_complete_target(_target())
    assert result["target_schema_version"] == "2"
    assert result["mappings"] == {"kind": "mojang", "version": "mojang"}
    assert result["pack_versions"] == {
        "data": "99.0",
        "resource": "88.0",
        "resource_major": 88,
    }


def test_gradle_root_path_is_not_a_module_id():
    game_design = {
        "_existing_project_inventory": {
            "modules": [
                {"module_id": ":", "source_sets": ["main"]},
                {"module_id": ":client", "source_sets": ["client"]},
            ]
        }
    }
    topology = contract._project_topology(
        game_design,
        {"module_ids": [":", ":client"], "loaders": ["fabric"]},
    )

    assert topology["module_ids"] == ["root", "client"]
    assert ":" not in topology["module_ids"]
    assert topology["gradle_project_paths"] == [":", ":client"]
    assert topology["modules"][0]["gradle_project_path"] == ":"


def test_aggregator_root_without_sources_is_not_a_production_module():
    game_design = {
        "_existing_project_inventory": {
            "modules": [
                {"module_id": ":", "source_sets": []},
                {"module_id": ":common", "source_sets": ["main"]},
                {"module_id": ":fabric", "source_sets": ["main"]},
            ]
        }
    }
    topology = contract._project_topology(game_design, {"loaders": ["fabric"]})

    assert topology["module_ids"] == ["common", "fabric"]
    assert ":" not in topology["gradle_project_paths"]
