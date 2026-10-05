from __future__ import annotations

from minecraft_mod_ai.generation_receipt_paths import (
    receipt_mutation_paths,
    receipt_output_paths,
)


def test_work_node_does_not_treat_semantic_module_ids_as_files() -> None:
    receipt = {
        "schema_version": "mmm/generation-work-node-v1",
        "status": "SUCCEEDED",
        "module_ids": ["space_mode_alien_crystals_loot"],
        "receipts": [
            {
                "schema_version": "mmm/extended-content-v2",
                "status": "GENERATED",
                "modules": ["space_mode_alien_crystals_loot"],
                "module_ids": ["space_mode_alien_crystals_loot"],
                "files": [
                    "src/main/java/example/GeneratedExtendedContent.java",
                ],
                "touched_paths": [
                    "src/main/java/example/GeneratedExtendedContent.java",
                ],
            }
        ],
        "semantic_observations": [
            {
                "schema_version": "mmm/semantic-task-observation-v2",
                "task_id": "space_mode_alien_crystals_loot",
                "task_ids": ["space_mode_alien_crystals_loot"],
                # Evidence is not a filesystem authority. Even a malformed legacy
                # observation must not make an identifier look like a project file.
                "touched_paths": ["space_mode_alien_crystals_loot"],
            }
        ],
    }

    expected = ("src/main/java/example/GeneratedExtendedContent.java",)
    assert receipt_output_paths(receipt) == expected
    assert receipt_mutation_paths(receipt) == expected


def test_source_patch_deletion_is_a_mutation_but_not_an_output() -> None:
    receipt = {
        "schema_version": "mmm/source-patch-receipt-v1",
        "status": "APPLIED",
        "operations": [
            {
                "operation": "replace",
                "path": "src/main/java/example/Kept.java",
            },
            {
                "operation": "delete",
                "path": "src/main/java/example/Removed.java",
            },
        ],
    }

    assert receipt_mutation_paths(receipt) == (
        "src/main/java/example/Kept.java",
        "src/main/java/example/Removed.java",
    )
    assert receipt_output_paths(receipt) == (
        "src/main/java/example/Kept.java",
    )


def test_resource_receipt_indexes_only_project_container_outputs() -> None:
    receipt = {
        "schema_version": "mmm/resource-production-receipt-v2",
        "status": "TEXTURE_PRODUCTION_PASS",
        "assets": [
            {
                "container": "mod",
                "target": "/project/src/main/resources/assets/mod/item.png",
            },
            {
                "container": "resource_pack",
                "target": "/run/resource-pack/assets/mod/standalone.png",
            },
        ],
        "documents": [
            {
                "container": "mod",
                "resolved_path": "/project/src/main/resources/assets/mod/model.json",
            },
            {
                "container": "resource_pack",
                "resolved_path": "/run/resource-pack/assets/mod/model.json",
            },
        ],
    }

    assert receipt_output_paths(receipt) == (
        "/project/src/main/resources/assets/mod/item.png",
        "/project/src/main/resources/assets/mod/model.json",
    )


def test_generic_nested_files_field_is_not_a_path_contract() -> None:
    receipt = {
        "schema_version": "mmm/custom-evidence-v1",
        "files": ["space_mode_alien_crystals_loot"],
        "evidence": {
            "files": ["logical-owner-id"],
            "path": "not-a-project-path",
        },
    }

    assert receipt_output_paths(receipt) == ()
    assert receipt_mutation_paths(receipt) == ()
