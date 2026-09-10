from __future__ import annotations

import pytest

from minecraft_mod_ai.stage_template_pipeline import (
    KNOWN_STAGES,
    execute_stage_step,
    run_stage_pipeline,
    validate_stage_workflows,
)


def _make_valid_stage_context(stage: str) -> dict:
    if stage == "code":
        return {
            "feature_id": "test_feature",
            "platform": "fabric",
            "artifact_responsibility": "item_behavior",
            "owned_anchor": "com.example.mod.TestItem",
            "file_target": {"path": "src/main/java/com/example/mod/TestItem.java", "owner": "com.example.mod.TestItem"},
            "ownership_reason": "Primary implementation of TestItem",
            "class_contract": {
                "name": "TestItem",
                "primary_responsibility": "Item behavior and activation",
                "file_owner": "src/main/java/com/example/mod/TestItem.java",
            },
            "behavior_step": "on_use_trigger",
            "method_contract": {
                "name": "use",
                "behavior_step": "on_use_trigger",
                "trigger": "right_click",
            },
            "field_contract": [
                {
                    "name": "cooldownTicks",
                    "type": "int",
                    "state_owner": "TestItem",
                    "purpose": "Tracks usage cooldown",
                }
            ],
            "unresolved_type": [],
            "registration_operation": {
                "registry_surface": "ITEM",
                "identifier": "mod:test_item",
                "initialization_point": "onInitialize",
            },
            "call_edge": {
                "producer": "com.example.mod.TestItem#use",
                "consumer": "com.example.mod.TestEffect#apply",
                "reason": "Apply item effect on use",
            },
            "dependency": [
                {
                    "name": "fabric-api",
                    "provider": "net.fabricmc.fabric-api",
                }
            ],
            "unresolved_dependency": [],
            "imports": ["net.minecraft.item.Item"],
            "unresolved_symbols": [],
            "generation_unit": {
                "id": "TestItem_unit",
                "owned_anchor": "com.example.mod.TestItem",
            },
            "blocked_reasons": [],
        }

    if stage == "asset":
        return {
            "feature_id": "test_feature",
            "atomic_feature": {"feature_id": "test_feature"},
            "asset_requirements": [
                {
                    "asset_id": "test_texture",
                    "type": "texture",
                    "trace_ref": "test_feature",
                }
            ],
            "texture_contract": {
                "path": "assets/mod/textures/item/test_item.png",
                "format": "png",
                "resolution": [16, 16],
            },
            "unresolved_properties": [],
            "model_contract": {
                "parent": "item/generated",
                "textures": {"layer0": "mod:item/test_item"},
            },
            "sound_contract": {
                "name": "mod.item.test_use",
                "category": "players",
            },
            "animation_contract": {
                "state_trigger": "use",
                "frames": 4,
            },
            "generated_resource": {"path": "assets/mod/textures/item/test_item.png"},
            "resource_contract": {"type": "texture"},
            "failed_checks": [],
            "valid": True,
        }

    if stage == "integration":
        return {
            "registration_operations": [
                {
                    "identifier": "mod:test_item",
                    "registry": "ITEM",
                    "init_point": "onInitialize",
                },
                {
                    "identifier": "mod:test_block",
                    "registry": "BLOCK",
                    "init_point": "onInitialize",
                },
            ],
            "conflicts": [],
            "dependency_edges": [
                {
                    "consumer": "unit_item",
                    "provider": "unit_registry",
                }
            ],
            "cycles": [],
            "feature_edges": [
                {
                    "source": "feature_a",
                    "target": "feature_b",
                    "contract": "item_triggers_block",
                }
            ],
            "unresolved_connections": [],
            "side_edges": [
                {
                    "side": "common",
                    "channel": "sync_packet",
                }
            ],
            "authority_violations": [],
            "resource_artifacts": [
                {"id": "mod:item/test_item", "path": "assets/mod/models/item/test_item.json"}
            ],
            "code_references": [
                {"resource_id": "mod:item/test_item"}
            ],
            "missing_resources": [],
            "orphaned_resources": [],
            "generation_units": [
                {"id": "unit_item", "anchor": "ItemClass"},
                {"id": "unit_registry", "anchor": "RegistryClass"},
            ],
            "call_edges": [
                {"caller": "unit_item", "callee": "unit_registry"}
            ],
            "invalid_edges": [],
        }

    if stage == "validation":
        return {
            "generated_project": {"root": "mod_project"},
            "target_cell": {"platform": "fabric"},
            "diagnostics": [],
            "compile_exit_code": 0,
            "passed": True,
            "failed_checks": [],
            "unit_tests": [{"name": "test_item_use"}],
            "failures": [],
            "integration_tests": [{"name": "test_registry_integration"}],
            "runtime_scenarios": [{"name": "spawn_and_use"}],
            "observations": ["item_used_successfully"],
            "atomic_feature_acceptance": [{"feature": "test_feature"}],
            "failed_acceptance": [],
            "requirements": ["REQ-1"],
            "implementation_receipts": ["REQ-1"],
            "test_receipts": ["REQ-1", "test_feature"],
            "unproven_requirements": [],
            "atomic_features": [{"feature_id": "test_feature"}],
            "artifact_receipts": ["test_feature"],
            "unproven_features": [],
            "resource_requirements": ["mod:item/test_item"],
            "generated_resources": ["mod:item/test_item"],
            "resource_validation_receipts": ["mod:item/test_item"],
            "missing_resources": [],
            "unowned_resources": [],
            "blockers": [],
        }
    raise ValueError(f"Unknown stage {stage}")


def test_stage_workflows_are_all_valid():
    validate_stage_workflows()
    assert KNOWN_STAGES == ("code", "asset", "integration", "validation")


@pytest.mark.parametrize("stage,expected_steps", [
    ("code", 9),
    ("asset", 6),
    ("integration", 6),
    ("validation", 10),
])
def test_stage_pipeline_execution_and_checkpoint_passing(stage, expected_steps):
    context = _make_valid_stage_context(stage)
    saved_progress = {}

    def checkpoint(binding, receipt):
        saved_progress[binding] = receipt

    result = run_stage_pipeline(stage, context, checkpoint=checkpoint)
    assert result["stage"] == stage
    assert len(result["receipts"]) == expected_steps
    assert len(saved_progress) == expected_steps
    assert all(r["status"] == "PASS" for r in result["receipts"])
    assert all(r["proof"]["passed"] is True for r in result["receipts"])

    # Replay with saved progress
    replayed = run_stage_pipeline(stage, context, progress=saved_progress)
    assert replayed["receipts"] == result["receipts"]


def test_code_stage_proof_blocking():
    # 1. Missing file target blocks code/file_plan
    r = execute_stage_step("code", "code/file_plan", context={"ownership_reason": "test"})
    assert r["status"] == "BLOCKED"
    assert r["proof"]["passed"] is False
    assert "Target file lacks explicit responsibility owner" in r["proof"]["reason"]

    # 2. Unresolved field type blocks code/field_plan
    r = execute_stage_step(
        "code",
        "code/field_plan",
        context={"unresolved_type": ["UnresolvedCustomType"]},
    )
    assert r["status"] == "BLOCKED"
    assert r["proof"]["passed"] is False
    assert "Unresolved field types present" in r["proof"]["reason"]

    # 3. Registration with blocked reason blocks code/registration_plan
    r = execute_stage_step(
        "code",
        "code/registration_plan",
        context={"blocked_reason": "Registry frozen during phase 1"},
    )
    assert r["status"] == "BLOCKED"
    assert r["proof"]["passed"] is False
    assert "Registry frozen" in r["proof"]["reason"]


def test_asset_stage_proof_blocking():
    # 1. Failed checks fails asset/resource_validation
    r = execute_stage_step(
        "asset",
        "asset/resource_validation",
        context={"failed_checks": ["Texture dimensions must be power of 2"]},
    )
    assert r["status"] == "FAIL"
    assert r["proof"]["passed"] is False
    assert r["output"]["valid"] is False

    # 2. Unresolved properties blocks asset/texture_plan
    r = execute_stage_step(
        "asset",
        "asset/texture_plan",
        context={"unresolved_properties": ["unknown_color_palette"]},
    )
    assert r["status"] == "BLOCKED"
    assert r["proof"]["passed"] is False


def test_integration_stage_proof_blocking():
    # 1. Duplicate registration triggers conflict and blocks integration/registration
    r = execute_stage_step(
        "integration",
        "integration/registration",
        context={
            "registration_operations": [
                {"identifier": "mod:item_a"},
                {"identifier": "mod:item_a"},  # duplicate
            ]
        },
    )
    assert r["status"] == "BLOCKED"
    assert r["proof"]["passed"] is False
    assert any("Duplicate registration identifier: mod:item_a" in c for c in r["output"]["conflicts"])

    # 2. Dependency cycle triggers cycles and blocks integration/dependency_connect
    r = execute_stage_step(
        "integration",
        "integration/dependency_connect",
        context={
            "dependency_edges": [
                {"consumer": "node_a", "provider": "node_b"},
                {"consumer": "node_b", "provider": "node_a"},  # cycle
            ]
        },
    )
    assert r["status"] == "BLOCKED"
    assert r["proof"]["passed"] is False
    assert any("Cycle detected" in c for c in r["output"]["cycles"])

    # 3. Missing code resource reference blocks integration/resource_connect
    r = execute_stage_step(
        "integration",
        "integration/resource_connect",
        context={
            "code_references": [{"resource_id": "mod:item/missing_item"}],
            "resource_artifacts": [{"id": "mod:item/existing_item"}],
        },
    )
    assert r["status"] == "BLOCKED"
    assert r["proof"]["passed"] is False
    assert "mod:item/missing_item" in r["output"]["missing_resources"]

    # 4. Authority violation blocks integration/client_server_connect
    r = execute_stage_step(
        "integration",
        "integration/client_server_connect",
        context={
            "authority_violations": ["Client directly mutated server inventory without packet"]
        },
    )
    assert r["status"] == "BLOCKED"
    assert r["proof"]["passed"] is False


def test_validation_stage_proof_blocking():
    # 1. Compiler diagnostics with error fails validation/compile
    r = execute_stage_step(
        "validation",
        "validation/compile",
        context={
            "diagnostics": [{"severity": "error", "message": "Cannot find symbol 'Foo'"}]
        },
    )
    assert r["status"] == "FAIL"
    assert r["proof"]["passed"] is False
    assert r["output"]["passed"] is False

    # 2. Static check failure fails validation/static_check
    r = execute_stage_step(
        "validation",
        "validation/static_check",
        context={"failed_checks": ["Found forbidden java.lang.Thread call"]},
    )
    assert r["status"] == "FAIL"
    assert r["proof"]["passed"] is False
    assert r["output"]["passed"] is False

    # 3. Unproven atomic features block validation/feature_trace
    r = execute_stage_step(
        "validation",
        "validation/feature_trace",
        context={
            "atomic_features": [{"feature_id": "feat_unproven"}],
            "artifact_receipts": [],
            "test_receipts": [],
        },
    )
    assert r["status"] == "BLOCKED"
    assert r["proof"]["passed"] is False
    assert "feat_unproven" in r["output"]["unproven_features"]

    # 4. Completeness dynamically detects blockers from previous receipts
    r = execute_stage_step(
        "validation",
        "validation/completeness",
        context={
            "validation_receipts": [
                {
                    "template_id": "validation/compile",
                    "status": "FAIL",
                    "proof": {"passed": False, "reason": "Compilation failed"},
                }
            ]
        },
    )
    assert r["status"] == "BLOCKED"
    assert r["proof"]["passed"] is False
    assert r["output"]["complete"] is False
    assert any("validation/compile" in b for b in r["output"]["blockers"])
