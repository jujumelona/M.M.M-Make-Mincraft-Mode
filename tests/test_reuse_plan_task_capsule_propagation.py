from __future__ import annotations

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import (
    _owned_reuse_plan,
    _source_donor_decisions,
)
from minecraft_mod_ai.live_module_lowering import _as_custom_carrier
from minecraft_mod_ai.small_model_task_capsule_contract import compile_task_capsule

TASK_ID = "task_trade_reuse_123"
JAVA_PATH = "src/main/java/generated/generated_mod/mmmplan/TaskTradeReuse123.java"
TEST_PATH = "src/test/java/generated/generated_mod/mmmplan/TaskTradeReuse123Test.java"


def _semantic_module() -> ProductionModule:
    main_anchor = {
        "kind": "symbol",
        "locator": f"{JAVA_PATH}#TaskTradeReuse123",
        "ownership": "exclusive",
        "status": "host_reserved",
        "module_id": "root",
        "source_set": "main",
    }
    test_anchor = {
        "kind": "test",
        "locator": f"{TEST_PATH}#TaskTradeReuse123Test",
        "ownership": "exclusive",
        "status": "host_reserved",
        "module_id": "root",
        "source_set": "main",
    }
    reuse_plan = {
        "schema_version": "mmm/grounded-repository-reuse-plan-v2",
        "capabilities": [
            {
                "capability": "space_mode_trading",
                "mode": "adapt",
                "donor": {
                    "repository": "example/trading-mod",
                    "commit_sha": "a" * 40,
                },
            }
        ],
    }
    return ProductionModule(
        module_id=TASK_ID,
        kind="economy",
        config={
            "evidence_task": {
                "task_id": TASK_ID,
                "semantic_outcome": "Trading changes the player's owned currency and inventory.",
                "owned_anchors": [main_anchor, test_anchor],
                "production_bindings": [
                    {
                        "task_ref": TASK_ID,
                        "reuse_action": "adapt",
                        "owned_anchors": [main_anchor],
                    }
                ],
                "acceptance": ["A completed trade changes observable player state."],
                "provides": ["capability:space_mode_trading"],
            },
            "_owned_reuse_plan": reuse_plan,
        },
        depends_on=(),
        required_gates=("source_static_validation", "target_compile"),
    )


def test_validated_reuse_plan_survives_live_lowering_to_generator() -> None:
    semantic = _semantic_module()
    original_plan = semantic.config["_owned_reuse_plan"]

    lowered = _as_custom_carrier(
        semantic,
        extra_config={"platform_generation": "canonical_live_target"},
    )

    assert lowered.kind == "custom_java"
    assert lowered.config["_owned_reuse_plan"] == original_plan
    assert _owned_reuse_plan(lowered) == original_plan
    decisions = _source_donor_decisions(_owned_reuse_plan(lowered))
    assert len(decisions) == 1
    assert decisions[0]["mode"] == "adapt"
    assert decisions[0]["donor"]["repository"] == "example/trading-mod"


def test_reuse_changes_coder_ingredients_but_never_planir_destination() -> None:
    lowered = _as_custom_carrier(_semantic_module(), extra_config={})
    capsule = compile_task_capsule(lowered)

    assert capsule is not None
    assert capsule.reuse_action == "adapt"
    assert capsule.primary_path == JAVA_PATH
    assert capsule.writable_paths == (JAVA_PATH, TEST_PATH)
    assert capsule.creatable_paths == (JAVA_PATH, TEST_PATH)
    assert _owned_reuse_plan(lowered) is not None
