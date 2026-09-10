from __future__ import annotations

from minecraft_mod_ai.translation_runtime import TRANSLATION_SEQUENCE, translate_requirement


def _requirement(name: str) -> dict:
    return {
        "requirement_id": name,
        "capability": name,
        "statement": name,
        "artifact_obligations": [
            {"kind": "block_entity"},
            {"kind": "network_payload"},
            {"kind": "screen"},
            {"kind": "loot_table"},
        ],
    }


def test_translation_templates_execute_in_fixed_order() -> None:
    plan = translate_requirement(_requirement("plain"))
    assert tuple(receipt["template_id"] for receipt in plan.receipts) == TRANSLATION_SEQUENCE
    assert len(plan.receipts) == 8
    assert all(receipt["proof"]["passed"] for receipt in plan.receipts)


def test_translation_is_invariant_to_feature_and_gameplay_names() -> None:
    first = translate_requirement(_requirement("plain"))
    second = translate_requirement(_requirement("boss_trade_economy_skill_dungeon"))
    assert first.artifact_kinds == second.artifact_kinds
    assert first.branch_features == second.branch_features
    assert first.server_artifacts == second.server_artifacts
    assert first.client_artifacts == second.client_artifacts
    assert first.shared_artifacts == second.shared_artifacts


def test_translation_expands_dependencies_before_minecraft_execution() -> None:
    plan = translate_requirement({"artifact_obligations": [{"kind": "dimension"}, {"kind": "mob"}, {"kind": "recipe"}]})
    assert plan.artifact_kinds == (
        "worldgen", "dimension", "entity", "mob", "datagen", "recipe",
    )


def test_translation_preserves_unknown_input_as_unresolved() -> None:
    plan = translate_requirement({"artifact_obligations": [{"kind": "unclassified_surface"}]})
    assert plan.artifact_kinds == ()
    assert plan.unresolved_inputs == ("unclassified_surface",)
    assert plan.receipts[0]["status"] == "BLOCKED"
    assert plan.receipts[1]["status"] == "BLOCKED"


def test_empty_structure_does_not_invent_artifacts() -> None:
    plan = translate_requirement({"capability": "boss.entity", "statement": "trade dungeon skill"})
    assert plan.artifact_kinds == ()
    assert plan.branch_features == frozenset()
