from __future__ import annotations

import pytest

from minecraft_mod_ai import evidence_first_planning as planning
from minecraft_mod_ai.requirement_branch_scope_contract import _scoped_branch_predicates
from minecraft_mod_ai.structural_minecraft_runtime_contract import structural_steps_for_requirement


def _requirement(name: str) -> dict:
    return {
        "requirement_id": name,
        "capability": name,
        "statement": name,
        "provides": [f"capability:{name}"],
        "artifact_obligations": [
            {"kind": "block_entity"},
            {"kind": "network_payload"},
            {"kind": "screen"},
            {"kind": "loot_table"},
        ],
    }


def test_same_structure_has_same_task_topology_regardless_of_name() -> None:
    first = structural_steps_for_requirement(_requirement("plain_feature"))
    second = structural_steps_for_requirement(
        _requirement("boss_economy_trade_skill_dungeon")
    )
    assert [step.name for step in first] == [step.name for step in second]
    assert "minecraft_block_entity_state" in {step.name for step in first}
    assert "minecraft_network_payload_codec" in {step.name for step in first}
    assert "minecraft_screen_render" in {step.name for step in first}


def test_same_structure_has_same_branch_topology_regardless_of_name() -> None:
    requirements = [
        _requirement("alpha"),
        _requirement("boss_economy_trade_skill_dungeon"),
    ]
    branches = _scoped_branch_predicates(
        requirements,
        (),
        {"project_topology": {"loaders": ["fabric"]}},
    )
    for branch in (
        "needs_registry",
        "needs_datagen",
        "needs_persistence",
        "needs_network",
        "needs_client_render",
        "needs_worldgen",
    ):
        statuses = branches[branch]["requirement_status"]
        assert statuses["alpha"] == statuses["boss_economy_trade_skill_dungeon"]


def test_live_planner_rejects_name_only_topology_routing() -> None:
    assert planning._compile_tasks.__module__.endswith(
        "structural_minecraft_runtime_contract"
    )
    with pytest.raises(planning.EvidencePlanError, match="STRUCTURAL_REQUIREMENT_REQUIRED"):
        planning._semantic_steps("boss.entity", {})
    with pytest.raises(planning.EvidencePlanError, match="STRUCTURAL_REQUIREMENT_REQUIRED"):
        planning.profile_for_capability("economy.trade")


def test_unknown_structural_artifact_is_blocked_instead_of_guessed() -> None:
    requirement = _requirement("anything")
    requirement["artifact_obligations"] = [{"kind": "unclassified_surface"}]
    with pytest.raises(ValueError, match="STRUCTURAL_ARTIFACT_UNRESOLVED"):
        structural_steps_for_requirement(requirement)
