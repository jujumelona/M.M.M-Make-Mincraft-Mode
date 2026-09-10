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
    assert "minecraft_screen_layout" in {step.name for step in first}


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
    assert not hasattr(planning, "_semantic_steps")
    assert not hasattr(planning, "profile_for_capability")


def test_unknown_structural_artifact_is_blocked_instead_of_guessed() -> None:
    requirement = _requirement("anything")
    requirement["artifact_obligations"] = [{"kind": "unclassified_surface"}]
    with pytest.raises(ValueError, match="STRUCTURAL_ARTIFACT_UNRESOLVED"):
        structural_steps_for_requirement(requirement)


def test_live_task_compiler_binds_structural_steps_and_preserves_gap_inputs():
    requirement = _requirement('plain_feature')
    requirement.update(acceptance=['The declared feature behaves as requested.'],
                       minecraft_structure={'required_artifacts': ['item']},
                       implementation_surfaces=['command'])
    gap = planning._gap_record(requirement, set())
    assert gap['minecraft_structure'] == requirement['minecraft_structure']
    assert gap['implementation_surfaces'] == ['command']
    ownership = planning._ownership_context({})
    tasks = planning._compile_tasks([gap], [], {'coordinates': {}}, {}, ownership, emit_trace=False)
    assert tasks
    providers = {provided: task['task_id'] for task in tasks for provided in task['provides']}
    for task in tasks:
        assert task['task_sha256'] == planning._hash_without(task, 'task_sha256')
        assert task['template_id'] == 'structural_artifact_pipeline'
        assert task['requirement_refs'] == ['plain_feature']
        assert set(task['depends_on']) == {providers[item] for item in task['consumes']
                                          if item != planning.ROOT_PROVIDE}
    assert 'capability:plain_feature' in tasks[-1]['provides']
    assert 'requirement_done:plain_feature' in tasks[-1]['provides']


def test_explicit_requirement_dependencies_are_not_invented_from_capability_names():
    from minecraft_mod_ai.minecraft_requirement_dependencies import bind_selected_feature_dependencies
    catalog = {'requirements': [
        {'requirement_id': 'one', 'capability': 'economy.currency', 'depends_on': []},
        {'requirement_id': 'two', 'capability': 'economy.trade', 'depends_on': []},
    ]}
    result = bind_selected_feature_dependencies(catalog)
    assert all(row['depends_on'] == [] for row in result['requirements'])
    catalog['requirements'][1]['depends_on'] = ['one']
    result = bind_selected_feature_dependencies(catalog)
    assert result['requirements'][1]['depends_on'] == ['one']
    catalog['requirements'][0]['depends_on'] = ['two']
    with pytest.raises(ValueError, match='cycle'):
        bind_selected_feature_dependencies(catalog)
