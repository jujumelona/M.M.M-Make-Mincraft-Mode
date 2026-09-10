from dataclasses import replace

from minecraft_mod_ai.minecraft_template_catalog import profile_for_capability, known_capability_ids
from minecraft_mod_ai.minecraft_template_steps import ROOT_PROVIDE, steps_for_profile


def test_entity_responsibilities_are_independent_tasks():
    steps = steps_for_profile(profile_for_capability('entity.lifecycle'))
    names = {step.name for step in steps}
    assert {'minecraft_entity_registry', 'minecraft_entity_type', 'minecraft_entity_dimensions',
            'minecraft_entity_attributes', 'minecraft_entity_ai', 'minecraft_entity_damage',
            'minecraft_entity_death', 'minecraft_entity_persistence',
            'minecraft_entity_renderer'} <= names
    assert not {'entity_type_attributes', 'entity_lifecycle', 'ai_damage_death'} & names


def test_game_name_does_not_choose_task_topology():
    profile = profile_for_capability('entity.lifecycle')
    renamed = replace(profile, capability='custom.request', template_id='arbitrary_label')
    assert [s.name for s in steps_for_profile(profile)] == [s.name for s in steps_for_profile(renamed)]


def test_all_catalog_profiles_have_resolved_dependencies_and_one_terminal_output():
    for capability in known_capability_ids():
        steps = steps_for_profile(profile_for_capability(capability))
        available, names = {ROOT_PROVIDE}, set()
        for step in steps:
            assert step.name not in names, capability
            names.add(step.name)
            assert set(step.consumes) <= available, (capability, step.name)
            assert not set(step.provides) & available, (capability, step.name)
            available.update(step.provides)
        assert steps[-1].provides == (capability,)


def test_unknown_feature_uses_the_same_authored_behavior_tasks():
    steps = steps_for_profile(profile_for_capability('custom.semantic_0123456789abcdef'))
    assert [s.name for s in steps[:6]] == ['trigger', 'input', 'state', 'transition', 'output', 'failure']
    assert steps[-1].name == 'runtime_scenario'


def test_distinct_model_obligations_keep_distinct_task_owners():
    profile = replace(profile_for_capability('entity.lifecycle'), artifact_kinds=('item_model', 'block_model'))
    names = {s.name for s in steps_for_profile(profile)}
    assert 'minecraft_item_model_geometry' in names
    assert 'minecraft_block_model_geometry' in names


def test_client_required_profile_keeps_client_tasks():
    from minecraft_mod_ai.minecraft_template_catalog import FEATURE_CLIENT
    profile = replace(profile_for_capability('custom.semantic'), features=frozenset({FEATURE_CLIENT}))
    steps = steps_for_profile(profile)
    assert {'client_projection', 'client_input', 'client_render'} <= {s.name for s in steps}
