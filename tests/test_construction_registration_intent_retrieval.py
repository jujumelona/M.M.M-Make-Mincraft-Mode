"""A real registry operation must outrank and exclude near-name distractors."""
from types import SimpleNamespace

from minecraft_mod_ai.construction_retrieval import ConstructionEvidenceIndex


def _row(owner, name, *, result='void'):
    return {'owner': owner, 'kind': 'METHOD', 'name': name, 'access': 1,
            'return_type': result, 'parameters': [], 'static': True}


def _index(*, with_registry=True):
    owner = 'example.GeneratedRegistry'
    entity = 'net.minecraft.world.entity.Entity'
    entity_type = 'net.minecraft.world.entity.EntityType'
    registry = 'net.minecraft.core.Registry'
    symbols = {
        owner: {'owner': owner, 'kind': 'CLASS', 'name': 'GeneratedRegistry',
                'access': 1, 'provenance': 'construction_plan'},
        entity: {'owner': entity, 'kind': 'CLASS', 'name': 'Entity', 'access': 1},
        entity_type: {'owner': entity_type, 'kind': 'CLASS', 'name': 'EntityType', 'access': 1},
        registry: {'owner': registry, 'kind': 'CLASS', 'name': 'Registry', 'access': 1},
        entity + '#get': _row(entity, 'get', result='java.lang.Object'),
        entity + '#registerDebugValues': _row(entity, 'registerDebugValues'),
        entity_type + '#create': _row(entity_type, 'create', result=entity),
    }
    if with_registry:
        symbols[registry + '#register'] = _row(registry, 'register', result=entity_type)
    return ConstructionEvidenceIndex(symbols, {}, SimpleNamespace(),
        target_path='src/main/java/example/GeneratedRegistry.java',
        owner=owner, authoring_contract={})


def test_actual_registration_method_beats_entity_name_and_debug_distractor():
    index = _index()
    goal = index.prepare("Bind the registry's EntityType registration API for an alien entity",
                         kinds=('METHOD',))
    page = index.page(goal, 0)
    assert page['matches'][0]['symbol_id'] == 'net.minecraft.core.Registry#register'
    assert index.exact_action_available
    assert index.intent_compatible('net.minecraft.core.Registry#register')
    assert not index.intent_compatible('net.minecraft.world.entity.Entity#get')
    assert not index.intent_compatible('net.minecraft.world.entity.Entity#registerDebugValues')
    assert not index.intent_compatible('net.minecraft.world.entity.EntityType#create')


def test_no_exact_action_evidence_does_not_invent_an_api():
    index = _index(with_registry=False)
    index.prepare('EntityType registration API', kinds=('METHOD',))
    assert not index.exact_action_available
    assert index.intent_compatible('net.minecraft.world.entity.Entity#registerDebugValues')


def test_intent_rank_is_not_reused_for_unrelated_query():
    index = _index()
    first = index.prepare('EntityType registration API', kinds=('METHOD',))
    assert index.page(first, 0)['matches'][0]['symbol_id'].endswith('#register')
    second = index.prepare('create an entity', kinds=('METHOD',))
    assert index.page(second, 0)['matches'][0]['symbol_id'].endswith('#create')
