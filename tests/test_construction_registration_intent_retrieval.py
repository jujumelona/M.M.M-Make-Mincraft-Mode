"""A real registry operation must outrank and exclude near-name distractors."""
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.construction_retrieval import ConstructionEvidenceIndex


def _row(owner, name, *, result='void'):
    return {'owner': owner, 'kind': 'METHOD', 'name': name, 'access': 1,
            'return_type': result, 'parameters': [], 'static': True}


def _index(*, with_registry=True):
    owner = 'example.GeneratedRegistry'
    entity = 'net.minecraft.world.entity.Entity'
    entity_type = 'net.minecraft.world.entity.EntityType'
    registry = 'net.minecraft.core.Registry'
    renderer_registry = 'net.fabricmc.fabric.api.client.rendering.v1.EntityRendererRegistry'
    block_renderer_registry = 'net.fabricmc.fabric.api.client.rendering.v1.BlockEntityRendererRegistry'
    symbols = {
        owner: {'owner': owner, 'kind': 'CLASS', 'name': 'GeneratedRegistry',
                'access': 1, 'provenance': 'construction_plan'},
        entity: {'owner': entity, 'kind': 'CLASS', 'name': 'Entity', 'access': 1},
        entity_type: {'owner': entity_type, 'kind': 'CLASS', 'name': 'EntityType', 'access': 1},
        registry: {'owner': registry, 'kind': 'CLASS', 'name': 'Registry', 'access': 1},
        renderer_registry: {'owner': renderer_registry, 'kind': 'CLASS',
                            'name': 'EntityRendererRegistry', 'access': 1},
        block_renderer_registry: {'owner': block_renderer_registry, 'kind': 'CLASS',
                                  'name': 'BlockEntityRendererRegistry', 'access': 1},
        entity + '#get': _row(entity, 'get', result='java.lang.Object'),
        entity + '#registerDebugValues': _row(entity, 'registerDebugValues'),
        entity_type + '#create': _row(entity_type, 'create', result=entity),
    }
    if with_registry:
        # The populated fixture verifies ranking among several same-verb APIs.
        # The empty-action fixture must not retain other exact register methods.
        symbols[registry + '#register'] = _row(registry, 'register', result=entity_type)
        symbols[renderer_registry + '#register'] = _row(renderer_registry, 'register')
        symbols[block_renderer_registry + '#register'] = _row(block_renderer_registry, 'register')
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


def test_dotted_method_intent_does_not_return_unrelated_value_methods():
    index = _index()
    goal = index.prepare(
        'Call EntityType.registerEntity with the entity type value to register a hostile entity',
        kinds=('METHOD',),
    )
    page = index.page(goal, 0)
    assert page['matches']
    assert page['matches'][0]['symbol_id'] == 'net.minecraft.core.Registry#register'
    assert page['total_count'] < 10
    assert not any('registerDebugValues' in row['symbol_id']
                   for row in page['matches'])


def test_explicit_method_intent_never_builds_the_global_eligible_method_inventory():
    index = _index()

    def global_scan_forbidden(*_args, **_kwargs):
        raise AssertionError('A specific method lookup must not scan the full classpath')

    index._eligible_owners = global_scan_forbidden
    goal = index.prepare(
        'Call EntityType.registerEntity with its value to register a hostile entity',
        kinds=('METHOD',),
    )
    page = index.page(goal, 0)
    assert [row['symbol_id'] for row in page['matches']] == [
        'net.minecraft.core.Registry#register'
    ]


def test_nonexistent_dotted_method_does_not_trigger_global_scan():
    index = _index()

    def global_scan_forbidden(*_args, **_kwargs):
        raise AssertionError('A fictional dotted API is not global-search evidence')

    index._eligible_owners = global_scan_forbidden
    goal = index.prepare(
        'Call EntityType.uninventedMagicalRegistry to register an entity',
        kinds=('METHOD',),
    )
    page = index.page(goal, 0)
    assert page['matches'] == []
    assert page['total_count'] == 0


def test_generic_registration_intent_uses_name_postings_before_global_scan():
    index = _index()
    index._eligible_owners = lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("The entity registration task must not scan the whole API inventory")
    )
    goal = index.prepare("Register the hostile entity type in the registry", kinds=("METHOD",))
    page = index.page(goal, 0)
    assert page["matches"][0]["symbol_id"] == "net.minecraft.core.Registry#register"
    assert not any("registerDebugValues" in item["symbol_id"] for item in page["matches"])


def test_exact_required_return_type_resolves_via_type_postings():
    index = _index()
    index._eligible_owners = lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("An exact type obligation must use indexed return types")
    )
    selection = {"contracts": [{"mode": "expression",
                                "expected_type": "net.minecraft.world.entity.EntityType"}],
                 "excluded_symbols": [], "constructor_result": False}
    goal = index.prepare("Provide the entity type value for the callback",
                         kinds=("METHOD",), selection=selection)
    page = index.page(goal, 0)
    assert page["matches"]
    assert page["matches"][0]["symbol_id"] == "net.minecraft.core.Registry#register"


def test_unavailable_exact_result_type_never_opens_global_method_search():
    index = _index()
    index._eligible_owners = lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("No matching result type cannot justify global method search")
    )
    selection = {"contracts": [{"mode": "expression",
                                "expected_type": "example.NotAvailable"}],
                 "excluded_symbols": [], "constructor_result": False}
    goal = index.prepare("Produce the unavailable required typed value",
                         kinds=("METHOD",), selection=selection)
    result = index.page(goal, 0)
    assert result["matches"] == []
    assert result["total_count"] == 0


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


@pytest.mark.parametrize("purpose, callable_name", [
    ("teleport the player", "teleport"),
    ("teleporting the player", "teleport"),
    ("launch a spacecraft", "launch"),
    ("initialize game state", "initialize"),
    ("initialization of game state", "initialize"),
    ("subscribe to events", "subscribe"),
    ("subscription to events", "subscribe"),
    ("trade with a merchant", "trade"),
    ("trading with a merchant", "trade"),
    ("mine an ore", "mine"),
    ("mining an ore", "mine"),
    ("spawn an alien", "spawn"),
    ("spawning an alien", "spawn"),
    ("remove the entity", "remove"),
    ("removal of an entity", "remove"),
])
def test_recognized_action_inflections_use_actual_method_postings(purpose, callable_name):
    """Every recognized root verb must use real indexed callables without global scan."""
    owner = "example.GeneratedAction"
    target = "example.ActionService"
    symbols = {
        owner: {"owner": owner, "kind": "CLASS", "name": "GeneratedAction",
                "access": 1, "provenance": "construction_plan"},
        target: {"owner": target, "kind": "CLASS", "name": "ActionService", "access": 1},
        target + "#" + callable_name: _row(target, callable_name),
        target + "#get": _row(target, "get"),
    }
    index = ConstructionEvidenceIndex(symbols, {}, SimpleNamespace(),
        target_path="src/main/java/example/GeneratedAction.java",
        owner=owner, authoring_contract={})
    index._eligible_owners = lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("Recognized action must never scan the full classpath"))
    goal = index.prepare(purpose, kinds=("METHOD",))
    page = index.page(goal, 0)
    assert [row["symbol_id"] for row in page["matches"]] == [
        target + "#" + callable_name]
    assert index.exact_action_available
    assert index.intent_compatible(target + "#" + callable_name)
    assert not index.intent_compatible(target + "#get")


def test_action_postings_do_not_qualify_unrelated_suffix_operations():
    """A verb match alone cannot authorize another behavior."""
    owner = "example.GeneratedAction"
    target = "example.ActionService"
    symbols = {
        owner: {"owner": owner, "kind": "CLASS", "name": "GeneratedAction",
                "access": 1, "provenance": "construction_plan"},
        target: {"owner": target, "kind": "CLASS", "name": "ActionService", "access": 1},
        target + "#teleport": _row(target, "teleport"),
        target + "#teleportDebugValues": _row(target, "teleportDebugValues"),
    }
    index = ConstructionEvidenceIndex(symbols, {}, SimpleNamespace(),
        target_path="src/main/java/example/GeneratedAction.java",
        owner=owner, authoring_contract={})
    goal = index.prepare("teleport the player", kinds=("METHOD",))
    page = index.page(goal, 0)
    assert [row["symbol_id"] for row in page["matches"]] == [target + "#teleport"]


def test_ordinary_getter_scoring_handles_no_action_alias():
    """Regression: max(int) is not a valid score reduction for ordinary names."""
    index = _index()
    index.prepare("create an entity", kinds=("METHOD",))
    assert index._member_name_score("net.minecraft.world.entity.Entity#get") == 0.0
