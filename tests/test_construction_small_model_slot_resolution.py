"""Typed slot grounding regressions extracted from real production failures."""
from types import SimpleNamespace

from minecraft_mod_ai.construction_operation_binding import (
    _requires_missing_self_receiver, _automatic_literal_type,
)
from minecraft_mod_ai.construction_static_providers import preferred_static_field
from minecraft_mod_ai.construction_retrieval import ConstructionEvidenceIndex


def _index(*, with_field=True, ambiguous=False):
    owner = 'example.ModEntities'
    registry = 'net.minecraft.core.Registry'
    builtins = 'net.minecraft.core.registries.BuiltInRegistries'
    symbols = {
        owner: {'owner': owner, 'kind': 'CLASS', 'name': 'ModEntities', 'access': 1},
        registry: {'owner': registry, 'kind': 'CLASS', 'name': 'Registry', 'access': 1},
        builtins: {'owner': builtins, 'kind': 'CLASS', 'name': 'BuiltInRegistries', 'access': 1},
        registry + '#freeze()': {'kind': 'METHOD', 'owner': registry, 'name': 'freeze',
                                'return_type': registry, 'access': 1, 'static': False},
    }
    if with_field:
        symbols[builtins + '#ENTITY_TYPE'] = {
            'kind': 'FIELD', 'owner': builtins, 'name': 'ENTITY_TYPE',
            'return_type': registry, 'access': 9, 'static': True,
        }
    if ambiguous:
        symbols[builtins + '#ENTITY_TYPE_OTHER'] = {
            'kind': 'FIELD', 'owner': builtins, 'name': 'ENTITY_TYPE',
            'return_type': registry, 'access': 9, 'static': True,
        }
    index = ConstructionEvidenceIndex(symbols, {}, SimpleNamespace(),
        target_path='src/main/java/example/ModEntities.java',
        owner=owner, authoring_contract={})
    return symbols, index


def test_static_entity_registry_provider_is_directly_grounded():
    symbols, index = _index()
    obligation = [{'mode': 'expression', 'expected_type': 'net.minecraft.core.Registry'}]
    assert preferred_static_field(symbols, index, obligation,
        'Provide the registry EntityType registration input for an entity') == (
            'net.minecraft.core.registries.BuiltInRegistries#ENTITY_TYPE')


def test_static_provider_must_be_unique_and_type_and_name_compatible():
    symbols, index = _index(ambiguous=True)
    obligation = [{'mode': 'expression', 'expected_type': 'net.minecraft.core.Registry'}]
    assert preferred_static_field(symbols, index, obligation, 'EntityType registry') is None
    symbols, index = _index()
    assert preferred_static_field(symbols, index, obligation, 'Provide a registry') is None
    assert preferred_static_field(symbols, index,
        [{'mode': 'expression', 'expected_type': 'java.lang.String'}],
        'EntityType registry') is None
    assert preferred_static_field(symbols, index, obligation, 'EntityType registry',
                                  excluded=['net.minecraft.core.registries.BuiltInRegistries#ENTITY_TYPE']) is None


def test_freeze_cannot_construct_its_own_missing_receiver():
    symbols, index = _index()
    row = symbols['net.minecraft.core.Registry#freeze()']
    obligation = [{'mode': 'expression', 'expected_type': 'net.minecraft.core.Registry'}]
    assert _requires_missing_self_receiver(row, obligation, index)
    assert not _requires_missing_self_receiver({**row, 'static': True}, obligation, index)
    assert not _requires_missing_self_receiver(row, [], index)
    assert not _requires_missing_self_receiver(row, [
        {'mode': 'expression', 'expected_type': 'java.lang.String'}], index)


def test_typed_string_literal_skips_model_type_navigation():
    branches = [{'properties': {
        'action': {'const': 'write_type'}, 'base_type': {'const': 'java.lang.String'},
        'array_dimensions': {'type': 'integer', 'minimum': 0, 'maximum': 0},
    }}]
    assert _automatic_literal_type('literal',
        [{'expected_type': 'java.lang.String'}], branches) == {
            'action': 'write_type', 'base_type': 'java.lang.String', 'array_dimensions': 0}
    assert _automatic_literal_type('cast',
        [{'expected_type': 'java.lang.String'}], branches) is None
    assert _automatic_literal_type('literal', [], branches) is None
