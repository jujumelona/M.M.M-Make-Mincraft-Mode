"""Regressions for the real class-only/JDK-Builder production failure."""
import json
from collections import OrderedDict
from types import SimpleNamespace

from minecraft_mod_ai.construction_file_api_evidence import prompt_symbol_record
from minecraft_mod_ai.construction_file_author import (
    _diagnostic_api_evidence,
    _evidence,
)

ENTITY = 'net.minecraft.world.entity.EntityType'
BUILDER = ENTITY + '.Builder'
CATEGORY = 'net.minecraft.world.entity.MobCategory'
REGISTRY = 'net.minecraft.core.Registry'
BUILTINS = 'net.minecraft.core.registries.BuiltInRegistries'
KEY = 'net.minecraft.resources.ResourceKey'
FACTORY = ENTITY + '.EntityFactory'


def test_prompt_projection_preserves_semantic_java_modifiers():
    declaration = {'kind': 'CLASS', 'owner': 'example.Base', 'name': 'Base'}
    for access in (0x0001, 0x0401, 0x0601):
        assert prompt_symbol_record({**declaration, 'access': access})['access'] == access
    field = {'kind': 'FIELD', 'owner': 'example.Base', 'name': 'VALUE', 'return_type': 'int'}
    for access in (0x0009, 0x0019):
        assert prompt_symbol_record({**field, 'access': access})['access'] == access


def inventory():
    symbols = OrderedDict()

    def cls(owner, **extra):
        symbols[owner] = {'kind': 'CLASS', 'owner': owner,
                          'name': owner.rsplit('.', 1)[-1], 'access': 1, **extra}

    def member(owner, name, result, params=(), **extra):
        key = owner + '#' + name + '(' + ','.join(params) + ')'
        symbols[key] = {'kind': 'METHOD', 'owner': owner, 'name': name,
                        'return_type': result, 'parameters': list(params),
                        'static': True, 'access': 1, **extra}

    # These appeared first in the real classpath and exhausted the old cap.
    for n in range(24):
        owner = f'java.example{n:02d}.Builder'
        cls(owner)
        member(owner, 'build', 'java.lang.Object')
    cls('java.rmi.registry.Registry')
    for owner in (ENTITY, CATEGORY, REGISTRY, BUILTINS, KEY):
        cls(owner)
    cls(BUILDER, enclosing_owner=ENTITY)
    cls(FACTORY, enclosing_owner=ENTITY, access=0x0201,
        functional_parameters=[ENTITY, 'net.minecraft.world.level.Level'],
        functional_return_type='net.minecraft.world.entity.Entity')
    member(BUILDER, 'of', BUILDER, (FACTORY, CATEGORY))
    member(BUILDER, 'build', ENTITY, (KEY,), static=False,
           generic_signature='(Lnet/minecraft/resources/ResourceKey<*>;)Lnet/minecraft/world/entity/EntityType<TT;>;')
    member(KEY, 'create', KEY, (KEY, 'net.minecraft.resources.Identifier'))
    member(REGISTRY, 'register', 'java.lang.Object', (REGISTRY, KEY, 'java.lang.Object'))
    for owner, name, result in ((CATEGORY, 'MONSTER', CATEGORY),
                                (BUILTINS, 'ENTITY_TYPE', REGISTRY)):
        symbols[owner + '#' + name] = {'kind': 'FIELD', 'owner': owner,
            'name': name, 'return_type': result, 'static': True, 'access': 1}
    return symbols


SOURCE = f'''package demo;
import {ENTITY};
import {CATEGORY};
import {REGISTRY};
import {BUILTINS};
public class AlienRegistry {{
  Object alien = EntityType.Builder.create(MobCategory.MONSTER, "alien").build("Alien");
}}
'''
DIAGNOSTIC = '''JAVA_PROJECT_ANALYSIS_FAILED: cannot find symbol
  symbol: method create(MobCategory,String)
  location: class Builder
'''


def context(symbols=None):
    return {'symbols': inventory() if symbols is None else symbols,
            'source_pages': {}, 'authoring_contract': {},
            'target_path': 'src/main/java/demo/AlienRegistry.java'}


def test_repair_resolves_imported_nested_builder_before_jdk_namesakes():
    evidence = _diagnostic_api_evidence(context(), SOURCE, DIAGNOSTIC)
    owners = [row['verified_class'] for row in evidence]
    assert owners[0] == BUILDER
    assert not any(owner.startswith('java.example') for owner in owners)
    builder = evidence[0]
    assert {row['name'] for row in builder['members']} >= {'of', 'build'}
    assert not any(row['name'] == 'create' for row in builder['members'])
    assert next(row for row in builder['members'] if row['name'] == 'build')['parameters'] == [KEY]


def test_initial_file_prompt_exposes_nested_factory_and_build_signatures():
    evidence = _evidence(context(), 'Register alien entity type',
                         target_path='src/main/java/demo/AlienRegistry.java',
                         owner='demo.AlienRegistry', router=SimpleNamespace())
    classes = {row['verified_class']: row for row in evidence.get('class_members', [])}
    assert BUILDER in classes
    assert FACTORY in classes
    assert classes[FACTORY]['declaration']['functional_parameters'] == [ENTITY, 'net.minecraft.world.level.Level']
    build = next(row for row in classes[BUILDER]['members'] if row['name'] == 'build')
    assert build['parameters'] == [KEY]
    assert 'generic_signature' in build
    assert all(row['evidence'] == 'actual_target_symbol_index' for row in classes.values())


def test_body_and_repair_retain_inspected_nested_builder_and_factory_contracts():
    from minecraft_mod_ai.construction_file_api_evidence import scoped_body_api_records
    from minecraft_mod_ai.construction_source_repair import (
        focused_java_correction_messages,
    )

    evidence = _evidence(context(), 'Register alien entity type',
                         target_path='src/main/java/demo/AlienRegistry.java',
                         owner='demo.AlienRegistry', router=SimpleNamespace())
    scoped = scoped_body_api_records(evidence)
    by_owner = {row['verified_class']: row for row in scoped}
    assert {BUILDER, FACTORY, KEY} <= by_owner.keys()
    assert any(row['name'] == 'build' for row in by_owner[BUILDER]['members'])
    assert by_owner[FACTORY]['declaration']['functional_parameters']
    correction = focused_java_correction_messages([
        {'role': 'user', 'content': json.dumps({'inspected_target_api': evidence})},
    ], frame={'kind': 'method', 'prefix': 'void run() {\n', 'suffix': '\n}'},
        response='{"implementation":"bad;"}', diagnostic='cannot find symbol')
    preserved = json.loads(correction[-1]['content'])['api_evidence']['class_members']
    assert {BUILDER, FACTORY, KEY} <= {row['verified_class'] for row in preserved}


def test_nested_factory_promotion_preserves_methods_in_every_input_order():
    from minecraft_mod_ai.construction_file_api_evidence import scoped_body_api_records

    parent, child = 'proof.ZParent', 'proof.ZParent.Builder'
    records = [
        {'verified_class': parent, 'members': []},
        {'verified_class': child, 'declaration': {'enclosing_owner': parent}, 'members': [
            {'owner': child, 'kind': 'METHOD', 'name': 'of', 'return_type': 'void'},
            {'owner': child, 'kind': 'METHOD', 'name': 'build', 'return_type': 'void'},
        ]},
    ]
    anchors = [{'owner': parent, 'kind': 'CLASS', 'name': 'ZParent'},
               {'owner': child, 'kind': 'METHOD', 'name': 'of'}]
    for ordering in (records, records[::-1]):
        scoped = scoped_body_api_records({'target_api': anchors, 'class_members': ordering})
        result = next(row for row in scoped if row['verified_class'] == child)
        assert {row['name'] for row in result['members']} == {'of', 'build'}


def test_diagnostic_selection_is_independent_of_classpath_insertion_order():
    symbols = inventory()
    first = _diagnostic_api_evidence(context(symbols), SOURCE, DIAGNOSTIC)
    second = _diagnostic_api_evidence(context(dict(reversed(list(symbols.items())))), SOURCE, DIAGNOSTIC)
    assert first == second


def test_wrong_mapping_import_can_recover_unique_actual_minecraft_class():
    source = SOURCE.replace('net.minecraft.world.entity.EntityType', 'net.minecraft.entity.EntityType')
    evidence = _diagnostic_api_evidence(context(), source, DIAGNOSTIC)
    assert evidence[0]['verified_class'] == BUILDER
    assert ENTITY in {row['verified_class'] for row in evidence}
    assert all('net.minecraft.entity.' not in row['verified_class'] for row in evidence)


def test_private_and_client_only_symbols_are_not_repair_candidates():
    symbols = inventory()
    for owner, access in [('net.minecraft.client.ClientBuilder', 1),
                          ('net.minecraft.world.HiddenBuilder', 2)]:
        symbols[owner] = {'kind': 'CLASS', 'owner': owner,
                          'name': owner.rsplit('.', 1)[-1], 'access': access}
        symbols[owner + '#build'] = {'kind': 'METHOD', 'owner': owner,
            'name': 'build', 'return_type': 'void', 'parameters': [], 'access': 1}
    symbols[BUILDER + '#secret'] = {'kind': 'METHOD', 'owner': BUILDER,
        'name': 'secret', 'return_type': 'void', 'parameters': [], 'access': 2}
    source = SOURCE + '\n// java.example00.Builder and more fake context\n'
    source += 'import net.minecraft.client.ClientBuilder; import net.minecraft.world.HiddenBuilder;'
    evidence = _diagnostic_api_evidence(context(symbols), source, DIAGNOSTIC)
    assert all('ClientBuilder' not in row['verified_class'] and 'HiddenBuilder' not in row['verified_class'] for row in evidence)
    assert all(member['name'] != 'secret' for row in evidence for member in row['members'])


def test_unknown_api_does_not_create_evidence():
    assert _diagnostic_api_evidence(context(),
        'package demo; class Test { Object x = ImaginaryType.fake(); }',
        'cannot find symbol\n symbol: method fake()\n location: class ImaginaryType') == []


def test_referenced_members_survive_large_class_inventory():
    symbols = inventory()
    for i in range(50):
        symbols[BUILDER + f'#noise{i:02d}'] = {'kind': 'METHOD', 'owner': BUILDER,
            'name': f'noise{i:02d}', 'return_type': 'void', 'parameters': [], 'access': 1}
    evidence = _diagnostic_api_evidence(context(symbols), SOURCE, DIAGNOSTIC)
    builder = next(row for row in evidence if row['verified_class'] == BUILDER)
    assert {'build', 'of'} <= {row['name'] for row in builder['members']}
    assert len(builder['members']) <= 16


def test_qualified_jdk_builder_still_resolves_when_explicitly_used():
    evidence = _diagnostic_api_evidence(context(),
        'package demo; class T { Object x = java.example12.Builder.create(); }', DIAGNOSTIC)
    assert evidence[0]['verified_class'] == 'java.example12.Builder'


def test_wildcard_import_resolves_package_before_global_simple_names():
    symbols = inventory()
    other = 'vendor.other.EntityType'
    symbols[other] = {'kind': 'CLASS', 'owner': other, 'name': 'EntityType', 'access': 1}
    source = SOURCE.replace(f'import {ENTITY};', 'import net.minecraft.world.entity.*;')
    evidence = _diagnostic_api_evidence(context(symbols), source, DIAGNOSTIC)
    assert evidence[0]['verified_class'] == BUILDER
    assert other not in {row['verified_class'] for row in evidence}


def test_same_package_type_resolves_before_global_namesake():
    symbols = inventory()
    other = 'vendor.other.EntityType'
    symbols[other] = {'kind': 'CLASS', 'owner': other, 'name': 'EntityType', 'access': 1}
    source = SOURCE.replace('package demo;', 'package net.minecraft.world.entity;')
    source = source.replace(f'import {ENTITY};', '')
    evidence = _diagnostic_api_evidence(context(symbols), source, DIAGNOSTIC)
    assert evidence[0]['verified_class'] == BUILDER


def test_inaccessible_signature_type_is_not_offered_to_common_author():
    symbols = inventory()
    client = 'net.minecraft.client.ClientToken'
    symbols[client] = {'kind': 'CLASS', 'owner': client, 'name': 'ClientToken', 'access': 1}
    symbols[BUILDER + '#clientOnly'] = {'kind': 'METHOD', 'owner': BUILDER,
        'name': 'clientOnly', 'return_type': 'void', 'parameters': [client], 'access': 1}
    evidence = _diagnostic_api_evidence(context(symbols), SOURCE, DIAGNOSTIC)
    assert all(member['name'] != 'clientOnly' for row in evidence for member in row['members'])


def test_factory_overloads_cannot_displace_referenced_build_member():
    symbols = inventory()
    for i in range(24):
        symbols[BUILDER + f'#of{i:02d}'] = {'kind': 'METHOD', 'owner': BUILDER,
            'name': 'of', 'return_type': BUILDER, 'parameters': ['int'] * (i + 1),
            'static': True, 'access': 1}
    evidence = _diagnostic_api_evidence(context(symbols), SOURCE.replace('create(', 'of('), DIAGNOSTIC)
    builder = next(row for row in evidence if row['verified_class'] == BUILDER)
    assert {'of', 'build'} <= {row['name'] for row in builder['members']}
    assert builder['member_count'] > len(builder['members'])
    assert builder['members_truncated'] is True


def test_public_inherited_member_is_available_on_imported_child():
    symbols = inventory()
    base, child = 'example.Base', 'example.Child'
    symbols[base] = {'kind': 'CLASS', 'owner': base, 'name': 'Base', 'access': 1}
    symbols[child] = {'kind': 'CLASS', 'owner': child, 'name': 'Child', 'access': 1,
                      'superclass': base}
    symbols[base + '#run'] = {'kind': 'METHOD', 'owner': base, 'name': 'run',
        'return_type': 'void', 'parameters': [], 'access': 1}
    symbols[base + '#private'] = {'kind': 'METHOD', 'owner': base, 'name': 'privateRun',
        'return_type': 'void', 'parameters': [], 'access': 2}
    source = 'package demo; import example.Child; class T { void f(Child c) { c.run(); } }'
    evidence = _diagnostic_api_evidence(context(symbols), source,
        'cannot find symbol\n symbol: method run()\n location: class Child')
    child_record = next(row for row in evidence if row['verified_class'] == child)
    run = next(row for row in child_record['members'] if row['name'] == 'run')
    assert run['owner'] == base
    assert all(row['name'] != 'privateRun' for row in child_record['members'])


def test_instance_diagnostic_owner_precedes_incidental_imports():
    symbols = inventory()
    imports = []
    for n in range(15):
        owner = f'example.A{n:02d}'
        symbols[owner] = {'kind': 'CLASS', 'owner': owner, 'name': f'A{n:02d}', 'access': 1}
        imports.append('import ' + owner + ';')
    source = ('package demo; ' + ' '.join(imports) + f' import {BUILDER}; '
              'class T { void f(Builder builder) { builder.create(); } }')
    evidence = _diagnostic_api_evidence(context(symbols), source,
        'cannot find symbol\n symbol: method create()\n location: variable builder of type Builder<T>')
    assert evidence[0]['verified_class'] == BUILDER
    assert 'build' in {row['name'] for row in evidence[0]['members']}


def test_callback_class_cannot_expose_inaccessible_functional_signature():
    symbols = inventory()
    client, callback = 'net.minecraft.client.ClientToken', 'example.Callback'
    symbols[client] = {'kind': 'CLASS', 'owner': client, 'name': 'ClientToken', 'access': 1}
    symbols[callback] = {'kind': 'CLASS', 'owner': callback, 'name': 'Callback', 'access': 0x601,
        'functional_parameters': ['java.lang.Object'], 'functional_return_type': client,
        'functional_throws': [client], 'functional_contract_sources': [callback + '#apply']}
    source = 'package demo; import example.Callback; class T { Callback c; }'
    evidence = _diagnostic_api_evidence(context(symbols), source,
        'cannot find symbol\n symbol: class Callback\n location: class T')
    declaration = next(row for row in evidence if row['verified_class'] == callback)['declaration']
    assert all(not key.startswith('functional_') for key in declaration)
    assert symbols[callback]['functional_return_type'] == client  # inventory stays immutable


def test_inherited_default_interface_method_and_override_keep_exact_owner():
    symbols = inventory()
    base, child = 'example.Interface', 'example.Implementation'
    symbols[base] = {'kind': 'CLASS', 'owner': base, 'name': 'Interface', 'access': 0x201}
    symbols[child] = {'kind': 'CLASS', 'owner': child, 'name': 'Implementation', 'access': 1,
                      'interfaces': [base]}
    for owner, name in ((base, 'defaultAction'), (base, 'overridden'), (child, 'overridden')):
        symbols[owner + '#' + name] = {'kind': 'METHOD', 'owner': owner, 'name': name,
            'return_type': 'void', 'parameters': [], 'access': 1}
    source = f'package demo; import {child}; class T {{ Implementation c; }}'
    evidence = _diagnostic_api_evidence(context(symbols), source,
        'cannot find symbol\n symbol: method action()\n location: class Implementation')
    members = next(row for row in evidence if row['verified_class'] == child)['members']
    assert next(row for row in members if row['name'] == 'defaultAction')['owner'] == base
    assert [row['owner'] for row in members if row['name'] == 'overridden'] == [child]


def test_prompt_api_contains_call_contracts_without_repeated_index_metadata():
    symbols = inventory()
    for row in symbols.values():
        row.update(source_hash='sha256:' + '1' * 64,
                   binary_owner=row['owner'].replace('.', '/'),
                   namespace='OFFICIAL', origin='inspected_bytecode',
                   descriptor='(Lnet/minecraft/world/entity/EntityType;)Ljava/lang/Object;',
                   index_debug={'classpath': ['exact-target.jar'] * 80})
    evidence = _diagnostic_api_evidence(context(symbols), SOURCE, DIAGNOSTIC)
    serialized = json.dumps(evidence, separators=(',', ':'))
    assert 'index_debug' not in serialized
    assert 'binary_owner' not in serialized
    assert 'descriptor' not in serialized
    builder = next(row for row in evidence if row['verified_class'] == BUILDER)
    build = next(row for row in builder['members'] if row['name'] == 'build')
    assert build['parameters'] == [KEY]
    assert 'generic_signature' in build
    assert build['owner'] == BUILDER
    assert symbols[BUILDER]['index_debug']['classpath'] == ['exact-target.jar'] * 80


def test_large_class_name_catalog_does_not_fill_the_model_prompt():
    symbols = inventory()
    for n in range(2500):
        symbols[BUILDER + f'#unrelated{n}'] = {'kind': 'METHOD', 'owner': BUILDER,
            'name': f'unrelated{n}', 'return_type': 'void', 'parameters': [], 'access': 1}
    evidence = _diagnostic_api_evidence(context(symbols), SOURCE, DIAGNOSTIC)
    builder = next(row for row in evidence if row['verified_class'] == BUILDER)
    shown = {row['name'] for row in builder['members']}
    assert set(builder['overload_counts']) <= shown
    assert builder['member_count'] == 2502
    assert builder['members_truncated'] is True


def test_artifact_responsibility_drives_api_lookup_for_broad_gameplay_request():
    symbols = inventory()
    for owner in ('net.minecraft.util.datafix.fixes.LegacyDragonFightFix',
                  'net.minecraft.world.level.dimension.end.EnderDragonFight'):
        symbols[owner] = {'kind': 'CLASS', 'owner': owner, 'name': owner.rsplit('.', 1)[-1], 'access': 1}
    selected = context(symbols)
    selected['authoring_contract'] = {
        'canonical_leaf': 'minecraft/entity/registry',
        'responsibility': 'Register the supplied entity type.',
        'host_requirements': {'symbols': ['EntityType']},
    }
    evidence = _evidence(selected, 'Fight aliens and colonize planets.',
                         target_path='src/main/java/demo/AlienRegistry.java',
                         owner='demo.AlienRegistry', router=SimpleNamespace())
    owners = {row['verified_class'] for row in evidence['class_members']}
    assert ENTITY in owners and BUILDER in owners
    assert not any('DragonFight' in owner for owner in owners)


def test_object_members_are_only_included_when_the_source_references_them():
    symbols = inventory()
    root = 'java.lang.Object'
    symbols[root] = {'kind': 'CLASS', 'owner': root, 'name': 'Object', 'access': 1}
    for name, params in (('wait', []), ('toString', []), ('equals', [root])):
        symbols[root + '#' + name] = {'kind': 'METHOD', 'owner': root, 'name': name,
            'parameters': params, 'return_type': 'void', 'access': 1}
    symbols[BUILDER]['superclass'] = root
    initial = _diagnostic_api_evidence(context(symbols), SOURCE, DIAGNOSTIC)
    builder = next(row for row in initial if row['verified_class'] == BUILDER)
    assert not any(row['owner'] == root for row in builder['members'])
    explicit = _diagnostic_api_evidence(context(symbols), SOURCE +
        '\nclass Helper { void call(Object value) { value.toString(); } }\n', DIAGNOSTIC)
    builder = next(row for row in explicit if row['verified_class'] == BUILDER)
    assert next(row for row in builder['members'] if row['name'] == 'toString')['owner'] == root


def test_registry_leaf_targets_real_registry_method_not_jdi_implementors():
    """The production registry responsibility must not become a JDI inspection request."""
    from minecraft_mod_ai.construction_file_author import _file_api_intent
    symbols = inventory()
    jdi = 'com.sun.jdi.InterfaceType'
    symbols[jdi] = {'kind': 'CLASS', 'owner': jdi, 'name': 'InterfaceType', 'access': 1}
    symbols[jdi + '#implementors'] = {'kind': 'METHOD', 'owner': jdi,
                                     'name': 'implementors', 'return_type': 'java.util.List',
                                     'parameters': [], 'access': 1}
    selected = context(symbols)
    selected['authoring_contract'] = {
        'canonical_leaf': 'minecraft/entity/registry',
        'responsibility': 'Implement only the registry responsibility of the supplied entity artifact.',
        'host_requirements': {'symbols': ['EntityType']},
    }
    intent = _file_api_intent(selected, 'Fight aliens and colonize planets.')
    evidence = _evidence(selected, intent,
                         target_path='src/main/java/demo/AlienRegistry.java',
                         owner='demo.AlienRegistry', router=SimpleNamespace())
    assert any(row.get('owner') == REGISTRY and row.get('name') == 'register'
               for row in evidence['target_api'])
    assert not any(row.get('owner', '').startswith('com.sun.jdi.')
                   for row in evidence['target_api'])


def test_initial_registry_api_includes_exact_resource_key_build_dependency():
    symbols = inventory()
    for i in range(24):
        owner = ENTITY + f'.Noise{i:02d}'
        symbols[owner] = {'kind': 'CLASS', 'owner': owner, 'name': owner.rsplit('.', 1)[-1],
                          'enclosing_owner': ENTITY, 'access': 1}
    evidence = _evidence(context(symbols), 'Register alien entity type',
                         target_path='src/main/java/demo/AlienRegistry.java',
                         owner='demo.AlienRegistry', router=SimpleNamespace())
    shown = {row['verified_class'] for row in evidence['class_members']}
    assert BUILDER in shown
    assert KEY in shown  # Builder.build(ResourceKey) cannot be guessed.


def test_registry_builder_evidence_includes_resource_key_identifier_factory():
    symbols = inventory()
    identifier = 'net.minecraft.resources.Identifier'
    symbols[identifier] = {'kind': 'CLASS', 'owner': identifier,
                           'name': 'Identifier', 'access': 1}
    symbols[identifier + '#parse'] = {'kind': 'METHOD', 'owner': identifier,
                                     'name': 'parse', 'static': True,
                                     'parameters': ['java.lang.String'],
                                     'return_type': identifier, 'access': 1}
    evidence = _evidence(context(symbols), 'Register alien entity type',
                         target_path='src/main/java/demo/AlienRegistry.java',
                         owner='demo.AlienRegistry', router=SimpleNamespace())
    classes = {row['verified_class']: row for row in evidence['class_members']}
    assert KEY in classes
    assert identifier in classes
    assert any(row['name'] == 'create' and identifier in row['parameters']
               for row in classes[KEY]['members'])


def test_callback_member_access_uses_inspected_class_owner_with_opaque_index_keys():
    from copy import deepcopy

    from minecraft_mod_ai.construction_file_api_evidence import FileAPIEvidence
    symbols = {
        'opaque-class': {'kind': 'CLASS', 'owner': 'example.Target', 'name': 'Target', 'access': 1},
        'opaque-method': {'kind': 'METHOD', 'owner': 'example.Target', 'name': 'updateFlag',
                          'return_type': 'void', 'parameters': [], 'access': 1},
        'opaque-private': {'kind': 'CLASS', 'owner': 'other.Hidden', 'name': 'Hidden', 'access': 2},
        'opaque-client': {'kind': 'CLASS', 'owner': 'other.Client', 'name': 'Client', 'access': 1, 'side': 'CLIENT'},
    }
    original = deepcopy(symbols)
    index = FileAPIEvidence(symbols, target_path='src/main/java/demo/Feature.java', owner='demo.Feature')
    records = index.callback_member_records(['example.Target', 'other.Hidden', 'other.Client'], intent='Update flag')
    assert [row['verified_class'] for row in records] == ['example.Target']
    assert [row['name'] for row in records[0]['members']] == ['updateFlag']
    assert symbols == original
