"""Regressions for the real class-only/JDK-Builder production failure."""
from collections import OrderedDict
from types import SimpleNamespace

from minecraft_mod_ai.construction_file_author import _diagnostic_api_evidence, _evidence


ENTITY = 'net.minecraft.world.entity.EntityType'
BUILDER = ENTITY + '.Builder'
CATEGORY = 'net.minecraft.world.entity.MobCategory'
REGISTRY = 'net.minecraft.core.Registry'
BUILTINS = 'net.minecraft.core.registries.BuiltInRegistries'
KEY = 'net.minecraft.resources.ResourceKey'
FACTORY = ENTITY + '.EntityFactory'


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
