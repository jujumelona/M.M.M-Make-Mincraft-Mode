"""Exercise the bytecode-to-prompt path, rather than hand-authored API records.

These are real javac fixtures, not a Minecraft runtime certification.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.construction_context import build_target_symbols
from minecraft_mod_ai.construction_file_author import _evidence
from minecraft_mod_ai.jar_api_extractor import parse_class

ENTITY = 'net.minecraft.world.entity.Entity'
MOB = 'net.minecraft.world.entity.Mob'
MONSTER = 'net.minecraft.world.entity.monster.Monster'
TYPE = 'net.minecraft.world.entity.EntityType'
LEVEL = 'net.minecraft.world.level.Level'


@pytest.fixture(scope='module')
def bytecode(tmp_path_factory):
    java, javac = shutil.which('java'), shutil.which('javac')
    if not javac and not java:
        pytest.skip('JDK required for bytecode-to-prompt regression')
    compiler = [javac] if javac else [java, 'com.sun.tools.javac.Main']
    if subprocess.run([*compiler, '-version'], capture_output=True, check=False).returncode:
        pytest.skip('JDK compiler module required for bytecode-to-prompt regression')
    root = tmp_path_factory.mktemp('generic-entity-api')
    sources = {
        ENTITY: '''package net.minecraft.world.entity;
            import net.minecraft.world.level.Level;
            public abstract class Entity {
                protected Entity(EntityType<?> type, Level level) {}
                protected abstract void save();
            }''',
        MOB: '''package net.minecraft.world.entity;
            import net.minecraft.world.level.Level;
            public abstract class Mob extends Entity {
                protected Mob(EntityType<? extends Mob> type, Level level) { super(type, level); }
                protected void save() {}
            }''',
        MONSTER: '''package net.minecraft.world.entity.monster;
            import net.minecraft.world.entity.*;
            import net.minecraft.world.level.Level;
            public abstract class Monster extends Mob {
                protected Monster(EntityType<? extends Monster> type, Level level) { super(type, level); }
            }''',
        TYPE: '''package net.minecraft.world.entity;
            public class EntityType<T extends Entity> {
                public static class Builder<T extends Entity> {
                    public EntityType<T> build(String key) { return null; }
                }
            }''',
        LEVEL: '''package net.minecraft.world.level;
            public class Level { public void unrelatedWorldOperation() {} }''',
        'example.Generic': '''package example;
            import java.util.*;
            public class Generic<T extends Number & Comparable<T>> {
                public Map<String, ? extends T[]> values;
                public <X extends Number> List<? super X[]> convert(List<? extends X> input)
                    throws java.io.IOException { return null; }
            }''',
    }
    paths = []
    for owner, source in sources.items():
        path = root / (owner.replace('.', '/') + '.java')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        paths.append(str(path))
    classes = root / 'classes'
    subprocess.run([*compiler, '--release', '17', '-d', str(classes), *paths],
                   check=True, capture_output=True, text=True)
    symbols = build_target_symbols([classes], java_version=17, namespace='OFFICIAL',
                                   classpath_sides={})
    return classes, symbols


def test_classfile_signature_survives_inventory_and_prompt(bytecode):
    classes, symbols = bytecode
    parsed = parse_class((classes / 'net/minecraft/world/entity/Mob.class').read_bytes())
    constructor = next(row for row in parsed.members if row.kind == 'CONSTRUCTOR')
    assert getattr(constructor, 'generic_signature', '') == (
        '(Lnet/minecraft/world/entity/EntityType<+Lnet/minecraft/world/entity/Mob;>;'
        'Lnet/minecraft/world/level/Level;)V')
    record = next(row for row in symbols.values()
                  if row['owner'] == MOB and row['kind'] == 'CONSTRUCTOR')
    # The verifier keeps erased descriptors; the model also sees Java generic bounds.
    assert record['parameters'] == [TYPE, LEVEL]
    from minecraft_mod_ai.construction_file_api_evidence import prompt_symbol_record
    prompt = prompt_symbol_record(record)
    assert prompt['generic_parameters'] == [TYPE + '<? extends ' + MOB + '>', LEVEL]


def test_nested_generic_bounds_fields_and_method_parameters_survive(bytecode):
    _, symbols = bytecode
    cls = symbols['example.Generic']
    assert cls.get('type_parameters') == [
        'T extends java.lang.Number & java.lang.Comparable<T>']
    field = next(row for row in symbols.values()
                 if row['owner'] == 'example.Generic' and row.get('name') == 'values')
    assert field.get('generic_return_type') == 'java.util.Map<java.lang.String, ? extends T[]>'
    method = next(row for row in symbols.values()
                  if row['owner'] == 'example.Generic' and row.get('name') == 'convert')
    assert method.get('generic_parameters') == ['java.util.List<? extends X>']
    assert method.get('generic_return_type') == 'java.util.List<? super X[]>'
    assert method.get('type_parameters') == ['X extends java.lang.Number']


def test_entity_type_prompt_contains_protected_super_contract_without_registry(bytecode):
    _, symbols = bytecode
    context = {'symbols': symbols, 'source_pages': {}, 'authoring_contract': {
        'canonical_leaf': 'minecraft/entity/type',
        'responsibility': 'Implement the concrete entity class; do not create a registry.',
        'host_requirements': {'symbols': ['EntityType']},
    }}
    evidence = _evidence(context, 'Fight aliens and colonize planets.',
        target_path='src/main/java/demo/Alien.java', owner='demo.Alien',
        router=SimpleNamespace())
    classes = {row['verified_class']: row for row in evidence['class_members']}
    assert {ENTITY, MOB, MONSTER, TYPE, LEVEL} <= set(classes)
    mob = classes[MOB]
    constructor = next((row for row in mob['members'] if row['kind'] == 'CONSTRUCTOR'), None)
    assert constructor is not None, 'protected super constructors are usable by this generated subclass'
    assert constructor['generic_parameters'][0] == TYPE + '<? extends ' + MOB + '>'
    assert TYPE + '.Builder' not in classes
    assert not any(row.get('name') == 'unrelatedWorldOperation'
                   for cls in classes.values() for row in cls['members'])


def test_rendered_constructor_contract_resolves_real_javac_generic_failure(bytecode, tmp_path):
    classes, symbols = bytecode
    from minecraft_mod_ai.construction_file_api_evidence import FileAPIEvidence
    evidence = FileAPIEvidence(symbols, target_path='src/main/java/demo/Alien.java',
                               owner='demo.Alien').entity_type_evidence()
    mob = next(row for row in evidence['class_members'] if row['verified_class'] == MOB)
    constructor = next(row for row in mob['members'] if row['kind'] == 'CONSTRUCTOR')
    javac = shutil.which('javac')
    compiler = [javac] if javac else [shutil.which('java'), 'com.sun.tools.javac.Main']
    file = tmp_path / 'Alien.java'

    def compile_source(parameters):
        args = ', '.join(f'{type_} p{i}' for i, type_ in enumerate(parameters))
        file.write_text(f'package demo; public class Alien extends {MOB} {{'
                        f'public Alien({args}) {{ super(p0, p1); }} }}')
        return subprocess.run([*compiler, '--release', '17', '-cp', str(classes),
            '-d', str(tmp_path / 'out'), str(file)], check=False, capture_output=True, text=True)

    erased = compile_source([TYPE + '<?>', LEVEL])
    assert erased.returncode != 0
    assert 'incompatible types' in erased.stderr
    corrected = compile_source(constructor['generic_parameters'])
    assert corrected.returncode == 0, corrected.stderr


def test_required_class_roots_are_not_displaced_by_incidental_signature_dependencies():
    from minecraft_mod_ai.construction_file_api_evidence import FileAPIEvidence
    symbols = {}
    roots = ['example.Root', 'example.RequiredSuperclass']
    for owner in [*roots, *(f'example.Noise{i}' for i in range(20))]:
        symbols[owner] = {'kind': 'CLASS', 'owner': owner,
                          'name': owner.rsplit('.', 1)[-1], 'access': 1}
    for i in range(20):
        symbols[f'example.Root#make{i}'] = {
            'kind': 'METHOD', 'owner': 'example.Root', 'name': f'make{i}',
            'parameters': [f'example.Noise{i}'], 'return_type': 'example.Root',
            'access': 9, 'static': True}
    evidence = FileAPIEvidence(symbols, target_path='src/main/java/demo/T.java').records(roots)
    assert roots[1] in {row['verified_class'] for row in evidence}


def test_diagnostic_evidence_preserves_protected_constructor_of_actual_superclass(bytecode):
    from minecraft_mod_ai.construction_file_author import _diagnostic_api_evidence
    _, symbols = bytecode
    source = f'''package demo;
        import {MOB}; import {TYPE}; import {LEVEL};
        public class Alien extends Mob {{
            public Alien(EntityType<?> type, Level level) {{ super(type, level); }}
        }}'''
    diagnostic = '''JAVA_PROJECT_ANALYSIS_FAILED: constructor Mob in class Mob cannot be applied;
        required: EntityType<? extends Mob>,Level
        found: EntityType<?>,Level'''
    records = _diagnostic_api_evidence({'symbols': symbols}, source, diagnostic)
    mob = next(row for row in records if row['verified_class'] == MOB)
    constructor = next((row for row in mob['members'] if row['kind'] == 'CONSTRUCTOR'), None)
    assert constructor is not None
    assert constructor['generic_parameters'] == [TYPE + '<? extends ' + MOB + '>', LEVEL]


def test_entity_type_cannot_publish_registry_code_from_recorded_failure():
    from minecraft_mod_ai.construction_file_author import (
        ConstructionFileError,
        _check_concrete_entity,
    )
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
                          'mimo_entity_failure_20261019.json').read_text())
    with pytest.raises(ConstructionFileError, match='REGISTRY_OUTSIDE_TYPE'):
        _check_concrete_entity(fixture['repair']['previous_source'],
            {'authoring_contract': fixture['prompt']['construction_requirements']},
            'AlienCombatHostileEntity')


@pytest.mark.parametrize('source_namespace', ['official', 'OFFICIAL'])
def test_generic_namespace_mapping_keeps_bound_owners_in_target_namespace(bytecode, tmp_path, source_namespace):
    classes, _ = bytecode
    mapping = tmp_path / 'types.tiny'
    mapping.write_text('tiny\t2\t0\tofficial\tnamed\n'
        'c\tnet/minecraft/world/entity/Mob\tgame/MappedMob\n'
        'c\tnet/minecraft/world/entity/EntityType\tgame/MappedType\n')
    symbols = build_target_symbols([classes], java_version=17, namespace='NAMED',
        classpath_sides={}, mappings_file=mapping, source_namespace=source_namespace)
    constructor = next(row for row in symbols.values()
                       if row['owner'] == 'game.MappedMob' and row['kind'] == 'CONSTRUCTOR')
    assert constructor['generic_parameters'][0] == 'game.MappedType<? extends game.MappedMob>'


def test_generic_bound_does_not_authorize_protected_constructor_access(bytecode):
    from minecraft_mod_ai.construction_file_author import _diagnostic_api_evidence
    _, symbols = bytecode
    source = f'''package demo; import {MOB};
        public class Alien<T extends Mob> {{ public void call() {{}} }}'''
    records = _diagnostic_api_evidence({'symbols': symbols}, source,
        'JAVA_PROJECT_ANALYSIS_FAILED: cannot find symbol')
    mob = next(row for row in records if row['verified_class'] == MOB)
    assert not any(row['kind'] == 'CONSTRUCTOR' for row in mob['members'])


@pytest.mark.parametrize('wildcard', [False, True])
def test_diagnostic_lookup_keeps_used_static_import_owner(wildcard):
    from minecraft_mod_ai.construction_file_author import _diagnostic_api_evidence
    owner = 'example.Actions'
    symbols = {owner: {'kind': 'CLASS', 'owner': owner, 'name': 'Actions', 'access': 1},
        owner + '#spawn': {'kind': 'METHOD', 'owner': owner, 'name': 'spawn',
            'parameters': ['int'], 'return_type': 'void', 'static': True, 'access': 9}}
    source = ('package demo; import static example.Actions.' + ('*' if wildcard else 'spawn') +
              '; class Alien { void run() { spawn(); } }')
    records = _diagnostic_api_evidence({'symbols': symbols}, source,
        'JAVA_PROJECT_ANALYSIS_FAILED: method spawn cannot be applied; required: int')
    assert owner in {row['verified_class'] for row in records}


@pytest.mark.parametrize(('signature', 'expected'), [
    ('<T:>Ljava/lang/Object;', ['T']),
    ('<T:U:Ljava/lang/Object;>Ljava/lang/Object;', ['T', 'U extends java.lang.Object']),
    ('<A:LongName:Ljava/lang/Object;>Ljava/lang/Object;', ['A', 'LongName extends java.lang.Object']),
    ('<A:Type:Ljava/lang/Object;>Ljava/lang/Object;', ['A', 'Type extends java.lang.Object']),
])
def test_valid_empty_class_bound_does_not_abort_generic_inventory(signature, expected):
    from minecraft_mod_ai.java_generic_signature import generic_contract
    assert generic_contract(signature, 'CLASS')['type_parameters'] == expected


def test_missing_members_stay_with_their_own_diagnostic_owner():
    from minecraft_mod_ai.construction_file_author import _diagnostic_api_evidence
    symbols = {owner: {'kind': 'CLASS', 'owner': owner, 'name': owner[-1], 'access': 1}
               for owner in ('example.A', 'example.B')}
    records = _diagnostic_api_evidence({'symbols': symbols},
        'package demo; import example.A; import example.B; '
        'class C { void run(A a, B b) { a.alpha(); b.beta(); } }',
        '''JAVA_PROJECT_ANALYSIS_FAILED:
        C.java:1: error: cannot find symbol
          symbol: method alpha()
          location: class A
        C.java:1: error: cannot find symbol
          symbol: method beta()
          location: class B''')
    classes = {row['verified_class']: row for row in records}
    assert classes['example.A']['unavailable_requested_members'] == ['alpha']
    assert classes['example.B']['unavailable_requested_members'] == ['beta']


def test_new_repair_evidence_clears_obsolete_unavailable_member_fact():
    from minecraft_mod_ai.construction_file_prompt import file_messages
    original = {'verified_class': 'example.A', 'declaration': {}, 'members': [],
                'unavailable_requested_members': ['newCall']}
    latest = {**original, 'unavailable_requested_members': [], 'members': [
        {'kind': 'METHOD', 'owner': 'example.A', 'name': 'newCall', 'parameters': []}]}
    prompt = {'inspected_target_api': {'class_members': [original]}}
    repair = {'previous_source': 'package demo; class C {}', 'error': 'cannot find symbol',
              'verified_symbols_from_compiler_failure': [latest]}
    messages = file_messages('Java only', prompt, repair)
    record = json.loads(messages[1]['content'])['inspected_target_api']['class_members'][0]
    assert not record.get('unavailable_requested_members')


def test_repair_prompt_does_not_grow_a_catalog_of_unrelated_api():
    from minecraft_mod_ai.construction_file_prompt import file_messages
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
                          'mimo_entity_failure_20261019.json').read_text())
    messages = file_messages(fixture['system'], fixture['prompt'], fixture['repair'])
    tasks = [json.loads(row['content']) for row in messages if row['role'] == 'user']
    api = tasks[0]['inspected_target_api']
    owners = {row['verified_class'] for row in api['class_members']}
    assert 'net.minecraft.server.MinecraftServer' not in owners
    assert 'net.minecraft.world.level.storage.LevelStorageSource.LevelStorageAccess' not in owners
    assert tasks[-1]['previous_source'] == fixture['repair']['previous_source']
    assert tasks[0]['construction_bindings'] == fixture['prompt']['construction_bindings']
    actual_chars = sum(len(row['content']) for row in messages) + fixture['host_policy_chars']
    assert actual_chars < fixture['original_repair_chars'] * 0.75
