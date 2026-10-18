"""Isolated regression tests for source- and diagnostic-driven construction.

Works without Fabric/Minecraft/JDK downloads by mocking the code dependencies.
Production still needs a REAL Minecraft 26.2 Gradle/GameTest run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def load_author(monkeypatch, tmp_path):
    root_name = 'minecraft_mod_ai'
    pkg = types.ModuleType(root_name)
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, root_name, pkg)
    def dependency(name, **attributes):
        obj = types.ModuleType(root_name + '.' + name)
        obj.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, root_name + '.' + name, obj)
        return obj

    def digest(v):
        return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'),default=str).encode()).hexdigest()

    from contextlib import nullcontext
    def checkpoint_read(path, identity):
        if not path.exists():
            return {}
        value = json.loads(path.read_text())
        assert value['identity'] == identity
        return value['members']
    def checkpoint_save(path, identity, members):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'identity':identity,'members':members}))

    dependency('construction_direct_writer',
               _digest=digest, _claim_checkpoint=lambda path:nullcontext(),
               _read_checkpoint=checkpoint_read, _save_checkpoint=checkpoint_save,
               _evidence=lambda index, intent, source_pages:{'target_api':[], 'source':[]})
    dependency('construction_retrieval', ConstructionEvidenceIndex=lambda *a,**k:object())
    observations = []
    dependency('java_region_parser', _parse=lambda src: observations.append(('parsed', src)))
    def compile_unit(src, *, classpath, java_version, filename):
        observations.append(('compiled',src))
        if 'INVALID_API' in src:
            raise ValueError('JAVA_ANALYSIS_FAILED: cannot find symbol INVALID_API')
        return []
    dependency('construction_project_compile', verify_java_source_with_dependencies=lambda source,**kw:
               compile_unit(source,classpath=kw['context']['java_classpath'],
                            java_version=kw['context']['target']['java_version'], filename=kw['filename']))
    dependency('construction_coverage',record_authored_source_calls=lambda *a, **k: observations.append(('coverage',a)))

    module_path = Path(__file__).resolve().parents[1] / 'minecraft_mod_ai' / 'construction_file_author.py'
    spec = importlib.util.spec_from_file_location(root_name+'.construction_file_author',module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    context = {'project_root':str(tmp_path), 'target':{'loader':'fabric','java_version':25,
        'minecraft_version':'26.2'}, 'symbols':{}, 'source_pages':{},'java_classpath':['classpath.jar'],
        'prerequisite_hashes':{}, 'feature_work_order_sha256':'abc'}
    source_spec = {'requirement':'Register test item','target_path':'src/main/java/demo/ModUnit.java',
                   'side':'COMMON', 'bindings':{}}
    return module, context, source_spec, observations


GOOD = 'package demo;\npublic class ModUnit {\n public static void register(){System.out.println("ok");}\n}\n'
BAD = GOOD.replace('System.out.println("ok")','INVALID_API()')


class Router:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role,list(messages),kwargs))
        return next(self.replies)


def create(module, context, source_spec, router):
    return module.compose_complete_source(router,source_spec,context,
        package='demo',name='ModUnit',owner='demo.ModUnit')


def test_file_source_compiles_and_checkpoints(monkeypatch,tmp_path):
    module, context, spec, observations = load_author(monkeypatch, tmp_path)
    router = Router([GOOD])
    result = create(module,context,spec,router)
    assert result == GOOD
    assert [s[0] for s in observations] == ['parsed','compiled','coverage']
    assert len(list((tmp_path/'.mmm'/'construction_files').glob('*.json'))) == 1
    assert router.calls[0][2] == {'enable_tools':False, 'force_non_thinking':True}
    router2 = Router([])
    assert create(module, context, spec, router2) == GOOD
    assert not router2.calls


def test_compiler_diagnostic_repairs_full_file(monkeypatch,tmp_path):
    module, context, spec, observations = load_author(monkeypatch, tmp_path)
    router = Router([BAD,GOOD])
    assert create(module,context,spec,router) == GOOD
    assert len(router.calls) == 2
    correction = json.loads(router.calls[1][1][-1]['content'])
    assert correction['previous_source'] == BAD
    assert 'cannot find symbol' in correction['error']
    failure_records = list((tmp_path/'.mmm'/'construction_file_failures').glob('*.json'))
    assert len(failure_records) == 1
    assert json.loads(failure_records[0].read_text())['source'] == BAD


def test_repeated_same_bad_source_fails_without_success_receipt(monkeypatch,tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    router = Router([BAD,BAD])
    with pytest.raises(module.ConstructionFileError, match='NO_SOURCE_PROGRESS'):
        create(module,context,spec,router)
    assert not list((tmp_path/'.mmm'/'construction_files').glob('*.json'))


def test_wrong_package_never_accepted(monkeypatch,tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    with pytest.raises(module.ConstructionFileError, match='PACKAGE_MISMATCH'):
        module._check_identity(GOOD.replace('package demo','package wrong'),'demo','ModUnit')


def test_legacy_ir_not_automatic_default():
    path = Path(__file__).resolve().parents[1] / 'minecraft_mod_ai' / 'construction_generation.py'
    # This source is only present after applying the patch against MMM.
    if not path.exists():
        pytest.skip('Integration entrypoint is supplied by the unified patch')
    code = path.read_text()
    assert 'MMM_LEGACY_TYPED_IR' in code
    assert 'compose_complete_source(' in code


def test_feature_registrations_are_preserved_during_full_file_repair(monkeypatch,tmp_path):
    module, context, spec, observations = load_author(monkeypatch, tmp_path)
    behavior_id = 'alien-spawn.test'
    register = 'registerBehavior'+hashlib.sha256(behavior_id.encode()).hexdigest()[:20]
    spec['bindings'] = {'request_feature_entrypoint': True,
                        'construction_behaviors_json':json.dumps([{'behavior_id':behavior_id,
                              'intent':'spawn alien', 'effects':'alien appears'}])}
    feature = (f'package demo;\npublic class ModUnit implements net.fabricmc.api.ModInitializer {{\n'
               f' private void {register}() {{ System.out.println("registered"); }}\n'
               f' public void onInitialize() {{ this.{register}(); }}\n}}\n')
    missing = feature.replace(f'this.{register}();', 'System.out.println("wrong");')
    router = Router([missing, feature])
    out = module.compose_complete_source(router,spec,context,package='demo',name='ModUnit',
        owner='demo.ModUnit',behaviors=[{'behavior_id':behavior_id,'intent':'spawn alien'}])
    assert out == feature
    prompt = json.loads(router.calls[0][1][1]['content'])
    assert prompt['required_registrations'][0]['method_name'] == register
    assert len(router.calls) == 2


def test_materialized_prerequisite_source_included_in_prompt(monkeypatch,tmp_path):
    module, context, spec, observations = load_author(monkeypatch,tmp_path)
    sibling = tmp_path/'src/main/java/demo/Helper.java'
    sibling.parent.mkdir(parents=True)
    sibling.write_text('package demo; public final class Helper {}', encoding='utf-8')
    router = Router([GOOD])
    create_args = {'package':'demo','name':'ModUnit','owner':'demo.ModUnit',
                   'source_context':[{'path':'src/main/java/demo/Helper.java'}]}
    result = module.compose_complete_source(router,spec,context,**create_args)
    assert result == GOOD
    prompt = json.loads(router.calls[0][1][1]['content'])
    assert prompt['materialized_project_source'][0]['path'].endswith('Helper.java')
    assert 'final class Helper' in prompt['materialized_project_source'][0]['code']


def test_registration_comment_and_string_cannot_satisfy_contract(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    bid = 'behavior-a'
    registrar = 'registerBehavior' + hashlib.sha256(bid.encode()).hexdigest()[:20]
    spec['bindings'] = {'request_feature_entrypoint': True,
        'construction_behaviors_json': json.dumps([{'behavior_id':bid, 'intent':'give item'}])}
    comment = (f'package demo;\npublic class ModUnit implements net.fabricmc.api.ModInitializer {{\n'
        f'// void {registrar}() {{ System.out.println("pretend"); }}\n'
        f'public void onInitialize() {{ System.out.println("this.{registrar}();"); }}\n}}\n')
    with pytest.raises(module.ConstructionFileError, match='BEHAVIOR_METHOD_MISSING'):
        module._check_entrypoint(comment, spec)
    wrong_call = (f'package demo; public class ModUnit implements net.fabricmc.api.ModInitializer {{\n'
        f'private void {registrar}() {{ System.out.println("test"); }}\n'
        f'public void onInitialize() {{ /* this.{registrar}(); */ }}\n}}')
    with pytest.raises(module.ConstructionFileError, match='BEHAVIOR_REGISTRATION_MISSING'):
        module._check_entrypoint(wrong_call, spec)


def test_existing_checkpoint_with_changed_dependency_is_repaired(monkeypatch, tmp_path):
    module, context, spec, observations = load_author(monkeypatch, tmp_path)
    assert create(module, context, spec, Router([GOOD])) == GOOD
    # Existing checkpoint now fails symbol verification (dependent source changed).
    # The author must not pretend checkpoint acceptance; it should pass the old
    # source and the new diagnostic back to the model for repair.
    newer = GOOD.replace('System.out.println("ok")', 'System.err.println("repaired")')
    def verifier(source, **kwargs):
        if source == GOOD:
            raise ValueError('JAVA_PROJECT_ANALYSIS_FAILED: dependency changed')
        observations.append(('compiled-new',source))
    monkeypatch.setattr(sys.modules['minecraft_mod_ai.construction_project_compile'],
                        'verify_java_source_with_dependencies', verifier)
    router = Router([newer])
    assert create(module, context, spec, router) == newer
    assert len(router.calls) == 1
    repair = json.loads(router.calls[0][1][2]['content'])
    assert repair['previous_source'] == GOOD
    assert 'dependency changed' in repair['error']
    assert create(module, context, spec, Router([])) == newer


def test_identity_not_satisfied_by_comments(monkeypatch, tmp_path):
    module, _, _, _ = load_author(monkeypatch, tmp_path)
    bad = ('// package demo;\n'
           'package bad; public class ModUnit { String s = "public class ModUnit"; }')
    with pytest.raises(module.ConstructionFileError, match='PACKAGE_MISMATCH'):
        module._check_identity(bad, 'demo', 'ModUnit')


@pytest.mark.skipif(__import__('shutil').which('javac') is None,reason='requires real JDK')
def test_model_to_real_javac_to_diagnostic_repair_to_checkpoint(monkeypatch,tmp_path):
    """End-to-end author+real JDK smoke, without Minecraft's unavailable jars."""
    import importlib.util
    import shutil
    import subprocess
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    java_version = subprocess.check_output(['javac','-version'],text=True).strip().split()[-1].split('.')[0]
    context['target']['java_version'] = int(java_version)
    context['java_classpath'] = [str(tmp_path)]
    javac = Path(shutil.which('javac')).resolve()
    import types
    lsp = types.ModuleType('minecraft_mod_ai.java_lsp')
    lsp._resolve_project_java_home = lambda version,**kw:javac.parent.parent
    monkeypatch.setitem(sys.modules, 'minecraft_mod_ai.java_lsp', lsp)
    compiler_path = Path(__file__).resolve().parents[1]/'minecraft_mod_ai'/'construction_project_compile.py'
    spec_ = importlib.util.spec_from_file_location('minecraft_mod_ai.construction_project_compile',compiler_path)
    actual_compiler = importlib.util.module_from_spec(spec_)
    monkeypatch.setitem(sys.modules,spec_.name,actual_compiler)
    spec_.loader.exec_module(actual_compiler)
    router = Router([BAD,GOOD])
    assert create(module,context,spec,router) == GOOD
    assert len(router.calls) == 2
    repair = json.loads(router.calls[1][1][-1]['content'])
    assert 'cannot find symbol' in repair['error']
    assert repair['previous_source'] == BAD
    assert list((tmp_path/'.mmm'/'construction_files').glob('*.json'))


def test_diagnostic_evidence_uses_existing_symbols_not_guessed_api(monkeypatch,tmp_path):
    author, context, spec, _ = load_author(monkeypatch, tmp_path)
    context['symbols'] = {
        'net.minecraft.core.registries.BuiltInRegistries': {
            'kind':'CLASS','owner':'net.minecraft.core.registries.BuiltInRegistries',
            'name':'BuiltInRegistries'},
        'net.minecraft.core.registries.BuiltInRegistries#ENTITY_TYPE': {
            'kind':'FIELD','owner':'net.minecraft.core.registries.BuiltInRegistries',
            'name':'ENTITY_TYPE','return_type':'Registry<EntityType<?>>','static':True},
        'net.minecraft.core.registries.BuiltInRegistries#BIOME': {
            'kind':'FIELD','owner':'net.minecraft.core.registries.BuiltInRegistries',
            'name':'BIOME','return_type':'Registry<Biome>','static':True},
    }
    bad = 'package demo; class Thing { void f() { BuiltInRegistries.HOSTILE_ENTITY_TYPE.toString(); }}'
    error = 'JAVA_PROJECT_ANALYSIS_FAILED:\n symbol:   variable BuiltInRegistries\n location: class Thing'
    evidence = author._diagnostic_api_evidence(context,bad,error)
    assert evidence[0]['verified_class'].endswith('BuiltInRegistries')
    names = [entry['name'] for entry in evidence[0]['members']]
    assert 'ENTITY_TYPE' in names
    assert 'HOSTILE_ENTITY_TYPE' not in names
    assert evidence[0]['behavior_verified'] is False


def test_entrypoint_rejects_noop_registrars(monkeypatch, tmp_path):
    module, _, spec, _ = load_author(monkeypatch, tmp_path)
    behavior_id = 'alien-spawn'
    registrar = 'registerBehavior' + hashlib.sha256(behavior_id.encode()).hexdigest()[:20]
    spec['bindings'] = {
        'request_feature_entrypoint': True,
        'construction_behaviors_json': json.dumps([{'behavior_id': behavior_id, 'intent': 'spawn alien'}]),
    }
    for body in ('return;', ';', 'return; /* pretend to spawn */'):
        source = (
            'package demo;\n'
            'public class ModUnit implements net.fabricmc.api.ModInitializer {\n'
            f'  private void {registrar}() {{ {body} }}\n'
            f'  public void onInitialize() {{ this.{registrar}(); }}\n'
            '}\n'
        )
        with pytest.raises(module.ConstructionFileError, match='BEHAVIOR_NOOP'):
            module._check_entrypoint(source, spec)
