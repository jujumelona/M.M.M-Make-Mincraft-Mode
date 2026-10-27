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

from minecraft_mod_ai.java_region_parser import _parser as _java_parser, _walk_named as _java_nodes


def load_author(monkeypatch, tmp_path):
    root_name = 'minecraft_mod_ai'
    pkg = types.ModuleType(root_name)
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / root_name)]
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
    dependency('construction_retrieval', ConstructionEvidenceIndex=lambda *a,**k:types.SimpleNamespace(by_owner={}))
    observations = []
    dependency('java_region_parser', _parse=lambda src: observations.append(('parsed', src)),
               _parser=_java_parser, _walk_named=_java_nodes)
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
    assert 'MMM_LEGACY_FILE_AUTHOR' in code
    assert 'compose_host_java_source as author' in code
    assert 'legacy_automatic_fallback' in code


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


def _large_api_context():
    return {'target_api': [], 'source': [], 'typed_input_producers': [],
            'class_members': [
                {'verified_class': f'example.Api{i}',
                 'declaration': {'kind': 'CLASS', 'owner': f'example.Api{i}', 'name': f'Api{i}'},
                 'members': [{'kind': 'METHOD', 'owner': f'example.Api{i}',
                              'name': f'auxiliary{n}', 'parameters': [], 'return_type': 'void'}
                             for n in range(16)],
                 'evidence': 'actual_target_symbol_index', 'behavior_verified': False}
                for i in range(12)]}


def _raise_context_pressure():
    from minecraft_mod_ai.llama_finish_reason_contract import LlamaCompletionBoundaryError, CONTEXT_PRESSURE
    try:
        raise LlamaCompletionBoundaryError('49422 input tokens exceed 32768 context',
            kind=CONTEXT_PRESSURE, prompt_tokens=49422)
    except LlamaCompletionBoundaryError as exc:
        raise RuntimeError('backend rejected exact context') from exc


def test_context_pressure_recovers_without_spending_compiler_repair(monkeypatch, tmp_path):
    module, context, spec, observations = load_author(monkeypatch, tmp_path)
    monkeypatch.setattr(module, '_evidence', lambda *a, **k: _large_api_context())
    context['feature_work_order'] = {'acceptance': {'max_compiler_repair_attempts': 0}}
    class PressureRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            if len(self.calls) == 1:
                _raise_context_pressure()
            return GOOD
    router = PressureRouter([])
    assert create(module, context, spec, router) == GOOD
    assert len(router.calls) == 2
    first = json.loads(router.calls[0][1][1]['content'])
    retried = json.loads(router.calls[1][1][1]['content'])
    assert first['requirement'] == retried['requirement'] == spec['requirement']
    assert retried['target'] == context['target']
    assert retried['feature_work_order'] == context['feature_work_order']
    assert len(json.dumps(router.calls[1][1])) < len(json.dumps(router.calls[0][1]))
    assert [row[0] for row in observations] == ['parsed', 'compiled', 'coverage']



def _raise_output_exhaustion():
    from minecraft_mod_ai.llama_finish_reason_contract import (
        LlamaCompletionBoundaryError, OUTPUT_EXHAUSTED)
    try:
        raise LlamaCompletionBoundaryError(
            'native llama-server exhausted bounded output allowance',
            kind=OUTPUT_EXHAUSTED,
            partial_message={'content': 'package demo; public class ModUnit { /* unfinished'},
            prompt_tokens=23824, completion_tokens=8944, max_tokens=8944)
    except LlamaCompletionBoundaryError as exc:
        raise RuntimeError('backend stopped before the Java source completed') from exc


def test_output_exhaustion_compacts_optional_api_once_without_accepting_partial(monkeypatch, tmp_path):
    module, context, spec, observations = load_author(monkeypatch, tmp_path)
    monkeypatch.setattr(module, '_evidence', lambda *a, **k: _large_api_context())
    context['feature_work_order'] = {'acceptance': {'max_compiler_repair_attempts': 0}}

    class ExhaustingRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            if len(self.calls) == 1:
                _raise_output_exhaustion()
            return GOOD

    router = ExhaustingRouter([])
    assert create(module, context, spec, router) == GOOD
    assert len(router.calls) == 2
    assert len(json.dumps(router.calls[1][1])) < len(json.dumps(router.calls[0][1]))
    assert all('unfinished' not in row['content'] for row in router.calls[1][1])
    assert [row[0] for row in observations] == ['parsed', 'compiled', 'coverage']


def test_output_exhaustion_is_not_retried_without_reducible_api(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)

    class ExhaustingRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            _raise_output_exhaustion()

    router = ExhaustingRouter([])
    with pytest.raises(RuntimeError, match='backend stopped before'):
        create(module, context, spec, router)
    assert len(router.calls) == 1
    assert not list((tmp_path / '.mmm' / 'construction_files').glob('*.json'))


def test_output_exhaustion_has_only_one_recovery_even_if_api_can_shrink_again(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    monkeypatch.setattr(module, '_evidence', lambda *a, **k: _large_api_context())

    class ExhaustingRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            _raise_output_exhaustion()

    router = ExhaustingRouter([])
    with pytest.raises(RuntimeError, match='backend stopped before'):
        create(module, context, spec, router)
    assert len(router.calls) == 2
    assert len(json.dumps(router.calls[1][1])) < len(json.dumps(router.calls[0][1]))
    assert not list((tmp_path / '.mmm' / 'construction_files').glob('*.json'))


def test_compiler_repairs_do_not_accumulate_old_source_and_api_history(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    second = BAD.replace('INVALID_API()', 'INVALID_API(1)')
    router = Router([BAD, second, GOOD])
    assert create(module, context, spec, router) == GOOD
    repairs = [json.loads(row['content']) for row in router.calls[2][1]
               if row['role'] == 'user' and json.loads(row['content']).get('repair_complete_file')]
    assert len(repairs) == 1
    assert repairs[0]['previous_source'] == second
    assert 'cannot find symbol' in repairs[0]['error']


def test_pressure_during_repair_preserves_complete_failed_source(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    monkeypatch.setattr(module, '_evidence', lambda *a, **k: _large_api_context())
    class RepairPressureRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            if len(self.calls) == 1:
                return BAD
            if len(self.calls) == 2:
                _raise_context_pressure()
            return GOOD
    router = RepairPressureRouter([])
    assert create(module, context, spec, router) == GOOD
    repair = json.loads(router.calls[-1][1][-1]['content'])
    assert repair['previous_source'] == BAD
    assert 'cannot find symbol' in repair['error']


def test_diagnostic_dedup_preserves_other_initial_contracts_for_same_class(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    initial = _large_api_context()
    owner = 'example.Api0'
    initial['class_members'][0]['members'][0]['name'] = 'initialFactory'
    monkeypatch.setattr(module, '_evidence', lambda *a, **k: initial)
    diagnostic = {'verified_class': owner, 'declaration': initial['class_members'][0]['declaration'],
                  'members': [{'kind': 'METHOD', 'owner': owner, 'name': 'correctedCall',
                               'parameters': ['int'], 'return_type': 'void'}]}
    monkeypatch.setattr(module, '_diagnostic_api_evidence', lambda *a, **k: [diagnostic])
    router = Router([BAD, GOOD])
    assert create(module, context, spec, router) == GOOD
    current = [json.loads(row['content']) for row in router.calls[-1][1] if row['role'] == 'user']
    records = [*current[0]['inspected_target_api']['class_members'],
               *current[-1]['verified_symbols_from_compiler_failure']]
    matching = [row for row in records if row['verified_class'] == owner]
    assert len(matching) == 1
    assert {'initialFactory', 'correctedCall'} <= {row['name'] for row in matching[0]['members']}


def test_second_repair_retains_discovery_and_prior_verified_contracts(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    initial = _large_api_context()
    owner = 'example.Api0'
    initial['class_members'][0]['members'][0]['name'] = 'initialFactory'
    monkeypatch.setattr(module, '_evidence', lambda *a, **k: initial)
    def diagnostic(_context, source, _error):
        return [{'verified_class': owner,
                 'declaration': initial['class_members'][0]['declaration'],
                 'members': [{'kind': 'METHOD', 'owner': owner,
                              'name': 'repairTwo' if 'INVALID_API(1)' in source else 'repairOne',
                              'parameters': [], 'return_type': 'void'}]}]
    monkeypatch.setattr(module, '_diagnostic_api_evidence', diagnostic)
    router = Router([BAD, BAD.replace('INVALID_API()', 'INVALID_API(1)'), GOOD])
    assert create(module, context, spec, router) == GOOD
    current = [json.loads(row['content']) for row in router.calls[-1][1] if row['role'] == 'user']
    records = [*current[0]['inspected_target_api']['class_members'],
               *current[-1]['verified_symbols_from_compiler_failure']]
    matching = [row for row in records if row['verified_class'] == owner]
    assert len(matching) == 1
    assert {'initialFactory', 'repairOne', 'repairTwo'} <= {row['name'] for row in matching[0]['members']}


def test_context_recovery_never_retries_an_unchanged_mandatory_prompt(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    class NoRoomRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            _raise_context_pressure()
    router = NoRoomRouter([])
    with pytest.raises(RuntimeError, match='backend rejected exact context'):
        create(module, context, spec, router)
    assert len(router.calls) == 1
    assert not list((tmp_path / '.mmm' / 'construction_files').glob('*.json'))


def test_exhausted_canonical_context_recovery_is_not_started_again(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    monkeypatch.setattr(module, '_evidence', lambda *a, **k: _large_api_context())
    from minecraft_mod_ai.llama_finish_reason_contract import (
        LlamaCompletionBoundaryError, CONTEXT_PRESSURE, mark_context_recovery_exhausted)
    error = LlamaCompletionBoundaryError('already handled', kind=CONTEXT_PRESSURE)
    mark_context_recovery_exhausted(error)
    class ExhaustedRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            raise error
    router = ExhaustedRouter([])
    with pytest.raises(LlamaCompletionBoundaryError, match='already handled'):
        create(module, context, spec, router)
    assert len(router.calls) == 1


def test_non_context_backend_error_is_not_retried(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    class UnavailableRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            raise RuntimeError('server unavailable')
    router = UnavailableRouter([])
    with pytest.raises(RuntimeError, match='server unavailable'):
        create(module, context, spec, router)
    assert len(router.calls) == 1


def test_context_compaction_preserves_generic_factories_and_referenced_overloads():
    from minecraft_mod_ai.construction_file_prompt import compact_api_observations
    api = _large_api_context()
    factory = {'kind': 'METHOD', 'owner': 'example.Api0', 'name': 'fromConfig',
               'static': True, 'return_type': 'example.Api0<java.lang.String>',
               'parameters': ['example.Config<java.lang.String>']}
    overloads = [{'kind': 'METHOD', 'owner': 'example.Api0', 'name': 'wanted',
                  'parameters': [type_], 'return_type': 'void'}
                 for type_ in ('int', 'java.lang.String')]
    api['class_members'][0]['members'].extend([*overloads, factory])
    source = 'package demo; class ModUnit { void f() { api.wanted(1); } }'
    prompt = {'requirement': '정확한 전체 요구사항', 'previous_source': source,
              'inspected_target_api': api}
    messages = [{'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False)}]
    result = compact_api_observations(messages)
    assert result is not None
    current = json.loads(result[0]['content'])
    assert current['previous_source'] == source
    assert current['requirement'] == prompt['requirement']
    members = current['inspected_target_api']['class_members'][0]['members']
    assert factory in members
    assert all(row in members for row in overloads)


def test_module_binding_values_reach_the_complete_file_author(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    bindings = {'module_config_json': [{'key': 'max_health', 'value': 100.0},
                                     {'key': 'movement_speed', 'value': 0.25}]}
    context['binding_pages'] = bindings
    router = Router([GOOD])
    assert create(module, context, spec, router) == GOOD
    prompt = json.loads(router.calls[0][1][1]['content'])
    assert prompt['construction_bindings'] == bindings


def test_feature_entrypoint_keeps_gameplay_api_intent(monkeypatch, tmp_path):
    """A file's registration leaf cannot suppress its requested teleport/trade actions."""
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    context['authoring_contract'] = {
        'canonical_leaf': 'minecraft/entity/registry',
        'responsibility': 'Register supplied entities',
        'host_requirements': {'symbols': ['EntityType']},
    }
    writer = sys.modules['minecraft_mod_ai.construction_direct_writer']
    monkeypatch.setattr(writer, '_evidence', lambda index, intent, pages: {
        'target_api': [], 'source': [], 'lookup_purpose': intent,
    })
    request = 'Teleport the player and trade with the alien'
    kwargs = {'target_path': spec['target_path'],
              'owner': 'demo.ModUnit', 'router': Router([])}
    registration = module._evidence(context, request, **kwargs)
    entrypoint = module._evidence(context, request, feature_entrypoint=True, **kwargs)
    assert registration['lookup_purpose'].startswith('register entity in registry')
    assert 'Teleport' not in registration['lookup_purpose']
    assert entrypoint['lookup_purpose'] == request


def _raise_repeated_stream_output():
    from minecraft_mod_ai.llama_completion_liveness_contract import LlamaDegenerateOutputError
    try:
        raise LlamaDegenerateOutputError(
            "LLAMA_DEGENERATE_REPEATED_OUTPUT: repeated 87-character pattern")
    except LlamaDegenerateOutputError as exc:
        raise RuntimeError("wrapped model stream failed") from exc


def test_repeated_stream_decoding_retries_only_with_reduced_evidence(monkeypatch, tmp_path):
    module, context, spec, observations = load_author(monkeypatch, tmp_path)
    monkeypatch.setattr(module, '_evidence', lambda *a, **k: _large_api_context())

    class RepeatingRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            if len(self.calls) == 1:
                _raise_repeated_stream_output()
            return GOOD

    router = RepeatingRouter([])
    assert create(module, context, spec, router) == GOOD
    assert len(router.calls) == 2
    assert len(json.dumps(router.calls[1][1])) < len(json.dumps(router.calls[0][1]))
    assert [entry[0] for entry in observations] == ['parsed', 'compiled', 'coverage']


def test_repeated_stream_without_reducible_api_fails_atomically(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)

    class RepeatingRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            _raise_repeated_stream_output()

    router = RepeatingRouter([])
    with pytest.raises(RuntimeError, match="wrapped model stream failed"):
        create(module, context, spec, router)
    assert len(router.calls) == 1
    assert not list((tmp_path / '.mmm' / 'construction_files').glob('*.json'))


def test_repeated_stream_recovery_never_replays_unbounded(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    monkeypatch.setattr(module, '_evidence', lambda *a, **k: _large_api_context())

    class RepeatingRouter(Router):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, list(messages), kwargs))
            _raise_repeated_stream_output()

    router = RepeatingRouter([])
    with pytest.raises(RuntimeError, match="wrapped model stream failed"):
        create(module, context, spec, router)
    assert len(router.calls) == 2
    assert not list((tmp_path / '.mmm' / 'construction_files').glob('*.json'))


def test_registry_leaf_queries_true_action_not_generic_implement(monkeypatch, tmp_path):
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    context['authoring_contract'] = {
        'canonical_leaf': 'minecraft/entity/registry',
        'responsibility': 'Implement only the registry responsibility of the supplied entity artifact.',
        'host_requirements': {'symbols': ['EntityType']},
    }
    intent = module._file_api_intent(context, 'Fight aliens and colonize planets.')
    assert intent.startswith('register entity in registry')
    assert 'EntityType' in intent
    assert 'Implement' not in intent and 'Fight aliens' not in intent
    assert module._file_api_intent(context, 'Fight aliens', feature_entrypoint=True) == 'Fight aliens'


def test_compiler_diagnostic_precedes_placeholder_when_both_are_broken(monkeypatch, tmp_path):
    """Do not mask a fake Minecraft method behind a generic TODO exception."""
    module, context, spec, observations = load_author(monkeypatch, tmp_path)
    source = BAD.replace('INVALID_API()', 'INVALID_API(); throw new UnsupportedOperationException()')
    router = Router([source, GOOD])
    assert create(module, context, spec, router) == GOOD
    assert 'compiled' in [row[0] for row in observations]
    repair = json.loads(router.calls[1][1][-1]['content'])
    assert 'cannot find symbol INVALID_API' in repair['error']
    assert 'CONSTRUCTION_FILE_UNIMPLEMENTED_PLACEHOLDER' not in repair['error']
    assert repair['previous_source'] == source.rstrip() + '\n'


def test_registry_author_prompt_scopes_feature_goal_to_leaf(monkeypatch, tmp_path):
    """The registry file cannot be tasked with the whole combat/colony feature."""
    module, context, spec, _ = load_author(monkeypatch, tmp_path)
    spec['requirement'] = 'Fight aliens and colonize planets.'
    context['authoring_contract'] = {
        'canonical_leaf': 'minecraft/entity/registry',
        'responsibility': 'Implement only the registry responsibility of the supplied entity artifact.',
        'host_requirements': {'symbols': ['EntityType']},
    }
    router = Router([GOOD])
    assert create(module, context, spec, router) == GOOD
    prompt = json.loads(router.calls[0][1][1]['content'])
    assert prompt['requirement'] == context['authoring_contract']['responsibility']
    assert prompt['original_feature_goal'] == spec['requirement']
    assert prompt['construction_requirements']['canonical_leaf'] == 'minecraft/entity/registry'
