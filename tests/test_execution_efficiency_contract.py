from __future__ import annotations
import inspect
from types import SimpleNamespace
from minecraft_mod_ai import execution_efficiency_contract, work_graph
from minecraft_mod_ai.complete_spec import ProductionModule

def _raw_module(deliverable: str) -> dict[str, object]:
    return {'module_id': deliverable, 'kind': 'item', 'config': {}, 'depends_on': [], 'required_gates': [], 'implements_deliverables': [deliverable]}

def test_module_shards_use_dependency_ready_waves() -> None:
    independent_content = ProductionModule(module_id='a_content', kind='item')
    dependent_entity = ProductionModule(module_id='b_dependent_entity', kind='entity', depends_on=('a_content',))
    independent_entity = ProductionModule(module_id='z_independent_entity', kind='entity')
    ordered = work_graph._topological_modules((independent_content, dependent_entity, independent_entity))
    assert [value.module_id for value in ordered] == ['a_content', 'b_dependent_entity', 'z_independent_entity']
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(ordered, policy=policy))
    shaped = [(stage, [module.module_id for module in members]) for stage, members in shards]
    assert shaped == [('content', ['a_content']), ('entity', ['z_independent_entity']), ('entity', ['b_dependent_entity'])]

def test_custom_llm_modules_are_bounded_without_one_node_per_module(monkeypatch) -> None:
    monkeypatch.setenv('MMM_LLAMA_ACTIVE_PARALLEL', '1')
    modules = tuple((ProductionModule(module_id=f'custom_{index:03d}', kind='custom_java') for index in range(100)))
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(modules, policy=policy))
    assert [stage for stage, _ in shards] == ['custom', 'custom', 'custom']
    assert [len(members) for _, members in shards] == [48, 48, 4]
    assert sum((len(members) for _, members in shards)) == 100

def test_small_custom_wave_uses_all_selected_llm_slots_without_row_explosion(monkeypatch) -> None:
    monkeypatch.setenv('MMM_LLAMA_ACTIVE_PARALLEL', '3')
    modules = tuple((ProductionModule(module_id=f'custom_{index:02d}', kind='custom_java') for index in range(10)))
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(modules, policy=policy))
    assert [stage for stage, _ in shards] == ['custom', 'custom', 'custom']
    assert [len(members) for _, members in shards] == [4, 4, 2]
    assert sum((len(members) for _, members in shards)) == 10

def test_large_custom_wave_keeps_java_shard_ceiling(monkeypatch) -> None:
    monkeypatch.setenv('MMM_LLAMA_ACTIVE_PARALLEL', '3')
    modules = tuple((ProductionModule(module_id=f'custom_{index:03d}', kind='custom_java') for index in range(200)))
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(modules, policy=policy))
    sizes = [len(members) for stage, members in shards if stage == 'custom']
    assert sizes == [48, 48, 48, 48, 8]
    assert max(sizes) <= 48
    assert sum(sizes) == 200

def test_long_serial_dependency_chain_is_compressed_into_bounded_shards() -> None:
    modules: list[ProductionModule] = []
    for index in range(120):
        module_id = f'chain_{index:03d}'
        depends_on = () if index == 0 else (f'chain_{index - 1:03d}',)
        modules.append(ProductionModule(module_id=module_id, kind='item', depends_on=depends_on))
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(tuple(modules), policy=policy))
    assert [stage for stage, _ in shards] == ['content', 'content', 'content']
    assert [len(members) for _, members in shards] == [48, 48, 24]
    assert sum((len(members) for _, members in shards)) == 120

def test_dependency_sharding_uses_indexed_lookup_not_all_group_reverse_scan() -> None:
    source = inspect.getsource(execution_efficiency_contract._dependency_wave_shards)
    assert 'open_by_key' in source
    assert 'for index in range(len(groups) - 1' not in source

def test_validation_resume_reuses_only_stable_exact_results(monkeypatch) -> None:
    from minecraft_mod_ai import complete_orchestrator
    captured: dict[str, object] = {}

    def fake_checkpoint(ledger, checkpoint_id, *, stage, input_value, action, encode, decode, validate_cached=None):
        captured.update(checkpoint_id=checkpoint_id, input_value=input_value, validate_cached=validate_cached)
        return {'status': 'captured'}
    monkeypatch.setattr(complete_orchestrator, 'run_named_checkpoint', fake_checkpoint)
    execution_efficiency_contract._install_validation_checkpoint_reuse(complete_orchestrator)
    result = complete_orchestrator.run_named_checkpoint(object(), 'validate-source', stage='validate:source', input_value={'graph_hash': 'g', 'project_manifest': 'm'}, action=lambda: {'status': 'PASS'}, encode=lambda value: value, decode=lambda value: value, validate_cached=lambda _cached: False)
    assert result == {'status': 'captured'}
    scoped = captured['input_value']
    assert isinstance(scoped, dict)
    assert str(scoped['_mmm_validation_implementation']).startswith('sha256:')
    validator = captured['validate_cached']
    assert callable(validator)
    assert validator({'status': 'PASS'}) is True
    assert validator({'status': 'FAIL'}) is False

def test_jdt_resume_never_reuses_unavailable_result(monkeypatch) -> None:
    from minecraft_mod_ai import complete_orchestrator
    captured: dict[str, object] = {}

    def fake_checkpoint(ledger, checkpoint_id, *, stage, input_value, action, encode, decode, validate_cached=None):
        captured['validate_cached'] = validate_cached
        return {'status': 'captured'}
    monkeypatch.setattr(complete_orchestrator, 'run_named_checkpoint', fake_checkpoint)
    execution_efficiency_contract._install_validation_checkpoint_reuse(complete_orchestrator)
    complete_orchestrator.run_named_checkpoint(object(), 'validate-jdt', stage='validate:jdt', input_value={'graph_hash': 'g', 'project_manifest': 'm'}, action=lambda: {}, encode=lambda value: value, decode=lambda value: value, validate_cached=lambda _cached: False)
    validator = captured['validate_cached']
    assert callable(validator)
    assert validator({'status': 'UNAVAILABLE', 'error': 'jdtls missing'}) is False
    assert validator({'schema_version': 'mmm/java-diagnostics-v2', 'diagnostics': {}}) is True

def test_validation_checkpoint_scope_changes_with_mmm_runtime_policy(monkeypatch) -> None:
    monkeypatch.setenv('MMM_VALIDATION_CACHE_TEST_SCOPE', 'one')
    first = execution_efficiency_contract._validation_implementation_fingerprint('validate-jdt')
    monkeypatch.setenv('MMM_VALIDATION_CACHE_TEST_SCOPE', 'two')
    second = execution_efficiency_contract._validation_implementation_fingerprint('validate-jdt')
    assert first != second
