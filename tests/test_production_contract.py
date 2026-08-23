from __future__ import annotations
import copy
import json
import pytest
from minecraft_mod_ai.production_contract import ProductionContractError, compile_production_contract, evaluate_quality_contract, persist_quality_report, quality_contract_summary, quality_unresolved, validate_production_contract
PROPOSAL_HASH = 'sha256:' + 'a' * 64

def _module(module_id: str, kind: str='custom_java', **config: object) -> dict:
    return {'module_id': module_id, 'kind': kind, 'config': config, 'depends_on': [], 'required_gates': []}

def _compile(**overrides: object):
    values = {'requested_prompt': 'Add a weather compass with a configurable display.', 'game_design': {'title': 'Weather compass', 'systems': [{'goal': 'Show the current weather state.'}]}, 'research_brief': {'claims': [{'statement': 'Use the supported item registration API.', 'url': 'https://example.invalid/not-a-requirement', 'evidence_sha256': 'sha256:' + 'b' * 64}]}, 'modules': [_module('weather_compass', 'item')], 'acceptance_tests': ['The compass reports the observed weather.']}
    values.update(overrides)
    return compile_production_contract(**values)

def _receipt(dimension_id: str, iteration: int, *, status: str='PASS', failure_signature: str='') -> dict[str, object]:
    receipt: dict[str, object] = {'dimension_id': dimension_id, 'route_ref': f'evidence:{dimension_id}', 'status': status, 'proposal_hash': PROPOSAL_HASH, 'receipt_id': f'{dimension_id}-{iteration}', 'observed_at': f'2026-08-02T00:{iteration:02d}:00+00:00', 'producer': 'production-worker', 'verified_by': 'host-quality-gate', 'evidence_refs': [f'artifact://{dimension_id}/{iteration}']}
    if failure_signature:
        receipt['failure_signature'] = failure_signature
    return receipt

def _all_receipts(contract: dict, iteration: int) -> list[dict[str, object]]:
    return [_receipt(item['dimension_id'], iteration) for item in contract['quality_dimension_catalog']]

def test_compile_is_deterministic_and_traces_every_source_item() -> None:
    first = _compile()
    second = _compile()
    assert first == second
    assert first.contract['contract_sha256'].startswith('sha256:')
    requirements = first.contract['requirement_catalog']
    assert requirements[0]['source'] == 'requested_prompt'
    assert requirements[0]['statement'] == 'Add a weather compass with a configurable display.'
    assert {item['source'] for item in requirements} == {'requested_prompt', 'game_design', 'research_brief'}
    assert all((item['coverage_group_ref'] for item in requirements))
    assert all((group['implementation_catalog_ref'] == 'catalog:implementations' and group['acceptance_refs'] and group['evidence_route_refs'] for group in first.contract['coverage_groups']))
    assert not any(('example.invalid' in item['statement'] for item in requirements))
    validate_production_contract(first.contract, ['weather_compass'], first.acceptance_tests)

def test_hash_binding_rejects_mutation_and_external_binding_mismatch() -> None:
    compiled = _compile()
    tampered = copy.deepcopy(compiled.contract)
    tampered['requirement_catalog'][0]['statement'] = 'A different request'
    with pytest.raises(ProductionContractError, match='sha256'):
        validate_production_contract(tampered, ['weather_compass'], compiled.acceptance_tests)
    with pytest.raises(ProductionContractError, match='module catalog'):
        validate_production_contract(compiled.contract, ['different_module'], compiled.acceptance_tests)
    with pytest.raises(ProductionContractError, match='acceptance catalog'):
        validate_production_contract(compiled.contract, ['weather_compass'], ['invented pass'])

def test_complete_module_and_asset_payloads_are_bound_to_current_content() -> None:
    module = _module('weather_compass', 'item', feature='weather')
    asset = {
        'asset_id': 'weather_compass_texture',
        'kind': 'item_texture',
        'prompt': 'weather compass icon',
        'target_path': 'assets/example/textures/item/weather_compass.png',
        'width': 16,
        'height': 16,
    }
    compiled = _compile(modules=[module], assets=[asset])
    validate_production_contract(
        compiled.contract,
        [module],
        compiled.acceptance_tests,
        [asset],
    )
    changed_module = copy.deepcopy(module)
    changed_module['config']['changed'] = True
    with pytest.raises(ProductionContractError, match='module source binding'):
        validate_production_contract(
            compiled.contract,
            [changed_module],
            compiled.acceptance_tests,
            [asset],
        )
    changed_asset = {**asset, 'prompt': 'different texture'}
    with pytest.raises(ProductionContractError, match='asset source binding'):
        validate_production_contract(
            compiled.contract,
            [module],
            compiled.acceptance_tests,
            [changed_asset],
        )

def test_evaluation_requires_fresh_independent_receipts() -> None:
    compiled = _compile()
    missing = evaluate_quality_contract(compiled.contract, {}, PROPOSAL_HASH)
    assert missing['overall_status'] == 'MISSING'
    assert set(quality_unresolved(missing)) == {'correctness', 'build', 'research', 'runtime'}
    self_certified = _receipt('correctness', 1)
    self_certified['complete'] = True
    rejected = evaluate_quality_contract(compiled.contract, {'correctness': self_certified}, PROPOSAL_HASH)
    result = rejected['dimensions'][0]
    assert result['status'] == 'MISSING'
    assert 'self-certified' in result['reason']
    passed = evaluate_quality_contract(compiled.contract, _all_receipts(compiled.contract, 1), PROPOSAL_HASH)
    assert passed['overall_status'] == 'PASS'
    assert quality_unresolved(passed) == ()
    stale = evaluate_quality_contract(compiled.contract, _all_receipts(compiled.contract, 1), PROPOSAL_HASH, previous=passed)
    assert stale['overall_status'] == 'MISSING'
    assert all((item['status'] == 'MISSING' for item in stale['dimensions']))
    assert all(('stale receipt' in item['reason'] for item in stale['dimensions']))

def test_three_fresh_identical_failures_detect_a_plateau() -> None:
    compiled = _compile()
    dimensions = [item['dimension_id'] for item in compiled.contract['quality_dimension_catalog']]

    def evidence(iteration: int) -> list[dict[str, object]]:
        return [_receipt(dimension_id, iteration, status='FAIL' if dimension_id == 'runtime' else 'PASS', failure_signature='runtime:missing-registry-entry' if dimension_id == 'runtime' else '') for dimension_id in dimensions]
    first = evaluate_quality_contract(compiled.contract, evidence(1), PROPOSAL_HASH)
    second = evaluate_quality_contract(compiled.contract, evidence(2), PROPOSAL_HASH, previous=first)
    third = evaluate_quality_contract(compiled.contract, evidence(3), PROPOSAL_HASH, previous=second)
    runtime = next((item for item in third['dimensions'] if item['dimension_id'] == 'runtime'))
    assert runtime['status'] == 'FAIL'
    assert runtime['failure_streak'] == 3
    assert runtime['plateau'] is True
    assert third['plateau'] == {'detected': True, 'dimension_ids': ['runtime'], 'identical_failure_threshold': 3}

def test_quality_report_is_persisted_as_valid_canonical_json(tmp_path) -> None:
    compiled = _compile()
    report = evaluate_quality_contract(compiled.contract, _all_receipts(compiled.contract, 1), PROPOSAL_HASH)
    target = persist_quality_report(tmp_path / 'quality' / 'report.json', report)
    assert json.loads(target.read_text(encoding='utf-8')) == report
    assert target.read_text(encoding='utf-8').endswith('\n')

def test_summary_is_human_readable_and_does_not_expose_hashes() -> None:
    compiled = _compile()
    summary = quality_contract_summary(compiled.contract)
    assert 'request-derived requirements' in summary
    assert 'independent verifier' in summary
    assert 'sha256' not in summary.casefold()

def test_thousands_of_requirements_and_modules_have_linear_bounded_links() -> None:
    count = 2000
    modules = [_module(f'system_{index:05d}', feature_key=f'feature_{index:05d}') for index in range(count)]
    design = {'requirements': [{'statement': f'feature_{index:05d} behaves as requested'} for index in range(count)]}
    compiled = compile_production_contract(requested_prompt='Implement every explicitly listed independent system.', game_design=design, research_brief=None, modules=modules, acceptance_tests=[])
    contract = compiled.contract
    assert contract['catalog_stats']['requirements'] == count + 1
    assert contract['catalog_stats']['implementations'] == count
    assert len(contract['coverage_groups']) == count + 1
    assert max((len(group['implementation_refs']) for group in contract['coverage_groups'])) <= 8
    assert all((group['implementation_catalog_ref'] == 'catalog:implementations' for group in contract['coverage_groups']))
    assert {item['implementation_id'] for item in contract['implementation_catalog']} == {f'system_{index:05d}' for index in range(count)}
    validate_production_contract(contract, [f'system_{index:05d}' for index in range(count)], compiled.acceptance_tests)
