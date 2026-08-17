from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import pytest
from minecraft_mod_ai.production_contract import ProductionContractError, compile_production_contract, evaluate_quality_contract
from minecraft_mod_ai.quality_evidence import compile_quality_evidence
PROPOSAL_HASH = 'sha256:' + 'a' * 64

def _digest(character: str) -> str:
    return 'sha256:' + character * 64

def _research_design() -> dict:
    return {'title': 'Evidence fixture', '_technology_radar': {'schema_version': 'mmm/technology-radar-page-v1', 'aggregate_schema_version': 'mmm/technology-radar-aggregate-v1', 'radar_sha256': _digest('b'), 'requirements': [{'requirement_id': 'fabric-target'}], 'pagination': {'offset': 0, 'page_size': 50, 'returned': 1, 'total_requirements': 1, 'next_cursor': '', 'pages_collected': 1, 'complete': True}, 'collection_receipt': {'schema_version': 'mmm/technology-page-collection-receipt-v1', 'page_count': 1, 'pages_sha256': _digest('c')}}, '_ecosystem_discovery': {'schema_version': 'mmm/ecosystem-seed-bundle-v1', 'aggregate_schema_version': 'mmm/ecosystem-seed-aggregate-v1', 'status': 'available', 'route_sha256': _digest('d'), 'route_count': 2, 'processed_route_count': 2, 'remaining_route_count': 0, 'next_route_cursor': '', 'routes_complete': True, 'errors': [], 'collection_receipt': {'schema_version': 'mmm/ecosystem-route-collection-receipt-v1', 'route_page_count': 1, 'route_pages_sha256': _digest('e')}}, '_technical_evidence': {'schema_version': 'mmm/central-evidence-graph-v1', 'evidence_sha256': _digest('f'), 'domains': [{'domain_id': 'fabric'}], 'unresolved_official_domains': []}}

def _module(module_id: str, kind: str='item') -> dict:
    return {'module_id': module_id, 'kind': kind, 'config': {}, 'depends_on': [], 'required_gates': []}

def _baseline_inputs(tmp_path: Path) -> dict:
    report = tmp_path / 'gametest.xml'
    report.write_text('<testsuite tests="2" failures="0" errors="0" skipped="0"><testcase name="registry"/><testcase name="behavior"/></testsuite>', encoding='utf-8')
    jar = tmp_path / 'mod.jar'
    jar.write_bytes(b'independently validated jar')
    return {'source_validation': {'status': 'PASS', 'checks_run': 18, 'findings': [], 'observed_at': '2026-08-02T01:00:00+09:00'}, 'build_report': {'status': 'PASS', 'gradle_version': '8.5', 'commands': [{'name': 'clean_build', 'exit_code': 0, 'timed_out': False}, {'name': 'gametest', 'exit_code': 0, 'timed_out': False}], 'jar_path': str(jar), 'gametest_report': str(report), 'observed_at': '2026-08-02T01:01:00+09:00'}, 'jar_validation': {'status': 'PASS', 'checks_run': 12, 'findings': []}, 'runtime_receipt': {'prepared': {'schema_version': 'mmm/runtime-instance-v1', 'minecraft_version': '1.20.1', 'disposable': True}, 'server': {'schema_version': 'mmm/runtime-status-v1', 'server_running': True, 'server_log_lines': 34}}, 'playtest_receipt': {'schema_version': 'mmm/playtest-result-v3', 'status': 'PASS', 'interaction_count': 1, 'assertion_count': 1, 'results': [{'action': 'connect'}, {'action': 'use', 'timestamp': '2026-08-02T00:00:01Z'}, {'action': 'wait_for', 'matched': True}]}}

def test_baseline_receipts_are_objective_stable_and_evaluator_compatible(tmp_path: Path) -> None:
    design = _research_design()
    compiled = _compile(design)
    inputs = _baseline_inputs(tmp_path)
    first = _compile_call(compiled, design, inputs)
    assert set(first) == {'correctness', 'build', 'research', 'runtime'}
    assert all((value['status'] == 'PASS' for value in first.values()))
    assert all((value['producer'] != value['verified_by'] for value in first.values()))
    assert all(('+00:00' in value['observed_at'] for value in first.values()))
    changed_time = copy.deepcopy(inputs)
    changed_time['source_validation']['observed_at'] = '2026-08-03T22:30:00-04:00'
    changed_time['build_report']['observed_at'] = '2026-08-04T03:00:00+00:00'
    changed_time['playtest_receipt']['results'][1]['timestamp'] = '2030-01-01T00:00:00Z'
    second = _compile_call(compiled, design, changed_time)
    assert {key: value['receipt_id'] for key, value in first.items()} == {key: value['receipt_id'] for key, value in second.items()}
    assert {key: value['evidence_refs'] for key, value in first.items()} == {key: value['evidence_refs'] for key, value in second.items()}
    report = evaluate_quality_contract(compiled.contract, first, PROPOSAL_HASH)
    assert report['overall_status'] == 'PASS'

@pytest.mark.parametrize('mutator', [lambda design: design['_technology_radar']['pagination'].update({'complete': False, 'next_cursor': 'next'}), lambda design: design['_ecosystem_discovery']['errors'].append({'provider': 'modrinth', 'error_type': 'timeout'}), lambda design: design['_technical_evidence']['unresolved_official_domains'].append('networking')])
def test_research_pass_requires_exhaustion_and_no_unresolved_evidence(tmp_path: Path, mutator) -> None:
    design = _research_design()
    mutator(design)
    compiled = _compile(design)
    evidence = _compile_call(compiled, design, _baseline_inputs(tmp_path))
    assert 'research' not in evidence

def test_completion_claim_is_not_evidence_and_large_receipt_sets_are_exhausted() -> None:
    design = {'title': 'Scale test'}
    compiled = _compile(design, prompt='Run a game-scale performance workload.')
    performance = {'schema_version': 'mmm/performance-validation-v1', 'status': 'PASS', 'sample_count': 10000, 'skipped_work_count': 0, 'budgets': {'tick_ms': {'observed': 42, 'limit': 50, 'comparison': 'lte'}}}
    receipts = tuple(({'module_id': f'module_{index:05d}', 'records': []} for index in range(10000))) + ({'nested': {'performance_validation': performance}},)
    empty_inputs = {'source_validation': None, 'build_report': None, 'jar_validation': None, 'runtime_receipt': None, 'playtest_receipt': None, 'module_receipts': receipts}
    evidence = _compile_call(compiled, design, empty_inputs)
    assert set(evidence) == {'performance'}
    assert evidence['performance']['producer'] != evidence['performance']['verified_by']
    claimed = copy.deepcopy(performance)
    claimed['complete'] = True
    rejected = _compile_call(compiled, design, {**empty_inputs, 'module_receipts': ({'performance_validation': claimed},)})
    assert 'performance' not in rejected

def test_game_design_must_match_the_contract_binding(tmp_path: Path) -> None:
    design = _research_design()
    compiled = _compile(design)
    changed = copy.deepcopy(design)
    changed['title'] = 'Different design'
    with pytest.raises(ProductionContractError, match='game_design'):
        _compile_call(compiled, changed, _baseline_inputs(tmp_path))
