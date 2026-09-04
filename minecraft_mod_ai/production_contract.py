from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .complete_spec import AssetRequest, ProductionModule
CONTRACT_SCHEMA = 'mmm/production-contract-v1'
REPORT_SCHEMA = 'mmm/quality-report-v1'
_SHA256 = re.compile('^sha256:[0-9a-f]{64}$')
_TOKEN = re.compile('[\\w]+', re.UNICODE)
_LEGACY_DIRECT_IMPLEMENTATION_REFS = 8
_MAX_MATCH_TERMS = 16
_MAX_POSTING_SCAN = 32
_PLATEAU_THRESHOLD = 3
_BASELINE_DIMENSIONS = ('correctness', 'build', 'research', 'runtime')
_DIMENSIONS: dict[str, dict[str, Any]] = {'correctness': {'title': 'Requirement correctness', 'objective': 'Every request-derived requirement is traceable to implementation and an observable acceptance check.', 'acceptance': 'Every request-derived requirement is linked to implemented output and independently observed behavior or artifacts.'}, 'build': {'title': 'Reproducible build', 'objective': 'The complete source builds from a clean checkout with the locked toolchain and declared dependencies.', 'acceptance': 'A clean, locked-toolchain build succeeds and its artifact inventory is recorded by the build verifier.'}, 'research': {'title': 'Research and compatibility', 'objective': 'Technical choices are supported by current, scoped evidence and checked for version, license, and provenance compatibility.', 'acceptance': 'Research-backed choices have current source receipts plus version, license, and provenance checks.'}, 'runtime': {'title': 'Runtime behavior', 'objective': 'The built result starts and completes representative requested scenarios without hidden planner assertions.', 'acceptance': 'The built result starts and representative requested scenarios pass under an external runtime verifier.'}, 'visual_3d': {'title': 'Visual, 3D, and animation quality', 'objective': 'Requested visual assets, models, animation, rendering, and effects load correctly and meet observable review criteria.', 'acceptance': 'Requested visual and 3D output loads without missing resources and passes deterministic render plus visual review checks.', 'terms': ('visual', 'texture', 'model', '3d', 'animation', 'shader', 'render', 'particle', 'sprite', 'image', '시각', '텍스처', '모델', '애니메이션', '셰이더', '렌더', '파티클', '이미지', '3차원'), 'asset_kinds': ('item', 'block', 'entity', 'gui', 'environment', 'icon')}, 'state_save_migration': {'title': 'State, saves, and migration', 'objective': 'Requested persistent state survives save/load and evolves without silently corrupting supported prior data.', 'acceptance': 'Persistent state passes round-trip, restart, corruption handling, and declared-version migration checks.', 'terms': ('save', 'persistence', 'persistent', 'migration', 'save upgrade', 'schema upgrade', 'nbt', 'codec', 'database', 'player state', '저장', '영속', '마이그레이션', '저장 데이터 업그레이드', '데이터 이전', '플레이어 상태')}, 'multiplayer': {'title': 'Multiplayer authority and synchronization', 'objective': 'Requested multiplayer behavior has explicit server authority, validated messages, and synchronized observable state.', 'acceptance': 'Requested multiplayer scenarios pass with multiple clients, server authority, message validation, and reconnect synchronization.', 'terms': ('multiplayer', 'network', 'server', 'client', 'packet', 'sync', 'party', 'guild', 'co-op', 'coop', '멀티플레이', '네트워크', '서버', '클라이언트', '패킷', '동기화', '파티', '길드', '협동'), 'module_kinds': ('networking', 'party', 'guild')}, 'performance': {'title': 'Performance and scale', 'objective': 'Explicit scale or software performance goals are measured against declared workloads, budgets, and regression baselines.', 'acceptance': 'Declared large-scale or software performance workloads meet recorded tick, latency, memory, and throughput budgets without skipped work.', 'phrases': ('game-scale', 'game scale', 'large-scale', 'large scale', 'massive', 'software performance', 'runtime performance', 'code performance', 'performance optimization', 'latency', 'throughput', 'frame rate', 'tick budget', 'memory budget', 'profiling', 'benchmark', 'stress test', '게임 수준', '대규모', '대형 모드', '코드 성능', '런타임 성능', '성능 최적화', '지연 시간', '처리량', '프레임', '틱 예산', '메모리 예산', '프로파일링', '벤치마크', '스트레스 테스트')}, 'accessibility': {'title': 'Accessibility and localization', 'objective': 'Requested accessibility and localization behavior remains usable across declared input, language, subtitle, and display paths.', 'acceptance': 'Declared accessibility and localization paths pass keyboard, subtitle, language, and readable-display checks as applicable.', 'terms': ('accessibility', 'subtitle', 'localization', 'language', 'colorblind', 'screen reader', 'key remap', '접근성', '자막', '현지화', '다국어', '언어', '색각', '스크린 리더', '키 변경')}}
_CONDITIONAL_ORDER = ('visual_3d', 'state_save_migration', 'multiplayer', 'performance', 'accessibility')
_COMPLETION_POLICY = {'owner': 'code', 'allowed_evidence_statuses': ['PASS', 'MISSING', 'FAIL'], 'all_dimensions_must_pass': True, 'fresh_receipt_required': True, 'proposal_hash_binding_required': True, 'independent_verifier_required': True, 'self_reported_completion_accepted': False, 'plateau_identical_failure_threshold': _PLATEAU_THRESHOLD}
_SKIP_SOURCE_KEYS = {'schema_version', 'approval_hash', 'brief_sha256', 'evidence_sha256', 'route_sha256', 'page_sha256', 'query_sha256', 'content_sha256', 'sha256', 'hash', 'url', 'uri', 'retrieved_at', 'created_at', 'updated_at', 'timestamp', 'next_cursor', 'cursor', 'offset'}
_PUBLIC_ACCEPTANCE_INTERNAL_MARKERS = (
    'all declared provides',
    'declared_provides',
    'owned anchor',
    'owned_anchor',
    'required gates',
    'required_gates',
    'task integrity',
    'task_sha256',
    'done_predicate',
)

class ProductionContractError(ValueError):
    """Raised when a code-owned production or quality contract is invalid."""

def _validate_public_acceptance(statement: str) -> None:
    """Reject implementation-internal invariants at the user-facing boundary."""
    if not isinstance(statement, str) or not statement.strip():
        raise ProductionContractError('public acceptance must be a non-empty string')
    folded = statement.casefold()
    if 'task_' in folded or any(marker in folded for marker in _PUBLIC_ACCEPTANCE_INTERNAL_MARKERS):
        matched_marker = (
            'task_'
            if 'task_' in folded
            else next(
                (marker for marker in _PUBLIC_ACCEPTANCE_INTERNAL_MARKERS if marker in folded),
                'unknown',
            )
        )
        raise ProductionContractError(
            'public acceptance contains internal task or integrity language: '
            f'marker={matched_marker!r}; value={folded!r}'
        )

def bound_game_design(game_design: Mapping[str, Any]) -> dict[str, Any]:
    """Return user/design content that is stable across execution decoration."""
    return {str(key): value for key, value in game_design.items() if not str(key).startswith('_')}

@dataclass(frozen=True)
class ProductionContractCompilation:
    contract: dict[str, Any]
    acceptance_tests: tuple[str, ...]

def compile_production_contract(requested_prompt: str, game_design: Mapping[str, Any], research_brief: Mapping[str, Any] | Sequence[Any] | None=None, modules: Sequence[ProductionModule | Mapping[str, Any]]=(), assets: Sequence[AssetRequest | Mapping[str, Any]]=(), acceptance_tests: Sequence[str]=(), evidence_plan: Mapping[str, Any] | None=None) -> ProductionContractCompilation:
    """Compile prompt-derived scope into a deterministic evidence contract."""
    if not isinstance(requested_prompt, str) or not requested_prompt.strip():
        raise ProductionContractError('requested_prompt must be a non-empty string')
    if not isinstance(game_design, Mapping):
        raise ProductionContractError('game_design must be an object')
    design_snapshot = _json_copy(bound_game_design(game_design), 'game_design')
    research_snapshot = None if research_brief is None else _json_copy(research_brief, 'research_brief')
    raw_modules = [_normalize_module(value) for value in modules]
    raw_assets = [_normalize_asset(value) for value in assets]
    normalized_modules: list[dict[str, Any]] = []
    seen_mids: set[str] = set()
    for module in raw_modules:
        module_id = module['module_id']
        if module_id in seen_mids:
            counter = 2
            while f'{module_id}_{counter}' in seen_mids:
                counter += 1
            module_id = f'{module_id}_{counter}'
            module = dict(module)
            module['module_id'] = module_id
        seen_mids.add(module_id)
        normalized_modules.append(module)
    normalized_assets: list[dict[str, Any]] = []
    seen_aids: set[str] = set()
    for asset in raw_assets:
        asset_id = asset['asset_id']
        if asset_id in seen_aids:
            counter = 2
            while f'{asset_id}_{counter}' in seen_aids:
                counter += 1
            asset_id = f'{asset_id}_{counter}'
            asset = dict(asset)
            asset['asset_id'] = asset_id
        seen_aids.add(asset_id)
        normalized_assets.append(asset)
    input_acceptance = _normalize_acceptance_tests(acceptance_tests)
    normalized_evidence_plan = _validated_evidence_plan(evidence_plan)
    requirements = (
        _compile_evidence_requirements(requested_prompt, normalized_evidence_plan)
        if normalized_evidence_plan is not None
        else _compile_requirements(requested_prompt, design_snapshot, research_snapshot)
    )
    implementation_catalog, implementation_search = _implementation_catalog(
        normalized_modules,
        normalized_assets,
        evidence_plan=normalized_evidence_plan,
    )
    if not implementation_catalog:
        raise ProductionContractError('production contract requires at least one implementation')
    active_dimensions, activation_reasons = _infer_dimensions(requested_prompt=requested_prompt, game_design=design_snapshot, research_brief=research_snapshot, modules=[item for item in normalized_modules if not (item['kind'] == 'integration' and isinstance(item.get('config'), Mapping) and item['config'].get('integration_type') == 'mmm_research_shard')], assets=normalized_assets)
    acceptance_catalog: list[dict[str, str]] = []
    used_acceptance_statements: set[str] = set()
    for index, statement in enumerate(input_acceptance):
        _validate_public_acceptance(statement)
        acceptance_catalog.append({'acceptance_ref': f'acceptance:input:{index:08d}', 'origin': 'input', 'visibility': 'public', 'statement': statement})
        used_acceptance_statements.add(statement)
    dimension_acceptance: dict[str, str] = {}
    for dimension_id in active_dimensions:
        statement = str(_DIMENSIONS[dimension_id]['acceptance'])
        statement = _unique_acceptance_statement(f'[{dimension_id}] {statement}', used_acceptance_statements)
        ref = f'acceptance:quality:{dimension_id}'
        acceptance_catalog.append({'acceptance_ref': ref, 'origin': 'quality', 'visibility': 'internal', 'statement': statement})
        used_acceptance_statements.add(statement)
        dimension_acceptance[dimension_id] = ref
    requirement_acceptance: dict[str, str] = {}
    for requirement in requirements:
        requirement_ref = requirement['requirement_ref']
        ref = 'acceptance:' + requirement_ref
        if requirement['source'] == 'research_brief':
            statement = f"[{requirement_ref}] Verify the scoped research item is current, compatible, licensed, and used only within its evidence scope: {requirement['statement']}"
        elif requirement['source'] == 'evidence_plan':
            if normalized_evidence_plan is None:
                raise ProductionContractError('evidence requirement has no validated evidence plan')
            request_catalog = normalized_evidence_plan.get('request_catalog')
            if not isinstance(request_catalog, Mapping):
                raise ProductionContractError('evidence plan request catalog is missing')
            approved_requirements = request_catalog.get('requirements')
            if not isinstance(approved_requirements, list):
                raise ProductionContractError('evidence plan request requirements are missing')
            matching_requirements = [
                item
                for item in approved_requirements
                if isinstance(item, Mapping)
                and item.get('requirement_id') == requirement_ref
            ]
            if len(matching_requirements) != 1:
                raise ProductionContractError(
                    f'evidence requirement needs exactly one approved request requirement: {requirement_ref}'
                )
            public_checks = _require_string_list(
                matching_requirements[0].get('acceptance'),
                f'evidence public acceptance: {requirement_ref}',
                nonempty=True,
            )
            for public_check in public_checks:
                _validate_public_acceptance(public_check)
            statement = '; '.join(public_checks)
        else:
            statement = f"[{requirement_ref}] Demonstrate observable behavior or output satisfying: {requirement['statement']}"
        statement = _unique_acceptance_statement(statement, used_acceptance_statements)
        _validate_public_acceptance(statement)
        acceptance_catalog.append({'acceptance_ref': ref, 'origin': 'requirement', 'visibility': 'public', 'statement': statement})
        used_acceptance_statements.add(statement)
        requirement_acceptance[requirement_ref] = ref
    quality_catalog: list[dict[str, Any]] = []
    evidence_routes: list[dict[str, Any]] = []
    for dimension_id in active_dimensions:
        definition = _DIMENSIONS[dimension_id]
        dimension_ref = f'quality:{dimension_id}'
        route_ref = f'evidence:{dimension_id}'
        quality_catalog.append({'dimension_ref': dimension_ref, 'dimension_id': dimension_id, 'title': definition['title'], 'activation': 'baseline' if dimension_id in _BASELINE_DIMENSIONS else 'request-derived', 'activation_reasons': activation_reasons[dimension_id], 'objective': definition['objective'], 'acceptance_ref': dimension_acceptance[dimension_id], 'evidence_route_ref': route_ref})
        evidence_routes.append({'route_ref': route_ref, 'dimension_ref': dimension_ref, 'accepted_statuses': ['PASS', 'FAIL'], 'requirements': ['current_proposal_hash', 'unique_receipt_id', 'timezone_observed_at', 'independent_verifier', 'objective_evidence_refs']})
    implementation_index = _build_token_index(implementation_search)
    input_test_search = {entry['acceptance_ref']: _tokens(entry['statement']) for entry in acceptance_catalog if entry['origin'] == 'input'}
    input_test_index = _build_token_index(input_test_search)
    active_set = set(active_dimensions)
    coverage_groups: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_ref = requirement['requirement_ref']
        direct_implementations = (
            _evidence_implementation_refs(
                normalized_evidence_plan,
                requirement,
                implementation_refs={item['implementation_ref'] for item in implementation_catalog},
            )
            if normalized_evidence_plan is not None
            else _bounded_matches(
                requirement_ref,
                requirement['statement'] + ' ' + requirement['source_ref'],
                implementation_search,
                implementation_index,
                _LEGACY_DIRECT_IMPLEMENTATION_REFS,
            )
        )
        matched_input_tests = _bounded_matches(requirement_ref, requirement['statement'], input_test_search, input_test_index, 2, fallback=False)
        relevant_dimensions = list(_BASELINE_DIMENSIONS)
        if requirement['source'] == 'requested_prompt':
            relevant_dimensions.extend(item for item in _CONDITIONAL_ORDER if item in active_set)
        else:
            source_text = requirement['source_ref'] + ' ' + requirement['statement']
            for dimension_id in _CONDITIONAL_ORDER:
                if dimension_id in active_set and _text_triggers_dimension(source_text, dimension_id):
                    relevant_dimensions.append(dimension_id)
        quality_refs = [f'quality:{item}' for item in relevant_dimensions]
        group_ref = 'coverage:' + requirement_ref
        requirement['coverage_group_ref'] = group_ref
        coverage_groups.append({'group_ref': group_ref, 'requirement_ref': requirement_ref, 'implementation_catalog_ref': 'catalog:implementations', 'implementation_refs': direct_implementations, 'acceptance_catalog_ref': 'catalog:acceptance', 'acceptance_refs': [requirement_acceptance[requirement_ref], *matched_input_tests], 'quality_dimension_refs': quality_refs, 'evidence_route_refs': ['evidence:' + value.removeprefix('quality:') for value in quality_refs]})
    acceptance_tuple = tuple(
        entry['statement']
        for entry in acceptance_catalog
        if entry['visibility'] == 'public'
    )
    source_bindings = {'game_design_sha256': _canonical_sha256(design_snapshot), 'research_brief_sha256': '' if research_snapshot is None else _canonical_sha256(research_snapshot), 'module_input_sha256': _canonical_sha256(normalized_modules), 'asset_input_sha256': _canonical_sha256(normalized_assets), 'evidence_plan_sha256': '' if normalized_evidence_plan is None else str(normalized_evidence_plan['plan_sha256'])}
    contract: dict[str, Any] = {'schema_version': CONTRACT_SCHEMA, 'requested_prompt': requested_prompt, 'source_bindings': source_bindings, 'requirement_catalog': requirements, 'implementation_catalog': implementation_catalog, 'acceptance_catalog': acceptance_catalog, 'quality_dimension_catalog': quality_catalog, 'evidence_route_catalog': evidence_routes, 'coverage_groups': coverage_groups, 'completion_policy': _json_copy(_COMPLETION_POLICY, 'completion_policy'), 'catalog_stats': {'requirements': len(requirements), 'implementations': len(implementation_catalog), 'acceptance_tests': len(acceptance_catalog), 'quality_dimensions': len(quality_catalog), 'coverage_groups': len(coverage_groups), 'max_direct_implementation_refs_per_group': max((len(item['implementation_refs']) for item in coverage_groups), default=0)}, 'contract_sha256': ''}
    contract['contract_sha256'] = _hash_without_field(contract, 'contract_sha256')
    validate_production_contract(
        contract,
        normalized_modules,
        acceptance_tuple,
        normalized_assets,
        normalized_evidence_plan,
    )
    return ProductionContractCompilation(contract=contract, acceptance_tests=acceptance_tuple)

def validate_production_contract(
    contract: Mapping[str, Any],
    modules: Iterable[Any],
    acceptance_tests: Iterable[str],
    assets: Iterable[Any] | None = None,
    evidence_plan: Mapping[str, Any] | None = None,
) -> None:
    """Strictly validate a contract and its external proposal bindings."""
    if not isinstance(contract, Mapping):
        raise ProductionContractError('production contract must be an object')
    expected_root = {'schema_version', 'requested_prompt', 'source_bindings', 'requirement_catalog', 'implementation_catalog', 'acceptance_catalog', 'quality_dimension_catalog', 'evidence_route_catalog', 'coverage_groups', 'completion_policy', 'catalog_stats', 'contract_sha256'}
    _require_exact_keys(contract, expected_root, 'production contract')
    _json_copy(dict(contract), 'production contract')
    if contract['schema_version'] != CONTRACT_SCHEMA:
        raise ProductionContractError('unsupported production contract schema')
    if not isinstance(contract['requested_prompt'], str) or not contract['requested_prompt'].strip():
        raise ProductionContractError('contract requested_prompt is empty')
    if contract['completion_policy'] != _COMPLETION_POLICY:
        raise ProductionContractError('completion policy is not code-owned')
    if not _SHA256.fullmatch(str(contract['contract_sha256'])):
        raise ProductionContractError('contract_sha256 is not a canonical SHA-256')
    if contract['contract_sha256'] != _hash_without_field(contract, 'contract_sha256'):
        raise ProductionContractError('contract_sha256 does not match the contract')
    source_bindings = contract['source_bindings']
    for key, value in source_bindings.items():
        if value != '' and not _SHA256.fullmatch(str(value)):
            raise ProductionContractError(f'invalid source binding: {key}')
    if not source_bindings['game_design_sha256']:
        raise ProductionContractError('game design source binding is required')
    normalized_evidence_plan = _validated_evidence_plan(evidence_plan)
    if normalized_evidence_plan is not None:
        if source_bindings.get('evidence_plan_sha256') != normalized_evidence_plan['plan_sha256']:
            raise ProductionContractError(
                'evidence plan source binding does not match the current proposal'
            )
    elif source_bindings.get('evidence_plan_sha256') not in {'', None}:
        if not _SHA256.fullmatch(str(source_bindings['evidence_plan_sha256'])):
            raise ProductionContractError('invalid evidence plan source binding')
    requirements = _require_list(contract['requirement_catalog'], 'requirements')
    implementations = _require_list(contract['implementation_catalog'], 'implementations')
    acceptances = _require_list(contract['acceptance_catalog'], 'acceptance')
    dimensions = _require_list(contract['quality_dimension_catalog'], 'quality dimensions')
    routes = _require_list(contract['evidence_route_catalog'], 'evidence routes')
    groups = _require_list(contract['coverage_groups'], 'coverage groups')
    if not requirements or not implementations or not acceptances:
        raise ProductionContractError('requirements, implementations, and acceptance catalogs must be non-empty')
    requirement_refs: set[str] = set()
    group_by_requirement: dict[str, str] = {}
    for item in requirements:
        _require_exact_keys(item, {'requirement_ref', 'source', 'source_ref', 'statement', 'coverage_group_ref'}, 'requirement')
        ref = _nonempty_string(item['requirement_ref'], 'requirement_ref')
        if ref in requirement_refs:
            raise ProductionContractError(f'duplicate requirement ref: {ref}')
        requirement_refs.add(ref)
        if item['source'] not in {'requested_prompt', 'game_design', 'research_brief', 'evidence_plan'}:
            raise ProductionContractError(f'invalid requirement source: {ref}')
        _nonempty_string(item['source_ref'], 'requirement source_ref')
        _nonempty_string(item['statement'], 'requirement statement')
        if normalized_evidence_plan is not None:
            prefix = 'evidence_plan:'
            source_ref = str(item['source_ref'])
            if item['source'] != 'evidence_plan' or not source_ref.startswith(prefix):
                raise ProductionContractError(
                    f'evidence-mode requirement has invalid source binding: {ref}'
                )
            if ref != source_ref[len(prefix):]:
                raise ProductionContractError(
                    f'evidence requirement ID was rewritten: {ref}'
                )
        else:
            expected_requirement_digest = hashlib.sha256(_canonical_json({'source': item['source'], 'source_ref': item['source_ref'], 'statement': item['statement']}).encode('utf-8')).hexdigest()
            if ref != f'requirement:{expected_requirement_digest}':
                raise ProductionContractError(f'requirement ref does not match content: {ref}')
        group_by_requirement[ref] = _nonempty_string(item['coverage_group_ref'], 'coverage_group_ref')
    root_requirements = [item for item in requirements if item['source'] == 'requested_prompt']
    if normalized_evidence_plan is not None:
        if root_requirements:
            raise ProductionContractError(
                'evidence mode must not create a second whole-request requirement'
            )
    else:
        if len(root_requirements) != 1:
            raise ProductionContractError('exactly one original request requirement is required')
        if root_requirements[0]['statement'] != contract['requested_prompt']:
            raise ProductionContractError('original request requirement was changed')
    if normalized_evidence_plan is not None:
        expected_requirements = _compile_evidence_requirements(
            str(contract['requested_prompt']), normalized_evidence_plan
        )
        actual_core = [
            {
                'requirement_ref': item['requirement_ref'],
                'source': item['source'],
                'source_ref': item['source_ref'],
                'statement': item['statement'],
            }
            for item in requirements
        ]
        expected_core = [
            {
                'requirement_ref': item['requirement_ref'],
                'source': item['source'],
                'source_ref': item['source_ref'],
                'statement': item['statement'],
            }
            for item in expected_requirements
        ]
        if actual_core != expected_core:
            raise ProductionContractError(
                'evidence requirement catalog does not match the validated plan'
            )
    implementation_refs: set[str] = set()
    catalog_module_ids: list[str] = []
    evidence_plan_entry: Mapping[str, Any] | None = None
    for item in implementations:
        _require_exact_keys(item, {'implementation_ref', 'source_kind', 'implementation_id', 'kind', 'content_sha256'}, 'implementation')
        ref = _nonempty_string(item['implementation_ref'], 'implementation_ref')
        if ref in implementation_refs:
            raise ProductionContractError(f'duplicate implementation ref: {ref}')
        implementation_refs.add(ref)
        _nonempty_string(item['implementation_id'], 'implementation_id')
        _nonempty_string(item['kind'], 'implementation kind')
        if not _SHA256.fullmatch(str(item['content_sha256'])):
            raise ProductionContractError(f'invalid implementation hash: {ref}')
        if item['source_kind'] == 'module':
            catalog_module_ids.append(item['implementation_id'])
        elif item['source_kind'] == 'evidence_plan':
            if evidence_plan_entry is not None:
                raise ProductionContractError('duplicate evidence plan implementation')
            evidence_plan_entry = item
    bound_evidence_sha = str(source_bindings.get('evidence_plan_sha256') or '')
    if bound_evidence_sha:
        if evidence_plan_entry is None:
            raise ProductionContractError('bound evidence plan implementation is missing')
        if (
            evidence_plan_entry['implementation_ref'] != 'implementation:evidence_plan'
            or evidence_plan_entry['implementation_id'] != bound_evidence_sha
            or evidence_plan_entry['content_sha256'] != bound_evidence_sha
        ):
            raise ProductionContractError('evidence plan implementation binding mismatch')
    elif evidence_plan_entry is not None:
        raise ProductionContractError('unbound evidence plan implementation')
    external_modules = list(modules)
    normalized_external_modules: list[dict[str, Any]] | None
    if all(isinstance(value, str) for value in external_modules):
        expected_module_ids = list(external_modules)
        normalized_external_modules = None
    elif any(isinstance(value, str) for value in external_modules):
        raise ProductionContractError(
            'external modules must be all IDs or all complete module objects'
        )
    else:
        normalized_external_modules = [
            _normalize_module(value) for value in external_modules
        ]
        expected_module_ids = [
            item['module_id'] for item in normalized_external_modules
        ]
    if any(not isinstance(value, str) or not value for value in expected_module_ids):
        raise ProductionContractError('modules must contain non-empty module IDs')
    _require_unique(expected_module_ids, 'external module ID')
    if set(catalog_module_ids) != set(expected_module_ids):
        raise ProductionContractError('module catalog does not match proposal module IDs')
    if normalized_external_modules is not None:
        if source_bindings['module_input_sha256'] != _canonical_sha256(
            normalized_external_modules
        ):
            raise ProductionContractError(
                'module source binding does not match the current proposal modules'
            )
        module_hashes = {
            item['implementation_id']: item['content_sha256']
            for item in implementations
            if item['source_kind'] == 'module'
        }
        for module in normalized_external_modules:
            if module_hashes.get(module['module_id']) != _canonical_sha256(module):
                raise ProductionContractError(
                    'module implementation hash does not match the current proposal: '
                    + module['module_id']
                )
    if assets is not None:
        normalized_external_assets = [_normalize_asset(value) for value in assets]
        asset_ids = [item['asset_id'] for item in normalized_external_assets]
        _require_unique(asset_ids, 'external asset ID')
        catalog_asset_ids = [
            item['implementation_id']
            for item in implementations
            if item['source_kind'] == 'asset'
        ]
        if set(catalog_asset_ids) != set(asset_ids):
            raise ProductionContractError(
                'asset catalog does not match proposal asset IDs'
            )
        if source_bindings['asset_input_sha256'] != _canonical_sha256(
            normalized_external_assets
        ):
            raise ProductionContractError(
                'asset source binding does not match the current proposal assets'
            )
        asset_hashes = {
            item['implementation_id']: item['content_sha256']
            for item in implementations
            if item['source_kind'] == 'asset'
        }
        for asset in normalized_external_assets:
            if asset_hashes.get(asset['asset_id']) != _canonical_sha256(asset):
                raise ProductionContractError(
                    'asset implementation hash does not match the current proposal: '
                    + asset['asset_id']
                )
    acceptance_refs: set[str] = set()
    catalog_acceptance: list[str] = []
    for item in acceptances:
        _require_exact_keys(item, {'acceptance_ref', 'origin', 'visibility', 'statement'}, 'acceptance entry')
        ref = _nonempty_string(item['acceptance_ref'], 'acceptance_ref')
        if ref in acceptance_refs:
            raise ProductionContractError(f'duplicate acceptance ref: {ref}')
        acceptance_refs.add(ref)
        if item['origin'] not in {'input', 'quality', 'requirement'}:
            raise ProductionContractError(f'invalid acceptance origin: {ref}')
        if item['visibility'] not in {'public', 'internal'}:
            raise ProductionContractError(f'invalid acceptance visibility: {ref}')
        if item['origin'] == 'quality' and item['visibility'] != 'internal':
            raise ProductionContractError(f'quality acceptance must remain internal: {ref}')
        if item['origin'] != 'quality' and item['visibility'] != 'public':
            raise ProductionContractError(f'non-quality acceptance must be public: {ref}')
        if item['origin'] == 'quality' and not ref.startswith('acceptance:quality:'):
            raise ProductionContractError(f'invalid quality acceptance ref: {ref}')
        if item['origin'] == 'requirement' and not ref.startswith('acceptance:'):
            raise ProductionContractError(f'invalid requirement acceptance ref: {ref}')
        statement = _nonempty_string(item['statement'], 'acceptance statement')
        if item['visibility'] == 'public':
            _validate_public_acceptance(statement)
            catalog_acceptance.append(statement)
    external_acceptance = list(acceptance_tests)
    if catalog_acceptance != external_acceptance:
        raise ProductionContractError('acceptance catalog does not match proposal acceptance tests')
    _require_unique(external_acceptance, 'acceptance test')
    dimension_refs: set[str] = set()
    dimension_ids: list[str] = []
    route_for_dimension: dict[str, str] = {}
    for item in dimensions:
        _require_exact_keys(item, {'dimension_ref', 'dimension_id', 'title', 'activation', 'activation_reasons', 'objective', 'acceptance_ref', 'evidence_route_ref'}, 'quality dimension')
        dimension_id = _nonempty_string(item['dimension_id'], 'dimension_id')
        dimension_ref = _nonempty_string(item['dimension_ref'], 'dimension_ref')
        if dimension_id not in _DIMENSIONS or dimension_ref != f'quality:{dimension_id}':
            raise ProductionContractError(f'unknown quality dimension: {dimension_id}')
        if dimension_ref in dimension_refs:
            raise ProductionContractError(f'duplicate quality dimension: {dimension_id}')
        dimension_refs.add(dimension_ref)
        dimension_ids.append(dimension_id)
        expected_activation = 'baseline' if dimension_id in _BASELINE_DIMENSIONS else 'request-derived'
        if item['activation'] != expected_activation:
            raise ProductionContractError(f'invalid activation: {dimension_id}')
        reasons = _require_string_list(item['activation_reasons'], 'activation reasons', nonempty=True)
        if len(reasons) > 8:
            raise ProductionContractError(f'activation reasons are unbounded: {dimension_id}')
        if item['title'] != _DIMENSIONS[dimension_id]['title']:
            raise ProductionContractError(f'quality title was modified: {dimension_id}')
        if item['objective'] != _DIMENSIONS[dimension_id]['objective']:
            raise ProductionContractError(f'quality objective was modified: {dimension_id}')
        if item['acceptance_ref'] not in acceptance_refs:
            raise ProductionContractError(f'missing quality acceptance: {dimension_id}')
        if item['acceptance_ref'] != f'acceptance:quality:{dimension_id}':
            raise ProductionContractError(f'quality acceptance binding mismatch: {dimension_id}')
        if item['evidence_route_ref'] != f'evidence:{dimension_id}':
            raise ProductionContractError(f'quality evidence binding mismatch: {dimension_id}')
        route_for_dimension[dimension_ref] = item['evidence_route_ref']
    if tuple(dimension_ids[:len(_BASELINE_DIMENSIONS)]) != _BASELINE_DIMENSIONS:
        raise ProductionContractError('baseline quality dimensions are missing or reordered')
    if len(dimension_ids) != len(set(dimension_ids)):
        raise ProductionContractError('quality dimensions must be unique')
    expected_dimension_order = list(_BASELINE_DIMENSIONS) + [value for value in _CONDITIONAL_ORDER if value in set(dimension_ids)]
    if dimension_ids != expected_dimension_order:
        raise ProductionContractError('quality dimensions are not in code-owned order')
    route_refs: set[str] = set()
    route_dimension_refs: set[str] = set()
    for item in routes:
        _require_exact_keys(item, {'route_ref', 'dimension_ref', 'accepted_statuses', 'requirements'}, 'evidence route')
        route_ref = _nonempty_string(item['route_ref'], 'route_ref')
        dimension_ref = _nonempty_string(item['dimension_ref'], 'dimension_ref')
        if route_ref in route_refs or dimension_ref in route_dimension_refs:
            raise ProductionContractError('duplicate evidence route')
        route_refs.add(route_ref)
        route_dimension_refs.add(dimension_ref)
        if dimension_ref not in dimension_refs:
            raise ProductionContractError(f'evidence route has no dimension: {route_ref}')
        if route_for_dimension[dimension_ref] != route_ref:
            raise ProductionContractError(f'evidence route binding mismatch: {route_ref}')
        if item['accepted_statuses'] != ['PASS', 'FAIL']:
            raise ProductionContractError(f'invalid accepted statuses: {route_ref}')
        if item['requirements'] != ['current_proposal_hash', 'unique_receipt_id', 'timezone_observed_at', 'independent_verifier', 'objective_evidence_refs']:
            raise ProductionContractError(f'evidence requirements were modified: {route_ref}')
    if route_dimension_refs != dimension_refs:
        raise ProductionContractError('each quality dimension needs one evidence route')
    group_refs: set[str] = set()
    covered_requirements: set[str] = set()
    for item in groups:
        _require_exact_keys(item, {'group_ref', 'requirement_ref', 'implementation_catalog_ref', 'implementation_refs', 'acceptance_catalog_ref', 'acceptance_refs', 'quality_dimension_refs', 'evidence_route_refs'}, 'coverage group')
        group_ref = _nonempty_string(item['group_ref'], 'group_ref')
        requirement_ref = _nonempty_string(item['requirement_ref'], 'requirement_ref')
        if group_ref in group_refs or requirement_ref in covered_requirements:
            raise ProductionContractError('duplicate coverage group or requirement')
        group_refs.add(group_ref)
        covered_requirements.add(requirement_ref)
        if requirement_ref not in requirement_refs:
            raise ProductionContractError(f'unknown covered requirement: {requirement_ref}')
        if group_by_requirement[requirement_ref] != group_ref:
            raise ProductionContractError(f'coverage group binding mismatch: {group_ref}')
        if group_ref != f'coverage:{requirement_ref}':
            raise ProductionContractError(f'coverage group ref is not code-owned: {group_ref}')
        if item['implementation_catalog_ref'] != 'catalog:implementations':
            raise ProductionContractError(f'invalid implementation catalog ref: {group_ref}')
        direct_refs = _require_string_list(item['implementation_refs'], 'implementation refs', nonempty=True)
        if len(direct_refs) != len(set(direct_refs)) or not set(direct_refs) <= implementation_refs:
            raise ProductionContractError(f'invalid implementation refs: {group_ref}')
        if normalized_evidence_plan is not None:
            requirement = next(
                value for value in requirements
                if value['requirement_ref'] == requirement_ref
            )
            expected_direct_refs = _evidence_implementation_refs(
                normalized_evidence_plan,
                requirement,
                implementation_refs=implementation_refs,
            )
            if direct_refs != expected_direct_refs:
                raise ProductionContractError(
                    f'evidence implementation binding mismatch: {group_ref}'
                )
        if item['acceptance_catalog_ref'] != 'catalog:acceptance':
            raise ProductionContractError(f'invalid acceptance catalog ref: {group_ref}')
        test_refs = _require_string_list(item['acceptance_refs'], 'acceptance refs', nonempty=True)
        if len(test_refs) > 3:
            raise ProductionContractError(f'too many direct acceptance refs: {group_ref}')
        if len(test_refs) != len(set(test_refs)) or not set(test_refs) <= acceptance_refs:
            raise ProductionContractError(f'invalid acceptance refs: {group_ref}')
        if f'acceptance:{requirement_ref}' not in test_refs:
            raise ProductionContractError(f'requirement acceptance is missing: {group_ref}')
        quality_refs = _require_string_list(item['quality_dimension_refs'], 'quality refs', nonempty=True)
        evidence_refs = _require_string_list(item['evidence_route_refs'], 'evidence refs', nonempty=True)
        if not set(quality_refs) <= dimension_refs or not set(evidence_refs) <= route_refs:
            raise ProductionContractError(f'invalid quality coverage refs: {group_ref}')
        if [route_for_dimension[value] for value in quality_refs] != evidence_refs:
            raise ProductionContractError(f'quality/evidence order mismatch: {group_ref}')
    if covered_requirements != requirement_refs:
        raise ProductionContractError('not every requirement has one coverage group')
    stats = contract['catalog_stats']
    _require_exact_keys(stats, {'requirements', 'implementations', 'acceptance_tests', 'quality_dimensions', 'coverage_groups', 'max_direct_implementation_refs_per_group'}, 'catalog stats')
    expected_stats = {'requirements': len(requirements), 'implementations': len(implementations), 'acceptance_tests': len(acceptances), 'quality_dimensions': len(dimensions), 'coverage_groups': len(groups), 'max_direct_implementation_refs_per_group': max((len(item['implementation_refs']) for item in groups), default=0)}
    if stats != expected_stats:
        raise ProductionContractError('catalog stats do not match contract catalogs')

def quality_contract_summary(contract: Mapping[str, Any]) -> str:
    module_ids = [item['implementation_id'] for item in contract.get('implementation_catalog', []) if isinstance(item, Mapping) and item.get('source_kind') == 'module']
    tests = [item['statement'] for item in contract.get('acceptance_catalog', []) if isinstance(item, Mapping) and isinstance(item.get('statement'), str)]
    validate_production_contract(contract, module_ids, tests)
    stats = contract['catalog_stats']
    dimensions = ', '.join(item['title'] for item in contract['quality_dimension_catalog'])
    return f"Tracks {stats['requirements']} request-derived requirements across {stats['implementations']} implementation entries and {stats['acceptance_tests']} observable checks. Required quality: {dimensions}. Completion requires fresh proposal-bound evidence from an independent verifier for every dimension."

def evaluate_quality_contract(contract: Mapping[str, Any], evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]], proposal_hash: str, previous: Mapping[str, Any] | None=None) -> dict[str, Any]:
    module_ids = [item['implementation_id'] for item in contract.get('implementation_catalog', []) if isinstance(item, Mapping) and item.get('source_kind') == 'module']
    acceptance = [item['statement'] for item in contract.get('acceptance_catalog', []) if isinstance(item, Mapping) and isinstance(item.get('statement'), str)]
    validate_production_contract(contract, module_ids, acceptance)
    if not isinstance(proposal_hash, str) or not _SHA256.fullmatch(proposal_hash):
        raise ProductionContractError('proposal_hash must be a canonical SHA-256')
    previous_by_dimension: dict[str, Mapping[str, Any]] = {}
    iteration = 1
    if previous is not None:
        _validate_quality_report(previous)
        if previous['proposal_hash'] != proposal_hash:
            raise ProductionContractError('previous report belongs to another proposal')
        if previous['contract_sha256'] != contract['contract_sha256']:
            raise ProductionContractError('previous report belongs to another contract')
        iteration = _strict_positive_int(previous['iteration'], 'previous iteration') + 1
        previous_by_dimension = {item['dimension_id']: item for item in previous['dimensions']}
    receipts = _normalize_evidence(evidence)
    active_ids = {item['dimension_id'] for item in contract['quality_dimension_catalog']}
    unknown = sorted(set(receipts) - active_ids)
    if unknown:
        raise ProductionContractError(f'evidence targets unknown dimensions: {unknown}')
    dimension_results: list[dict[str, Any]] = []
    plateau_dimensions: list[str] = []
    for dimension in contract['quality_dimension_catalog']:
        dimension_id = dimension['dimension_id']
        receipt_values = receipts.get(dimension_id, [])
        prior = previous_by_dimension.get(dimension_id)
        result = _evaluate_dimension_receipt(dimension_id=dimension_id, route_ref=dimension['evidence_route_ref'], receipts=receipt_values, proposal_hash=proposal_hash, previous=prior)
        if result['plateau']:
            plateau_dimensions.append(dimension_id)
        dimension_results.append(result)
    unresolved = [item['dimension_id'] for item in dimension_results if item['status'] != 'PASS']
    overall_status = 'PASS' if not unresolved else 'FAIL' if any(item['status'] == 'FAIL' for item in dimension_results) else 'MISSING'
    report: dict[str, Any] = {'schema_version': REPORT_SCHEMA, 'proposal_hash': proposal_hash, 'contract_sha256': contract['contract_sha256'], 'iteration': iteration, 'overall_status': overall_status, 'dimensions': dimension_results, 'unresolved_dimension_ids': unresolved, 'plateau': {'detected': bool(plateau_dimensions), 'dimension_ids': plateau_dimensions, 'identical_failure_threshold': _PLATEAU_THRESHOLD}, 'report_sha256': ''}
    report['report_sha256'] = _hash_without_field(report, 'report_sha256')
    _validate_quality_report(report)
    return report

def persist_quality_report(path: str | os.PathLike[str], report: Mapping[str, Any]) -> Path:
    _validate_quality_report(report)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(report)) + '\n'
    temporary_name = ''
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=target.parent, prefix=f'.{target.name}.', suffix='.tmp', delete=False) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
    return target

def quality_unresolved(report: Mapping[str, Any]) -> tuple[str, ...]:
    _validate_quality_report(report)
    return tuple(report['unresolved_dimension_ids'])

def _compile_requirements(requested_prompt: str, game_design: Any, research_brief: Any) -> list[dict[str, str]]:
    raw: list[tuple[str, str, str]] = [('requested_prompt', 'request:$', requested_prompt)]
    raw.extend(_source_items('game_design', game_design))
    if research_brief is not None:
        raw.extend(_source_items('research_brief', research_brief))
    seen: set[tuple[str, str, str]] = set()
    requirements: list[dict[str, str]] = []
    for source, source_ref, statement in raw:
        key = (source, source_ref, statement)
        if key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha256(_canonical_json({'source': source, 'source_ref': source_ref, 'statement': statement}).encode('utf-8')).hexdigest()
        requirements.append({'requirement_ref': f'requirement:{digest}', 'source': source, 'source_ref': source_ref, 'statement': statement, 'coverage_group_ref': ''})
    return requirements


def _compile_evidence_requirements(
    requested_prompt: str,
    evidence_plan: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Project the approved evidence graph without inventing new identities.

    ``requested_prompt`` is already bound by ``source_bindings`` and by every
    requirement's exact source receipt. Treating it as another requirement would
    add a synthetic eighth requirement to a seven-node request graph, so evidence
    mode preserves each original ``requirement_id`` verbatim and adds no root row.
    """
    if not requested_prompt.strip():
        raise ProductionContractError('requested prompt is empty')
    request_catalog = evidence_plan.get('request_catalog')
    if not isinstance(request_catalog, Mapping):
        raise ProductionContractError('evidence plan request catalog is missing')
    requirements = request_catalog.get('requirements')
    if not isinstance(requirements, list) or not requirements:
        raise ProductionContractError('evidence plan has no request requirements')
    compiled: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            raise ProductionContractError('evidence requirement must be an object')
        requirement_id = _nonempty_string(
            requirement.get('requirement_id'), 'evidence requirement ID'
        )
        if requirement_id in seen_ids:
            raise ProductionContractError(
                f'duplicate evidence requirement ID: {requirement_id}'
            )
        seen_ids.add(requirement_id)
        source_span = requirement.get('source_span')
        if not isinstance(source_span, Mapping):
            raise ProductionContractError(
                f'evidence requirement has no source span: {requirement_id}'
            )
        statement = _nonempty_string(
            source_span.get('text') or requirement.get('capability'),
            f'evidence requirement statement: {requirement_id}',
        )
        compiled.append(
            {
                'requirement_ref': requirement_id,
                'source': 'evidence_plan',
                'source_ref': f'evidence_plan:{requirement_id}',
                'statement': statement,
                'coverage_group_ref': '',
            }
        )
    return compiled


def _evidence_implementation_refs(
    evidence_plan: Mapping[str, Any],
    requirement: Mapping[str, Any],
    *,
    implementation_refs: set[str],
) -> list[str]:
    source_ref = str(requirement.get('source_ref') or '')
    prefix = 'evidence_plan:'
    if not source_ref.startswith(prefix):
        raise ProductionContractError(
            'evidence-mode requirement lacks an exact evidence-plan reference'
        )
    requirement_id = source_ref[len(prefix):]
    if str(requirement.get('requirement_ref') or '') != requirement_id:
        raise ProductionContractError(
            f'evidence requirement identity drifted: {requirement_id}'
        )
    bindings = evidence_plan.get('acceptance_release_bindings')
    if not isinstance(bindings, list):
        raise ProductionContractError(
            'evidence plan acceptance bindings are missing'
        )
    matches = [
        item
        for item in bindings
        if isinstance(item, Mapping)
        and item.get('requirement_ref') == requirement_id
    ]
    if len(matches) != 1:
        raise ProductionContractError(
            f'evidence requirement needs exactly one release binding: {requirement_id}'
        )
    binding = matches[0]
    component_refs = _require_string_list(
        binding.get('component_refs'),
        f'evidence component refs: {requirement_id}',
        nonempty=False,
    )
    task_refs = _require_string_list(
        binding.get('task_refs'),
        f'evidence task refs: {requirement_id}',
        nonempty=False,
    )
    result = [
        *(f'implementation:retained_component:{value}' for value in component_refs),
        *(f'implementation:module:{value}' for value in task_refs),
    ]
    result = list(dict.fromkeys(result))
    if not result or not set(result) <= implementation_refs:
        raise ProductionContractError(
            'evidence requirement does not resolve to current exact implementations'
        )
    return result

def _source_items(source: str, value: Any) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    stack: list[tuple[str, Any]] = [('$', value)]
    while stack:
        path, current = stack.pop()
        if isinstance(current, Mapping):
            children: list[tuple[str, Any]] = []
            for raw_key in sorted(current, key=lambda item: str(item)):
                if not isinstance(raw_key, str):
                    raise ProductionContractError(f'{source} object keys must be strings')
                if raw_key.startswith('_') or _skip_source_key(raw_key):
                    continue
                children.append((f'{path}.{raw_key}', current[raw_key]))
            stack.extend(reversed(children))
            continue
        if isinstance(current, (list, tuple)):
            stack.extend((f'{path}[{index}]', current[index]) for index in range(len(current) - 1, -1, -1))
            continue
        if current is None:
            continue
        if isinstance(current, bool):
            statement = f"{path}: {'true' if current else 'false'}"
        elif isinstance(current, (int, float)):
            statement = f'{path}: {_canonical_json(current)}'
        elif isinstance(current, str):
            stripped = current.strip()
            if not stripped:
                continue
            statement = stripped
        else:
            raise ProductionContractError(f'{source} contains unsupported value {type(current).__name__}')
        result.append((source, f'{source}:{path}', statement))
    return result

def _skip_source_key(key: str) -> bool:
    lowered = key.casefold()
    return lowered in _SKIP_SOURCE_KEYS or lowered.endswith('_sha256') or lowered.endswith('_hash')

def _normalize_module(value: Any) -> dict[str, Any]:
    data = _object_mapping(value, 'module')
    module_id = _nonempty_string(data.get('module_id'), 'module_id')
    kind = _nonempty_string(data.get('kind'), f'module kind for {module_id}')
    config = _json_copy(data.get('config', {}), f'module config for {module_id}')
    if not isinstance(config, dict):
        raise ProductionContractError(f'module config must be an object: {module_id}')
    depends_on = _string_sequence(data.get('depends_on', ()), 'module dependencies')
    gates = _string_sequence(data.get('required_gates', ()), 'module gates')
    return {'module_id': module_id, 'kind': kind, 'config': config, 'depends_on': depends_on, 'required_gates': gates}

def _normalize_asset(value: Any) -> dict[str, Any]:
    data = _object_mapping(value, 'asset')
    asset_id = _nonempty_string(data.get('asset_id'), 'asset_id')
    return {'asset_id': asset_id, 'kind': _nonempty_string(data.get('kind'), f'asset kind for {asset_id}'), 'prompt': str(data.get('prompt', '')), 'target_path': str(data.get('target_path', '')), 'width': data.get('width'), 'height': data.get('height')}

def _object_mapping(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    fields = getattr(value, '__dict__', None)
    if isinstance(fields, dict):
        return dict(fields)
    raise ProductionContractError(f'{label} must be an object or dataclass')

def _validated_evidence_plan(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ProductionContractError('evidence_plan must be an object')
    try:
        from .evidence_first_planning import validate_evidence_first_plan
        validate_evidence_first_plan(value)
    except (ImportError, ValueError, TypeError, RecursionError) as exc:
        raise ProductionContractError(f'invalid evidence plan: {exc}') from exc
    return _json_copy(dict(value), 'evidence_plan')

def _implementation_catalog(
    modules: Sequence[Mapping[str, Any]],
    assets: Sequence[Mapping[str, Any]],
    *,
    evidence_plan: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    catalog: list[dict[str, str]] = []
    search: dict[str, set[str]] = {}
    def add(ref: str, source_kind: str, identity: str, kind: str, value: Any, *, searchable: bool=True) -> None:
        catalog.append({'implementation_ref': ref, 'source_kind': source_kind, 'implementation_id': identity, 'kind': kind, 'content_sha256': _canonical_sha256(value)})
        if searchable:
            search[ref] = _tokens(f'{source_kind} {identity} {kind} ' + ' '.join(_scalar_text(value)))
    for item in modules:
        module_id = item['module_id']
        add(f'implementation:module:{module_id}', 'module', module_id, item['kind'], item, searchable=not (item['kind'] == 'integration' and isinstance(item.get('config'), Mapping) and item['config'].get('integration_type') == 'mmm_research_shard'))
    for item in assets:
        asset_id = item['asset_id']
        add(f'implementation:asset:{asset_id}', 'asset', asset_id, item['kind'], item)
    if evidence_plan is not None:
        plan_sha256 = str(evidence_plan['plan_sha256'])
        catalog.append({'implementation_ref': 'implementation:evidence_plan', 'source_kind': 'evidence_plan', 'implementation_id': plan_sha256, 'kind': 'semantic_dependency_manifest', 'content_sha256': plan_sha256})
        search['implementation:evidence_plan'] = _tokens(' '.join(str(item.get('capability') or '') for item in evidence_plan.get('request_catalog', {}).get('requirements', []) if isinstance(item, Mapping)))
        for component in evidence_plan.get('component_catalog', []):
            if not isinstance(component, Mapping):
                continue
            if component.get('verification_status') != 'verified' or component.get('bound_to_project') is not True:
                continue
            component_id = str(component.get('component_id') or '')
            content_sha256 = str(component.get('content_sha256') or '')
            if not component_id or not _SHA256.fullmatch(content_sha256):
                raise ProductionContractError('retained component has invalid evidence binding')
            ref = f'implementation:retained_component:{component_id}'
            catalog.append({'implementation_ref': ref, 'source_kind': 'retained_component', 'implementation_id': component_id, 'kind': str(component.get('kind') or 'project_component'), 'content_sha256': content_sha256})
            search[ref] = _tokens(' '.join([component_id, str(component.get('kind') or ''), *(str(item) for item in component.get('provides', ())) ]))
    return catalog, search

def _scalar_text(value: Any) -> Iterable[str]:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key in sorted(current, reverse=True):
                stack.append(current[key])
                stack.append(key)
        elif isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
        elif isinstance(current, (str, int, float, bool)):
            yield str(current)

def _infer_dimensions(*, requested_prompt: str, game_design: Any, research_brief: Any, modules: Sequence[Mapping[str, Any]], assets: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
    active = list(_BASELINE_DIMENSIONS)
    reasons: dict[str, list[str]] = {value: ['code-owned baseline'] for value in _BASELINE_DIMENSIONS}
    primary_text = ' '.join([requested_prompt, *_scalar_text(game_design), *_scalar_text(modules), *_scalar_text(assets)])
    text = ' '.join([primary_text, *([] if research_brief is None else _scalar_text(research_brief))])
    module_kinds = {str(item['kind']).casefold() for item in modules}
    asset_kinds = {str(item['kind']).casefold() for item in assets}
    for dimension_id in _CONDITIONAL_ORDER:
        definition = _DIMENSIONS[dimension_id]
        dimension_reasons: list[str] = []
        trigger_text = primary_text if dimension_id == 'performance' else text
        if _text_triggers_dimension(trigger_text, dimension_id):
            dimension_reasons.append('request/design/research text')
        matching_module_kinds = sorted(module_kinds & set(definition.get('module_kinds', ())))
        if matching_module_kinds:
            dimension_reasons.append('module kinds: ' + ', '.join(matching_module_kinds[:4]))
        matching_asset_kinds = sorted(asset_kinds & set(definition.get('asset_kinds', ())))
        if matching_asset_kinds:
            dimension_reasons.append('asset kinds: ' + ', '.join(matching_asset_kinds[:4]))
        if dimension_reasons:
            active.append(dimension_id)
            reasons[dimension_id] = dimension_reasons[:8]
    return active, reasons

def _text_triggers_dimension(text: str, dimension_id: str) -> bool:
    lowered = ' ' + text.casefold() + ' '
    definition = _DIMENSIONS[dimension_id]
    for phrase in definition.get('phrases', ()):
        if phrase.casefold() in lowered:
            return True
    token_set = _tokens(lowered)
    for term in definition.get('terms', ()):
        lowered_term = term.casefold().strip()
        if not lowered_term:
            continue
        if ' ' in lowered_term or '-' in lowered_term:
            if lowered_term in lowered:
                return True
        elif lowered_term in token_set:
            return True
    return False

def _build_token_index(search: Mapping[str, set[str]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for ref in search:
        for token in search[ref]:
            index.setdefault(token, []).append(ref)
    return index

def _bounded_matches(seed: str, text: str, search: Mapping[str, set[str]], index: Mapping[str, list[str]], limit: int, *, fallback: bool=True) -> list[str]:
    if not search or limit <= 0:
        return []
    query_terms = _tokens(text)
    ranked_terms = sorted((term for term in query_terms if term in index), key=lambda term: (len(index[term]), term))[:_MAX_MATCH_TERMS]
    candidate_refs: set[str] = set()
    seed_number = int(hashlib.sha256(seed.encode('utf-8')).hexdigest(), 16)
    for term in ranked_terms:
        postings = index[term]
        if len(postings) <= _MAX_POSTING_SCAN:
            candidate_refs.update(postings)
            continue
        offset = seed_number % len(postings)
        for index_offset in range(_MAX_POSTING_SCAN):
            candidate_refs.add(postings[(offset + index_offset) % len(postings)])
    scored = sorted(((len(query_terms & search[ref]), ref) for ref in candidate_refs if query_terms & search[ref]), key=lambda item: (-item[0], item[1]))
    if scored:
        return [ref for _, ref in scored[:limit]]
    if not fallback:
        return []
    refs = list(search)
    offset = seed_number % len(refs)
    return [refs[(offset + index) % len(refs)] for index in range(min(limit, len(refs)))]

def _normalize_evidence(evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    if isinstance(evidence, Mapping) and 'receipts' in evidence:
        if set(evidence) != {'receipts'}:
            raise ProductionContractError('evidence wrapper only accepts receipts')
        values = evidence['receipts']
        if not isinstance(values, (list, tuple)):
            raise ProductionContractError('evidence receipts must be a list')
        raw_receipts = list(values)
    elif isinstance(evidence, Mapping):
        raw_receipts = []
        for dimension_id, value in evidence.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                if not isinstance(item, Mapping):
                    raise ProductionContractError('each evidence receipt must be an object')
                copied = dict(item)
                declared = copied.get('dimension_id')
                if declared is not None and declared != dimension_id:
                    raise ProductionContractError('evidence dimension key does not match receipt')
                copied['dimension_id'] = dimension_id
                raw_receipts.append(copied)
    elif isinstance(evidence, (list, tuple)):
        raw_receipts = list(evidence)
    else:
        raise ProductionContractError('evidence must be an object or list of receipts')
    result: dict[str, list[Mapping[str, Any]]] = {}
    receipt_ids: set[str] = set()
    for value in raw_receipts:
        if not isinstance(value, Mapping):
            raise ProductionContractError('each evidence receipt must be an object')
        copied = _json_copy(dict(value), 'evidence receipt')
        dimension_id = _nonempty_string(copied.get('dimension_id'), 'evidence dimension_id')
        receipt_id = copied.get('receipt_id')
        if isinstance(receipt_id, str) and receipt_id:
            if receipt_id in receipt_ids:
                raise ProductionContractError(f'duplicate evidence receipt_id: {receipt_id}')
            receipt_ids.add(receipt_id)
        result.setdefault(dimension_id, []).append(copied)
    return result

def _evaluate_dimension_receipt(*, dimension_id: str, route_ref: str, receipts: Sequence[Mapping[str, Any]], proposal_hash: str, previous: Mapping[str, Any] | None) -> dict[str, Any]:
    base = {'dimension_id': dimension_id, 'route_ref': route_ref, 'status': 'MISSING', 'reason': 'no fresh receipt', 'receipt_id': '', 'receipt_sha256': '', 'failure_signature': '', 'failure_streak': 0, 'plateau': False}
    if not receipts:
        return base
    if len(receipts) != 1:
        base['reason'] = 'ambiguous receipts'
        return base
    receipt = receipts[0]
    receipt_id = receipt.get('receipt_id')
    receipt_sha = _canonical_sha256(receipt)
    base['receipt_id'] = receipt_id if isinstance(receipt_id, str) else ''
    base['receipt_sha256'] = receipt_sha
    problem = _receipt_problem(receipt, route_ref, proposal_hash)
    if problem:
        base['reason'] = problem
        return base
    if previous is not None and (receipt_id == previous.get('receipt_id') or receipt_sha == previous.get('receipt_sha256')):
        base['reason'] = 'stale receipt reused from the previous iteration'
        return base
    status = receipt['status']
    base['status'] = status
    base['reason'] = 'independently verified' if status == 'PASS' else 'verified failure'
    if status == 'FAIL':
        signature = receipt.get('failure_signature')
        if not isinstance(signature, str) or not signature.strip():
            signature = _canonical_sha256({'dimension_id': dimension_id, 'failure': receipt.get('failure', receipt.get('reason', 'failed')), 'evidence_refs': receipt['evidence_refs']})
        signature = signature.strip()
        streak = 1
        if previous is not None and previous.get('status') == 'FAIL' and previous.get('failure_signature') == signature:
            streak = _strict_nonnegative_int(previous.get('failure_streak', 0), 'previous failure streak') + 1
        base['failure_signature'] = signature
        base['failure_streak'] = streak
        base['plateau'] = streak >= _PLATEAU_THRESHOLD
    return base

def _receipt_problem(receipt: Mapping[str, Any], route_ref: str, proposal_hash: str) -> str:
    forbidden = {'complete', 'completion', 'overall_status', 'all_passed'}
    if forbidden & set(receipt):
        return 'self-certified completion fields are not accepted'
    if receipt.get('status') not in {'PASS', 'FAIL'}:
        return 'receipt status must be PASS or FAIL'
    if receipt.get('route_ref') != route_ref:
        return 'receipt uses the wrong evidence route'
    if receipt.get('proposal_hash') != proposal_hash:
        return 'receipt is not bound to the current proposal'
    receipt_id = receipt.get('receipt_id')
    if not isinstance(receipt_id, str) or not receipt_id.strip():
        return 'receipt_id is missing'
    observed_at = receipt.get('observed_at')
    if not isinstance(observed_at, str) or not _timezone_datetime(observed_at):
        return 'observed_at must be an ISO-8601 timestamp with timezone'
    producer = receipt.get('producer')
    verifier = receipt.get('verified_by')
    if not isinstance(producer, str) or not producer.strip():
        return 'receipt producer is missing'
    if not isinstance(verifier, str) or not verifier.strip():
        return 'independent verifier is missing'
    if producer.strip().casefold() == verifier.strip().casefold():
        return 'producer may not self-verify'
    refs = receipt.get('evidence_refs')
    if not isinstance(refs, (list, tuple)) or not refs or any(not isinstance(value, str) or not value.strip() for value in refs):
        return 'objective evidence_refs are missing'
    return ''

def _timezone_datetime(value: str) -> bool:
    candidate = value[:-1] + '+00:00' if value.endswith('Z') else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None

def _validate_quality_report(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise ProductionContractError('quality report must be an object')
    expected = {'schema_version', 'proposal_hash', 'contract_sha256', 'iteration', 'overall_status', 'dimensions', 'unresolved_dimension_ids', 'plateau', 'report_sha256'}
    _require_exact_keys(report, expected, 'quality report')
    _json_copy(dict(report), 'quality report')
    if report['schema_version'] != REPORT_SCHEMA:
        raise ProductionContractError('unsupported quality report schema')
    for field in ('proposal_hash', 'contract_sha256', 'report_sha256'):
        if not _SHA256.fullmatch(str(report[field])):
            raise ProductionContractError(f'invalid quality report {field}')
    if report['report_sha256'] != _hash_without_field(report, 'report_sha256'):
        raise ProductionContractError('quality report hash mismatch')
    _strict_positive_int(report['iteration'], 'quality report iteration')
    if report['overall_status'] not in {'PASS', 'MISSING', 'FAIL'}:
        raise ProductionContractError('invalid quality report overall_status')
    dimensions = _require_list(report['dimensions'], 'quality report dimensions')
    ids: list[str] = []
    unresolved: list[str] = []
    plateau_ids: list[str] = []
    for item in dimensions:
        _require_exact_keys(item, {'dimension_id', 'route_ref', 'status', 'reason', 'receipt_id', 'receipt_sha256', 'failure_signature', 'failure_streak', 'plateau'}, 'quality dimension result')
        dimension_id = _nonempty_string(item['dimension_id'], 'result dimension_id')
        if dimension_id not in _DIMENSIONS:
            raise ProductionContractError(f'unknown report dimension: {dimension_id}')
        ids.append(dimension_id)
        if item['status'] not in {'PASS', 'MISSING', 'FAIL'}:
            raise ProductionContractError(f'invalid dimension status: {dimension_id}')
        _nonempty_string(item['route_ref'], 'result route_ref')
        _nonempty_string(item['reason'], 'result reason')
        if not isinstance(item['receipt_id'], str):
            raise ProductionContractError(f'invalid receipt ID: {dimension_id}')
        if item['receipt_sha256'] and not _SHA256.fullmatch(item['receipt_sha256']):
            raise ProductionContractError(f'invalid receipt hash: {dimension_id}')
        streak = _strict_nonnegative_int(item['failure_streak'], 'failure_streak')
        if type(item['plateau']) is not bool:
            raise ProductionContractError('dimension plateau must be boolean')
        if item['status'] != 'FAIL' and (item['failure_signature'] or streak or item['plateau']):
            raise ProductionContractError('only failed evidence can carry failure state')
        if item['status'] in {'PASS', 'FAIL'} and (not item['receipt_id'] or not item['receipt_sha256']):
            raise ProductionContractError('passing or failing evidence needs a receipt')
        if item['status'] == 'FAIL' and (not isinstance(item['failure_signature'], str) or not item['failure_signature'] or streak < 1):
            raise ProductionContractError('failed evidence needs a failure signature')
        if item['plateau'] != (streak >= _PLATEAU_THRESHOLD):
            raise ProductionContractError('dimension plateau does not match failure streak')
        if item['status'] != 'PASS':
            unresolved.append(dimension_id)
        if item['plateau']:
            plateau_ids.append(dimension_id)
    if len(ids) != len(set(ids)):
        raise ProductionContractError('quality report dimensions must be unique')
    if report['unresolved_dimension_ids'] != unresolved:
        raise ProductionContractError('quality report unresolved list mismatch')
    expected_overall = 'PASS' if not unresolved else 'FAIL' if any(item['status'] == 'FAIL' for item in dimensions) else 'MISSING'
    if report['overall_status'] != expected_overall:
        raise ProductionContractError('quality report overall status mismatch')
    plateau = report['plateau']
    _require_exact_keys(plateau, {'detected', 'dimension_ids', 'identical_failure_threshold'}, 'quality plateau')
    if plateau != {'detected': bool(plateau_ids), 'dimension_ids': plateau_ids, 'identical_failure_threshold': _PLATEAU_THRESHOLD}:
        raise ProductionContractError('quality plateau summary mismatch')

def _normalize_acceptance_tests(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ProductionContractError('acceptance_tests must be a sequence')
    result = [_nonempty_string(value, 'acceptance test').strip() for value in values]
    _require_unique(result, 'acceptance test')
    return result

def _unique_acceptance_statement(statement: str, used: set[str]) -> str:
    if statement not in used:
        return statement
    suffix = hashlib.sha256(statement.encode('utf-8')).hexdigest()[:12]
    return f'{statement} [{suffix}]'

def _tokens(value: str) -> set[str]:
    return {item for item in _TOKEN.findall(value.casefold()) if item}

def _string_sequence(value: Any, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ProductionContractError(f'{label} must be a list')
    return [_nonempty_string(item, label) for item in value]

def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProductionContractError(f'{label} must contain finite JSON values') from exc

def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))

def _canonical_sha256(value: Any) -> str:
    return 'sha256:' + hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()

def _hash_without_field(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload[field] = ''
    return _canonical_sha256(payload)

def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ProductionContractError(f'{label} must be an object')
    actual = set(value)
    if actual != expected:
        raise ProductionContractError(f'invalid {label} fields; missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}')

def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProductionContractError(f'{label} must be a JSON list')
    return value

def _require_string_list(value: Any, label: str, *, nonempty: bool) -> list[str]:
    values = _require_list(value, label)
    if nonempty and not values:
        raise ProductionContractError(f'{label} must not be empty')
    if any(not isinstance(item, str) or not item for item in values):
        raise ProductionContractError(f'{label} must contain non-empty strings')
    return values

def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionContractError(f'{label} must be a non-empty string')
    return value

def _require_unique(values: Sequence[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ProductionContractError(f'duplicate {label}: {duplicates}')

def _strict_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ProductionContractError(f'{label} must be a positive integer')
    return value

def _strict_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ProductionContractError(f'{label} must be a non-negative integer')
    return value
__all__ = ['ProductionContractCompilation', 'ProductionContractError', 'compile_production_contract', 'evaluate_quality_contract', 'persist_quality_report', 'quality_contract_summary', 'quality_unresolved', 'validate_production_contract']
