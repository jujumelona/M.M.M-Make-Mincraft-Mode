from __future__ import annotations

import pytest

from minecraft_mod_ai.evidence_first_planning import (
    _hash_without,
    _sha,
    compile_evidence_first_plan,
)
from minecraft_mod_ai.production_contract import (
    ProductionContractError,
    compile_production_contract,
)


def _target() -> dict[str, object]:
    return {
        'target': {
            'minecraft_version': '26.1.2',
            'loader': 'fabric',
            'java_version': '25',
            'fabric_loader': '0.18.4',
            'fabric_api': '0.140.2+26.1',
            'fabric_loom': '1.14.10',
            'gradle': '9.2.1',
            'gradle_sha256': 'a' * 64,
            'data_pack_version': '101.1',
            'resource_pack_version': '84.0',
            'resource_pack_format': 84,
            'release_metadata_url': (
                'https://piston-meta.mojang.com/v1/packages/deadbeef/26.1.2.json'
            ),
            'source_api_family': 'fabric_live_ai',
        }
    }


def _request_catalog(prompt: str, public_acceptance: str) -> dict[str, object]:
    capability = 'researched.weather_compass'
    requirement = {
        'requirement_id': 'req_weather_compass',
        'capability': capability,
        'statement': prompt,
        'semantic_statement': prompt,
        'mandatory': True,
        'provenance_role': 'authored',
        'source_span': {
            'source_id': 'requested_prompt',
            'char_start': 0,
            'char_end': len(prompt),
            'text': prompt,
            'text_sha256': _sha(prompt),
        },
        'evidence_refs': [],
        'derived_from': [],
        'depends_on': [],
        'provides': [f'capability:{capability}'],
        'gameplay_capabilities': [capability],
        'implementation_capabilities': ['weather compass item behavior'],
        'implementation_obligations': [
            'Implement the weather compass item and expose only player-observable behavior.'
        ],
        'artifact_task_ids': [],
        'semantic_type': 'researched_gameplay_requirement',
        'unlock_policy': {
            'required_capabilities': [],
            'required_requirement_refs': [],
            'optional_capabilities': [],
            'optional_requirement_refs': [],
            'policy': 'grounded_planning_state_only',
        },
        'artifact_obligations': [],
        'design_resolution_obligations': [
            'Keep source-span trace metadata internal to planning.'
        ],
        'runtime_acceptance': [public_acceptance],
        'semantic_status': 'RESOLVED',
        'unresolved_spans': [],
        'acceptance': [public_acceptance],
        'observable_behavior': {
            'given': 'the weather compass exists',
            'when': 'the player uses the weather compass',
            'then': public_acceptance,
        },
        'template_profile': {
            'template_id': 'grounded_researched_requirement',
            'architecture_owner': 'planning_state',
        },
        'search_queries': [],
        'reuse_candidates': [],
        'detailed_plan_ref': 'detail_weather_compass',
        'engineering_worksheet': None,
    }
    catalog: dict[str, object] = {
        'prompt_sha256': _sha(prompt),
        'prompt_char_length': len(prompt),
        'purpose': prompt,
        'requirements': [requirement],
        'constraints': [],
        'non_goals': [],
        'deployment_expectations': [],
        'requirement_graph': {
            'node_ids': ['req_weather_compass'],
            'edges': [],
        },
        'dependency_provenance': [],
        'semantic_audit': {
            'status': 'APPROVED',
            'generation_policy': 'explicit_test_authority',
        },
        'planning_state_sha256': 'test-fixture',
        'catalog_sha256': '',
    }
    catalog['catalog_sha256'] = _hash_without(catalog, 'catalog_sha256')
    return catalog


def _task_modules(plan: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            'module_id': task['task_id'],
            'kind': 'custom_java',
            'config': {},
            'depends_on': [],
            'required_gates': [],
        }
        for task in plan['tasks']
    ]


def test_evidence_public_acceptance_does_not_republish_internal_source_span() -> None:
    prompt = 'Add a weather compass and keep task_internal trace metadata private.'
    public_acceptance = 'The weather compass reports the observed weather to the player.'
    design = {
        'title': 'Weather compass',
        'acceptance_tests': [public_acceptance],
        '_evidence_request_catalog': _request_catalog(prompt, public_acceptance),
    }
    plan = compile_evidence_first_plan(
        prompt,
        design,
        target_decision=_target(),
    )

    requirements = plan['request_catalog']['requirements']
    assert len(requirements) == 1
    assert 'task_internal' in requirements[0]['source_span']['text']
    assert plan['acceptance_release_bindings'][0]['acceptance'] == [public_acceptance]

    compiled = compile_production_contract(
        requested_prompt=prompt,
        game_design=design,
        modules=_task_modules(plan),
        acceptance_tests=[],
        evidence_plan=plan,
    )

    public_requirement_checks = [
        item['statement']
        for item in compiled.contract['acceptance_catalog']
        if item['origin'] == 'requirement' and item['visibility'] == 'public'
    ]
    assert len(public_requirement_checks) == 1
    assert public_acceptance in public_requirement_checks[0]
    assert 'task_internal' not in public_requirement_checks[0]
    assert compiled.contract['requirement_catalog'][0]['statement'] == prompt


def test_public_input_acceptance_still_rejects_internal_task_language() -> None:
    with pytest.raises(
        ProductionContractError,
        match='public acceptance contains internal task or integrity language',
    ):
        compile_production_contract(
            requested_prompt='Add a weather compass.',
            game_design={'title': 'Weather compass'},
            modules=[
                {
                    'module_id': 'weather_compass',
                    'kind': 'item',
                    'config': {},
                    'depends_on': [],
                    'required_gates': [],
                }
            ],
            acceptance_tests=['task_internal: all declared provides exist'],
        )
