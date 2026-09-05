from __future__ import annotations

import pytest

from minecraft_mod_ai.evidence_first_planning import compile_evidence_first_plan
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
