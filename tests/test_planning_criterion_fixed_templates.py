from copy import deepcopy

import pytest

from minecraft_mod_ai import planning_criterion_fragments as fragments
from minecraft_mod_ai.planning_detail_template import CORE_WORKSHEET_SECTIONS
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.planning_detail_checkpoint import refresh_worksheet_checkpoint
from minecraft_mod_ai.planning_state_contract import _hash_without


def authored(section):
    spec = {concern: [{field: f'{section}.{concern}.{field}' for field in columns.split()}]
            for concern, columns in DETAIL_RECORDS[section].items()}
    spec['inapplicable_concerns'] = []
    return {'section': section, 'specification': spec, 'constraint_evidence_refs': []}


def test_each_call_receives_one_concern_and_results_are_not_rewritten(monkeypatch):
    calls = []
    def run(router, identifier, *, context, allowed_refs, progress, checkpoint):
        section, concern = identifier.split('/')[1:]
        calls.append(identifier)
        return {'records': authored(section)['specification'][concern], 'reason': '', 'evidence_refs': []}
    monkeypatch.setattr(fragments, 'run_record_template', run)
    result = fragments.generate_criterion_fragment(None, requirement={'statement': 'Activate'},
        criterion='Activation changes state', selected_sections=CORE_WORKSHEET_SECTIONS,
        evidence=[], allowed_refs=set())
    assert len(calls) == sum(len(DETAIL_RECORDS[s]) for s in CORE_WORKSHEET_SECTIONS)
    worksheet = fragments.assemble_worksheet_from_fragments({}, selected_sections=CORE_WORKSHEET_SECTIONS,
        criteria=('Activation changes state',), fragments={0: result}, allowed_refs=set())
    for section in CORE_WORKSHEET_SECTIONS:
        assert worksheet[section]['specification'] == authored(section)['specification']


def test_prose_cannot_be_promoted_to_state_or_test_records():
    value = {'section_updates': [{'section': 'state_model', 'implementation': 'Works correctly', 'constraint': '', 'evidence_refs': []}]}
    with pytest.raises(ValueError, match='prose'):
        fragments.validate_criterion_fragment(value, selected_sections=CORE_WORKSHEET_SECTIONS, allowed_refs=set())


def test_unknown_evidence_is_rejected_instead_of_removed():
    row = authored('state_model')
    row['constraint_evidence_refs'] = ['invented']
    with pytest.raises(ValueError, match='evidence'):
        fragments.validate_criterion_fragment({'section_updates': [row]}, selected_sections=CORE_WORKSHEET_SECTIONS, allowed_refs=set())


def test_missing_concern_is_not_automatically_marked_inapplicable():
    row = authored('state_model')
    row['specification']['transitions'] = []
    with pytest.raises(ValueError, match='inapplicable'):
        fragments.validate_criterion_fragment({'section_updates': [row]}, selected_sections=CORE_WORKSHEET_SECTIONS, allowed_refs=set())


def test_legacy_checkpoint_invalidates_derived_work_only():
    state = {'decisions': [{'decision_type': 'detailed_implementation_plan', 'engineering_worksheet': {}}],
             'detail_progress': [], 'coverage': ['old'], 'plan_ready': True,
             'evidence': [{'source': 'retained'}]}
    state['state_sha256'] = _hash_without(state, 'state_sha256')
    result = refresh_worksheet_checkpoint(state)
    assert result['evidence'] == state['evidence']
    assert result['decisions'] == [] and result['coverage'] == []
    assert result['plan_ready'] is False
    assert result['state_sha256'] == _hash_without(result, 'state_sha256')
    assert state['plan_ready'] is True
    tampered = deepcopy(state)
    tampered['evidence'] = []
    with pytest.raises(ValueError, match='hash mismatch'):
        refresh_worksheet_checkpoint(tampered)
