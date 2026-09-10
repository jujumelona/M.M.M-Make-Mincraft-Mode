from copy import deepcopy

import pytest

from minecraft_mod_ai import task_template_runner as runner
from minecraft_mod_ai.task_template_catalog import ROOT, load_template


def reply(status, record=None, reason='', refs=None):
    return {'status': status, 'record': record, 'reason': reason, 'evidence_refs': refs or []}


def drive(monkeypatch, replies):
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return deepcopy(next(replies))
    monkeypatch.setattr(runner, 'generate_fixed_template_value', generate)
    return calls


def test_record_roundtrip_uses_exact_catalog_contract(monkeypatch):
    record = {'trigger': 'right click', 'owner': 'server player'}
    calls = drive(monkeypatch, iter([reply('record', record), reply('done')]))
    result = runner.run_record_template(None, 'feature/behavior_contract/entry_conditions', context={'criterion': 'activate'}, allowed_refs=set())
    assert result['records'] == [record]
    assert len(calls) == 2
    assert calls[0]['response_schema']['properties']['record']['anyOf'][0] == load_template('feature/behavior_contract/entry_conditions')['record_schema']


@pytest.mark.parametrize('response', [reply('done'), reply('not_applicable'), reply('record', {'trigger': 'x', 'owner': 'y'}, refs=['invented'])])
def test_empty_or_unproven_completion_is_rejected(monkeypatch, response):
    drive(monkeypatch, iter([response]))
    with pytest.raises(ValueError):
        runner.run_record_template(None, 'feature/behavior_contract/entry_conditions', context={}, allowed_refs=set())


def test_repeated_record_stops_without_claiming_completion(monkeypatch):
    record = {'trigger': 'right click', 'owner': 'server player'}
    drive(monkeypatch, iter([reply('record', record), reply('record', record)]))
    with pytest.raises(runner.TemplateBlocked, match='NO_PROGRESS'):
        runner.run_record_template(None, 'feature/behavior_contract/entry_conditions', context={}, allowed_refs=set())


def test_missing_information_is_not_inapplicability(monkeypatch):
    drive(monkeypatch, iter([reply('blocked', reason='trigger not established')]))
    with pytest.raises(runner.TemplateBlocked, match='trigger not established'):
        runner.run_record_template(None, 'feature/behavior_contract/entry_conditions', context={}, allowed_refs=set())


def test_catalog_manifests_resolve_every_declared_task():
    for path in ROOT.rglob('*.yaml'):
        task = load_template(path.relative_to(ROOT).with_suffix('').as_posix())
        for identifier in task.get('steps', []):
            assert load_template(identifier)['id'] == identifier


def test_allowed_evidence_is_not_automatically_attached(monkeypatch):
    responses=iter([{"status":"record","record":{"trigger":"click","owner":"server"},"reason":""},
                    {"status":"done","record":None,"reason":""}])
    drive(monkeypatch,responses)
    result=runner.run_record_template(None,'feature/behavior_contract/entry_conditions',context={},allowed_refs={'unrelated_a','unrelated_b'})
    assert result['evidence_refs']==[]
