from copy import deepcopy

import pytest

from minecraft_mod_ai import task_template_runner as runner


IDENTIFIER = 'feature/behavior_contract/entry_conditions'


def response(status, record=None, refs=None):
    return {'status': status, 'record': record, 'reason': '', 'evidence_refs': refs or []}


def test_interruption_resumes_accepted_record_and_completed_concern_without_calls(monkeypatch):
    progress, calls = {}, []
    record = {'trigger': 'activate', 'owner': 'server'}

    def generate(*args, **kwargs):
        calls.append(args[2])
        if len(calls) == 1:
            return response('record', record, ['source'])
        assert progress  # The record was saved before the failed request.
        raise TimeoutError('transport interrupted')

    monkeypatch.setattr(runner, 'generate_fixed_template_value', generate)
    kwargs = dict(context={'criterion': 'A'}, allowed_refs={'source'}, progress=progress,
                  checkpoint=lambda key, value: progress.update({key: deepcopy(value)}))
    with pytest.raises(TimeoutError):
        runner.run_record_template(None, IDENTIFIER, **kwargs)

    def complete(*args, **kw):
        import json
        assert json.loads(args[2][1]['content'])['accepted_records'] == [record]
        return response('done', refs=['source'])

    monkeypatch.setattr(runner, 'generate_fixed_template_value', complete)
    result = runner.run_record_template(None, IDENTIFIER, **kwargs)
    assert result == {'records': [record], 'reason': '', 'evidence_refs': ['source']}
    monkeypatch.setattr(runner, 'generate_fixed_template_value', lambda *a, **kw: pytest.fail('completed concern regenerated'))
    assert runner.run_record_template(None, IDENTIFIER, **kwargs) == result


@pytest.mark.parametrize('changed', ['context', 'allowed_refs', 'template'])
def test_changed_inputs_or_contract_do_not_reuse_prior_records(monkeypatch, changed):
    progress = {}
    replies = iter([response('record', {'trigger': 'x', 'owner': 'y'}), response('done')])
    monkeypatch.setattr(runner, 'generate_fixed_template_value', lambda *a, **kw: next(replies))
    kwargs = dict(context={'criterion': 'A'}, allowed_refs=set(), progress=progress,
                  checkpoint=lambda key, value: progress.update({key: value}))
    runner.run_record_template(None, IDENTIFIER, **kwargs)
    if changed == 'context':
        kwargs['context'] = {'criterion': 'B'}
    elif changed == 'allowed_refs':
        kwargs['allowed_refs'] = {'new_source'}
    else:
        template = runner.load_template(IDENTIFIER)
        template['task'] += ' Clarified instruction.'
        monkeypatch.setattr(runner, 'load_template', lambda _: template)
    with pytest.raises(StopIteration):
        runner.run_record_template(None, IDENTIFIER, **kwargs)


def test_saved_records_are_revalidated_before_any_model_call(monkeypatch):
    progress = {}
    replies = iter([response('record', {'trigger': 'x', 'owner': 'y'}), response('done')])
    monkeypatch.setattr(runner, 'generate_fixed_template_value', lambda *a, **kw: next(replies))
    kwargs = dict(context={}, allowed_refs=set(), progress=progress,
                  checkpoint=lambda key, value: progress.update({key: value}))
    runner.run_record_template(None, IDENTIFIER, **kwargs)
    next(iter(progress.values()))[0]['evidence_refs'] = ['invented']
    with pytest.raises(ValueError, match='TEMPLATE_EVIDENCE'):
        runner.run_record_template(None, IDENTIFIER, **kwargs)


def test_record_count_is_not_a_completion_limit(monkeypatch):
    replies = iter([response('record', {'trigger': str(i), 'owner': 'server'})
                    for i in range(129)] + [response('done')])
    monkeypatch.setattr(runner, 'generate_fixed_template_value', lambda *a, **kw: next(replies))
    result = runner.run_record_template(None, IDENTIFIER, context={}, allowed_refs=set())
    assert len(result['records']) == 129


def test_invalid_record_is_never_checkpointed(monkeypatch):
    saved = []
    monkeypatch.setattr(runner, 'generate_fixed_template_value',
                            lambda *a, **kw: response('record', {'trigger': '   ', 'owner': 'server'}))
    with pytest.raises(ValueError, match='TEMPLATE_RECORD'):
        runner.run_record_template(None, IDENTIFIER, context={}, allowed_refs=set(),
                                   checkpoint=lambda *args: saved.append(args))
    assert not saved
