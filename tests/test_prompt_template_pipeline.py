import json
from copy import deepcopy

import pytest
from jsonschema import ValidationError

from minecraft_mod_ai.planning_state_contract import build_initial_planning_state, validate_planning_state
from minecraft_mod_ai.prompt_template_pipeline import extract_prompt_records
from minecraft_mod_ai.task_template_catalog import load_template
from minecraft_mod_ai.task_value_runner import run_value_template


class PromptRouter:
    def __init__(self, prompt, records=None):
        self.prompt = prompt
        self.records = records or {}
        self.calls = []
        self.interrupt_at = None

    def generate_tool_decision(self, role, messages, *, tool_name, parameters, **kwargs):
        context = json.loads(messages[1]['content'])
        self.calls.append((tool_name, deepcopy(context), deepcopy(parameters)))
        assert context['original_prompt'] == self.prompt
        assert set(context) <= {'original_prompt', 'accepted_records', 'allowed_evidence_refs'}
        if len(self.calls) == self.interrupt_at:
            raise TimeoutError('interrupted during prompt extraction')
        if tool_name == 'submit_prompt_intent':
            return {'statement': self.prompt, 'source_quote': self.prompt}
        if tool_name == 'submit_prompt_scope':
            return {'scope_status': 'explicit'}
        records = self.records.get(tool_name, [])
        index = len(context['accepted_records'])
        if index < len(records):
            return {'status': 'record', 'record': records[index], 'reason': '', 'evidence_refs': []}
        return {'status': 'done' if index else 'not_applicable', 'record': None,
                'reason': '' if index else 'The original prompt states no requirement of this kind.',
                'evidence_refs': []}


def test_live_prompt_boundary_executes_only_declared_small_tasks():
    prompt = 'Add a weather compass and deliver a ZIP with Korean labels.'
    router = PromptRouter(prompt, {
        'submit_prompt_parse': [{'statement': 'Add a weather compass'}],
        'submit_prompt_output_requirements': [{'statement': 'Deliver a ZIP with Korean labels'}],
    })
    state = build_initial_planning_state(router, prompt)
    validate_planning_state(state, prompt=prompt)
    assert {row['statement'] for row in state['known']} == {
        'Add a weather compass', 'Deliver a ZIP with Korean labels'}
    assert state['references'] == []
    assert state['unresolved'] == []
    names = [call[0] for call in router.calls]
    assert set(names) == {'submit_prompt_intent', 'submit_prompt_parse',
        'submit_prompt_entity_resolution', 'submit_prompt_constraints',
        'submit_prompt_output_requirements', 'submit_prompt_scope', 'submit_prompt_ambiguities'}
    assert 'submit_prompt_state' not in names
    for name, _, schema in router.calls:
        if name in {'submit_prompt_intent', 'submit_prompt_scope'}:
            identifier = 'prompt/' + name.removeprefix('submit_prompt_')
            assert schema == load_template(identifier)['output_schema']
        else:
            assert set(schema['properties']) == {'status', 'record', 'reason', 'evidence_refs'}


def test_named_reference_retains_host_owned_research_routing():
    prompt = 'Create a mod inspired by Example Game.'
    router = PromptRouter(prompt, {'submit_prompt_entity_resolution': [{
        'name': 'Example Game', 'source_quote': 'Example Game',
        'what_must_be_learned': 'Documented behavior requested by the user'}]})
    state = build_initial_planning_state(router, prompt)
    assert state['references'][0]['name'] == 'Example Game'
    assert state['unresolved'][0]['reason'] == 'reference_semantics'
    assert state['unresolved'][0]['resolution_route'] == 'reference_research'
    assert state['research_queue'][0]['queries'] == []


def test_prompt_tasks_resume_partial_and_completed_records_without_repeating_calls():
    prompt = 'Add a compass.'
    router = PromptRouter(prompt, {'submit_prompt_parse': [{'statement': prompt}]})
    router.interrupt_at = 3  # intent, first fact, interrupted fact completion
    progress = {}
    kwargs = dict(progress=progress, checkpoint=lambda key, value: progress.update({key: value}))
    with pytest.raises(TimeoutError):
        extract_prompt_records(router, prompt, **kwargs)
    assert len(progress) == 3  # exact capture, intent, first fact
    result = extract_prompt_records(router, prompt, **kwargs)
    assert result['known'] == [{'statement': prompt}]
    assert router.calls[2][:2] == router.calls[3][:2]
    count = len(router.calls)
    assert extract_prompt_records(router, prompt, **kwargs) == result
    assert len(router.calls) == count


def test_value_task_projects_only_declared_input_and_rejects_missing_input():
    prompt = 'Add a compass.'
    router = PromptRouter(prompt)
    assert run_value_template(router, 'prompt/intent', context={
        'original_prompt': prompt, 'unrelated_repository': 'must not enter this call'
    })['statement'] == prompt
    with pytest.raises(ValidationError):
        run_value_template(router, 'prompt/intent', context={})


@pytest.mark.parametrize('reason', ['scope', 'minecraft_api', 'implementation_method'])
def test_prompt_task_cannot_author_downstream_blocker_reasons(reason):
    prompt = 'Create a compass.'
    router = PromptRouter(prompt, {'submit_prompt_ambiguities': [{
        'question': 'Which implementation?', 'reason': reason, 'information_needed': 'An API'}]})
    from minecraft_mod_ai.structured_output import StructuredOutputValidationError
    with pytest.raises(StructuredOutputValidationError):
        build_initial_planning_state(router, prompt)


def test_capture_is_exact_and_never_invokes_a_model():
    prompt = '  First line.\nSecond line.  '
    assert run_value_template(None, 'prompt/capture', context={'original_prompt': prompt}) == {
        'original_prompt': prompt}


def test_pipeline_restores_prompt_checkpoint_before_research(monkeypatch):
    from minecraft_mod_ai import planning_state_pipeline as pipeline
    prompt = 'Add a compass.'
    router = PromptRouter(prompt, {'submit_prompt_parse': [{'statement': prompt}]})
    router.interrupt_at = 3
    snapshots = []
    with pytest.raises(TimeoutError):
        build_initial_planning_state(router, prompt, checkpoint=snapshots.append)
    restored = deepcopy(snapshots[-1])
    assert restored['checkpoint_kind'] == 'prompt_tasks'
    assert 'plan_ready' not in restored

    class ReachedResearch(Exception):
        pass

    def research(_router, _prompt, state, **kwargs):
        validate_planning_state(state, prompt=prompt)
        assert state['goal']['statement'] == prompt
        assert state['known'][0]['statement'] == prompt
        assert 'checkpoint_kind' not in state
        raise ReachedResearch

    monkeypatch.setattr(pipeline, 'collect_planning_state_research_convergent', research)
    with pytest.raises(ReachedResearch):
        pipeline.prepare_planning_state(router, prompt, existing_state=restored, checkpoint=snapshots.append)
    assert [c[0] for c in router.calls].count('submit_prompt_intent') == 1
    assert router.calls[2][:2] == router.calls[3][:2]


@pytest.mark.parametrize('change', ['prompt', 'tamper'])
def test_prompt_resume_rejects_changed_or_corrupted_checkpoint_before_model_call(change):
    prompt = 'Add a compass.'
    router = PromptRouter(prompt)
    snapshots = []
    build_initial_planning_state(router, prompt, checkpoint=snapshots.append)
    saved = deepcopy(snapshots[-1])
    count = len(router.calls)
    if change == 'prompt':
        prompt = 'Add something else.'
    else:
        saved['template_progress'] = {}
    with pytest.raises(ValueError, match='PROMPT_TASK_CHECKPOINT'):
        build_initial_planning_state(router, prompt, existing_checkpoint=saved)
    assert len(router.calls) == count
