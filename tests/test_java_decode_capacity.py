import json
from minecraft_mod_ai.first_pass_java_source import _one_structured_action, _MEMBER_SCHEMA, _PLAN_SCHEMA
from minecraft_mod_ai.model_output_atomicity_contract import structured_output_token_ceiling

class RecordingRouter:
    def __init__(self): self.calls = []
    def generate_structured_data(self, role, messages, **kwargs):
        self.calls.append(kwargs)
        props = kwargs['parameters']['properties']
        if 'count' in props: return {'count': 0}
        return {key: ([] if value.get('type') == 'array' else '' if value.get('type') == 'string' else {}) for key, value in props.items()}


def test_member_capacity_covers_escaped_unicode_not_character_token_ratio():
    router = RecordingRouter()
    schema = {**_MEMBER_SCHEMA, 'properties': {'member': {**_MEMBER_SCHEMA['properties']['member'], 'minLength': 0}}}
    _one_structured_action(router, [], name='member', schema=schema)
    call = router.calls[0]
    worst = json.dumps({'member': '\u0001' * 2600}, ensure_ascii=True)
    assert call['output_token_ceiling'] >= len(worst.encode()) + 32
    assert call['output_token_floor'] == call['output_token_ceiling']


def test_large_plan_is_paged_before_inference():
    router = RecordingRouter()
    schema = {**_PLAN_SCHEMA, 'properties': {**_PLAN_SCHEMA['properties'], 'units': {**_PLAN_SCHEMA['properties']['units'], 'minItems': 0}}}
    result = _one_structured_action(router, [], name='plan', schema=schema)
    assert set(result) == set(schema['properties'])
    assert len(router.calls) > 1
    for call in router.calls:
        assert structured_output_token_ceiling(call['parameters'], absolute_ceiling=16384) <= call['output_token_ceiling'] <= 16384


def test_final_wire_budget_cannot_silently_shrink_complete_action():
    from types import SimpleNamespace
    import pytest
    from minecraft_mod_ai.generation_output_budget import (
        assert_request_output_capacity, GenerationOutputBudgetError,
    )
    request = SimpleNamespace(metadata={'mmm_output_token_floor': 15899})
    with pytest.raises(GenerationOutputBudgetError, match='required=15899 available=2304'):
        assert_request_output_capacity(request, {'max_tokens': 2304})
    assert_request_output_capacity(request, {'max_tokens': 15899})
    assert_request_output_capacity(SimpleNamespace(metadata={}), {'max_tokens': 128})
