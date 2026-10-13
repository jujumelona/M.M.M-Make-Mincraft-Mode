"""Regression for cefced5: import pages were sent as unconstrained native tools."""
import json
from contextlib import contextmanager
import pytest

from minecraft_mod_ai.first_pass_java_source import _one_structured_action
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai import llama_exact_context, llama_structured_decode_policy
from minecraft_mod_ai.model_adapters import llama_cpp_adapter

IMPORT_PAGE = {
    'type': 'object', 'properties': {'value': {'type': 'string', 'maxLength': 240}},
    'required': ['value'], 'additionalProperties': False,
}

class DataRouter(ModelRouter):
    def __init__(self, adapter): self.adapter = adapter
    def _generation_adapter(self, role): return self.adapter.config, self.adapter
    @contextmanager
    def _generation_scope(self, config): yield


def test_first_pass_import_uses_sampler_constrained_data_payload(monkeypatch):
    config = AdapterConfig(role='coder', adapter='llama_cpp', model_id='MiMo',
                           max_new_tokens=8192, max_context=32768,
                           extra={'runtime_contract': 'mimo', 'dynamic_output_budget': True})
    adapter = LlamaCppAdapter(config)
    monkeypatch.setattr(adapter, '_server_url', lambda request: 'http://local/v1')
    monkeypatch.setattr(llama_exact_context, 'capacity_safe_payload',
                        lambda url, payload, **kw: dict(payload))
    captured = []
    def completion(url, payload):
        captured.append(payload)
        # The deterministic server fixture fails if there is no actual sampler schema.
        assert not payload.get('tools')
        assert payload['json_schema'] == IMPORT_PAGE
        return {'content': json.dumps({'value': 'java.util.List'})}
    monkeypatch.setattr(llama_cpp_adapter, '_completion_message', completion)
    result = _one_structured_action(DataRouter(adapter), [], name='import_page', schema=IMPORT_PAGE)
    assert result == {'value': 'java.util.List'}
    assert len(captured) == 1
    assert captured[0]['chat_template_kwargs'] == {'enable_thinking': False}
    assert captured[0]['response_format']['type'] == 'json_object'


def test_strict_data_does_not_downgrade_schema_on_format_error(monkeypatch):
    from minecraft_mod_ai.llama_sse_protocol import LlamaNativeResponseFormatError
    from minecraft_mod_ai.model_adapters.base import GenerationRequest, ModelBackendError
    adapter = LlamaCppAdapter(AdapterConfig(role='coder', adapter='llama_cpp'))
    monkeypatch.setattr(adapter, '_server_url', lambda request: 'http://local/v1')
    calls = []
    def fail(adapter, url, request):
        calls.append(request)
        raise LlamaNativeResponseFormatError(500, 'common_chat_peg_parse failed')
    monkeypatch.setattr(llama_cpp_adapter, '_generate_one_turn', fail)
    request = GenerationRequest(messages=(), response_format='json',
                                response_schema=IMPORT_PAGE,
                                metadata={'mmm_strict_structured_data': True})
    with pytest.raises(ModelBackendError):
        adapter.generate_turn(request)
    assert len(calls) == 1


def test_data_page_does_not_extract_json_from_prose(monkeypatch):
    # Exercise the validation wrapper installed by production bootstrap as well.
    llama_structured_decode_policy._bind_structured_generation_retry(llama_cpp_adapter)
    config = AdapterConfig(role='coder', adapter='llama_cpp', max_new_tokens=8192,
                           extra={'runtime_contract': 'mimo', 'dynamic_output_budget': True})
    adapter = LlamaCppAdapter(config)
    monkeypatch.setattr(adapter, '_server_url', lambda request: 'http://local/v1')
    monkeypatch.setattr(llama_exact_context, 'capacity_safe_payload',
                        lambda url, payload, **kw: dict(payload))
    monkeypatch.setattr(llama_cpp_adapter, '_completion_message',
                        lambda url, payload: {'content': 'Here is the result: {"value":"java.util.List"}'})
    with pytest.raises(json.JSONDecodeError):
        _one_structured_action(DataRouter(adapter), [], name='import_page', schema=IMPORT_PAGE)
