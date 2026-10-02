from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy

import pytest
from test_atomic_concern_response_contract import _executor
from test_small_model_implementation_decisions import NoPlanningModelRouter

from minecraft_mod_ai import implementation_ir as ir
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError


def test_exhausted_concern_keeps_owner_and_assembles_executable_members(tmp_path):
    calls = []
    pages = iter([
        'private static final int LIMIT = 100;\n// MMM_REGION_MORE',
        'public static boolean allowed(int count) { return count <= LIMIT; }\n// MMM_REGION_DONE',
    ])

    def coder(messages):
        payload = json.loads(messages[-1]['content'])
        calls.append(payload)
        if len(calls) == 1:
            raise ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED: completion_tokens=4096')
        return next(pages)

    executor = _executor([])
    executor.call_coder = coder
    result = executor.run()
    assert len(calls) == 3
    assert all(p['host_selected_class'] == 'Test' for p in calls)
    assert all(p['task_authority'] == calls[0]['task_authority'] for p in calls)
    assert calls[2]['region_page']['accepted_api'][0]['symbol'] == 'LIMIT'
    assert result['source'].count('LIMIT = 100') == 1
    assert 'Part1' not in result['source']
    assert 'MMM_REGION_DONE' not in result['source']
    javac, java = shutil.which('javac'), shutil.which('java')
    if not javac or not java:
        pytest.skip('JDK required for executable verification')
    source = tmp_path / 'Test.java'
    source.write_text(result['source'], encoding='utf-8')
    harness = tmp_path / 'Probe.java'
    harness.write_text(
        'package example; public class Probe { public static void main(String[] args) {'
        'if (!Test.allowed(100) || Test.allowed(101)) throw new AssertionError(); }}',
        encoding='utf-8',
    )
    subprocess.run([javac, '-d', str(tmp_path), str(source), str(harness)], check=True, capture_output=True)
    subprocess.run([java, '-cp', str(tmp_path), 'example.Probe'], check=True, capture_output=True)


@pytest.mark.parametrize('page,code', [
    ('private static int value = 1;', 'COMPLETION_REQUIRED'),
    ('private static void broken( {\n// MMM_REGION_DONE', 'SCOPE_ESCAPE'),
    ('// MMM_REGION_MORE', 'NO_PROGRESS'),
    ('private static int a; private static int b;\n// MMM_REGION_DONE', 'UNIT_CARDINALITY'),
])
def test_page_admission_rejects_incomplete_or_unbounded_work(page, code):
    calls = 0

    def coder(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED')
        return page

    executor = _executor([])
    executor.call_coder = coder
    with pytest.raises(CustomModuleGenerationError, match=code):
        executor.run()
    assert calls == 2
    assert executor.state == {}


def test_paging_cannot_redeclare_accepted_member_or_write_partial_source():
    responses = iter([
        ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED'),
        'private static int value = 1;\n// MMM_REGION_MORE',
        'private static int value = 2;\n// MMM_REGION_DONE',
    ])

    def coder(messages):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    executor = _executor([])
    executor.call_coder = coder
    executor.write_source = lambda *_: pytest.fail('partial source must not be written')
    with pytest.raises(CustomModuleGenerationError, match='OWNERSHIP_VIOLATION'):
        executor.run()


def test_canonical_graph_budget_does_not_split_state_owners_before_region_generation():
    router = NoPlanningModelRouter()
    router.implementation_output_budget = 100
    text = '## failure_and_limits\n- missing_dependencies: reject missing dependencies.\n'
    graph = ir.compile_authored_graph(router, text=text, package='example', mod_id='test', target={})
    assert [n['symbol'] for n in graph['nodes']] == ['AuthoredFailureLimits']
    assert graph['source_text'] == text
    assert router.calls == []


def test_refinement_schema_and_node_admission_agree_for_internal_canonical_owner():
    raw = {
        'symbol': 'AuthoredFailureLimits', 'kind': 'java', 'resource_path': '',
        'responsibility': 'Handle failures', 'requirements': ['R1'],
        'obligations': ['Reject missing dependencies'], 'public_api': [],
        'depends_on': [], 'activation': False, 'estimated_tokens': 1000,
    }
    ir.validate_node(raw, package='example', mod_id='test', refs={'R1'})
    schema = ir._page_schema({'rejected_node': raw, 'requirements': {'R1': 'Failure rule'}})
    assert ir._schema_diagnostics({'nodes': [raw]}, schema) == []
    unrelated = deepcopy(raw)
    unrelated['symbol'] = 'Unrelated'
    assert ir._schema_diagnostics({'nodes': [unrelated]}, schema)


def test_initialize_paging_preserves_statement_order():
    responses = iter([
        'private static int value;',
        ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED'),
        'value = 1;\n// MMM_REGION_MORE',
        'value += 2;\n// MMM_REGION_DONE',
    ])

    def coder(messages):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    executor = _executor([], section='integration', require_initialize=True)
    executor.call_coder = coder
    result = executor.run()
    assert result['source'].index('value = 1;') < result['source'].index('value += 2;')


def test_initialize_paging_allows_required_repeated_side_effects():
    responses = iter([
        'private static int value;',
        ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED'),
        'value++;\n// MMM_REGION_MORE',
        'value++;\n// MMM_REGION_DONE',
    ])

    def coder(messages):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    executor = _executor([], section='integration', require_initialize=True)
    executor.call_coder = coder
    assert executor.run()['source'].count('value++;') == 2


def test_completed_method_bodies_do_not_accumulate_in_following_prompts():
    calls = []
    large_body = 'int value = 0; ' + 'value++; ' * 1500 + 'return value;'

    def coder(messages):
        payload = json.loads(messages[-1]['content'])
        calls.append(payload)
        if len(calls) == 1:
            raise ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED')
        if len(calls) == 2:
            return f'private static int count() {{ {large_body} }}\n// MMM_REGION_MORE'
        return 'public static int result() { return count(); }\n// MMM_REGION_DONE'

    executor = _executor([])
    executor.call_coder = coder
    source = executor.run()['source']
    assert large_body in source
    assert large_body not in json.dumps(calls[-1])
    assert len(json.dumps(calls[-1])) - len(json.dumps(calls[-2])) < 1000
    assert calls[-1]['region_page']['accepted_api'][0]['symbol'] == 'count'


def test_paged_context_preserves_nested_constructor_and_component_contracts():
    nested = 'private static record Snapshot(int storedCount, boolean active) {}'
    calls = []

    def coder(messages):
        payload = json.loads(messages[-1]['content'])
        calls.append(payload)
        if len(calls) == 1:
            raise ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED')
        if len(calls) == 2:
            return nested + '\n// MMM_REGION_MORE'
        assert payload['region_page']['accepted_nested_types'] == [nested]
        return 'public static int count() { return new Snapshot(10, true).storedCount(); }\n// MMM_REGION_DONE'

    executor = _executor([])
    executor.call_coder = coder
    assert nested in executor.run()['source']


def test_page_limit_requires_completion_without_compiling_partial_work(monkeypatch):
    from minecraft_mod_ai import atomic_region_paging as paging

    monkeypatch.setattr(paging, 'MAX_REGION_PAGES', 2)
    calls = 0

    def coder(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED')
        return f'private static int value{calls};\n// MMM_REGION_MORE'

    executor = _executor([])
    executor.call_coder = coder
    executor.write_source = lambda *_: pytest.fail('no partial publication')
    with pytest.raises(CustomModuleGenerationError, match='PAGE_LIMIT'):
        executor.run()
    assert calls == 3


@pytest.mark.parametrize('reject_page', [False, True])
def test_production_graph_output_pressure_preserves_siblings_and_rolls_back(tmp_path, monkeypatch, reject_page):
    from types import SimpleNamespace

    from test_direct_custom_module_generator import _adapter

    from minecraft_mod_ai import custom_module_generator as direct
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    from minecraft_mod_ai.authored_production import _compile_new_authored_modules
    from minecraft_mod_ai.llama_finish_reason_contract import (
        OUTPUT_EXHAUSTED,
        LlamaCompletionBoundaryError,
    )

    text = (
        '## failure_and_limits\n'
        '- invalid_inputs: reject negative count.\n'
        '- missing_dependencies: reject missing dependencies; limit count to 100.\n'
    )
    modules, _ = _compile_new_authored_modules(
        AuthoredPlan('limits', text), mod_id='test', package_name='example',
        target={'minecraft_version': '1.21.1', 'loader': 'fabric', 'mappings': ''},
    )
    request = modules[0].config['implementation_graph_request']
    original_request = deepcopy(request)
    main = tmp_path / request['entrypoint_path']
    main.parent.mkdir(parents=True)
    main.write_text(
        f"package example;\npublic final class {request['entrypoint_symbol']} {{ public void onInitialize() {{}} }}",
        encoding='utf-8',
    )
    original_main = main.read_bytes()
    owner = main.parent / 'AuthoredFailureLimits.java'
    original_owner = b'package example;\npublic final class AuthoredFailureLimits {\n// MMM_AUTHORED_FEATURE_BODY\n}\n'
    owner.write_bytes(original_owner)
    calls = []
    compiles = []

    class Router:
        implementation_output_budget = 100

        def __init__(self):
            self.calls = []

        def generate_implementation_decision(self, name, payload, *, state=None, checkpoint=None):
            from minecraft_mod_ai.implementation_decisions import compile_contribution

            return compile_contribution(self, name, payload, state or {}, checkpoint or (lambda: None))

        def generate_tool_decision(self, role, messages, **kwargs):
            assert role == 'coder'
            assert kwargs['tool_name'] == 'report_java_region_completion'
            payload = json.loads(messages[-1]['content'])
            page = payload['current_page_source']
            if 'LIMIT = 100' in page:
                return {'done': False, 'next_work': 'implement allowed(int count)'}
            return {'done': True, 'next_work': ''}

        def generate_text(self, role, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            calls.append((payload, kwargs))
            assert role == 'coder'
            assert kwargs['enable_tools'] is False
            assert kwargs['force_non_thinking'] is True
            if payload['concern']['name'] == 'invalid_inputs':
                return 'public static boolean valid(int count) { return count >= 0; }'
            if 'region_page' not in payload:
                raise LlamaCompletionBoundaryError(
                    'limit', kind=OUTPUT_EXHAUSTED, completion_tokens=4096, max_tokens=4096,
                )
            assert any(row['symbol'] == 'valid' for row in payload['available_sibling_api'])
            if payload['region_page']['index'] == 0:
                return 'private static final int LIMIT = 100;'
            if reject_page:
                return 'private static final int LIMIT = 200;'
            return 'public static boolean allowed(int count) { return valid(count) && count <= LIMIT; }'

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, root):
            compiles.append(root)
            javac = shutil.which('javac')
            if not javac:
                pytest.skip('JDK required')
            subprocess.run([javac, '-d', str(tmp_path / 'classes'),
                            *map(str, (root / 'src/main/java').rglob('*.java'))], check=True, capture_output=True)
            return SimpleNamespace(status='PASS', error='', commands=())

    monkeypatch.setattr(direct, 'adapter_for_target', lambda *_: _adapter())
    monkeypatch.setattr(direct, 'GradleRunner', Runner)
    router = Router()
    generator = direct.CustomModuleGenerator(router)
    if reject_page:
        with pytest.raises(CustomModuleGenerationError, match='OWNERSHIP_VIOLATION'):
            generator.generate(tmp_path, module=modules[0], minecraft_version='1.21.1', loader='fabric')
        assert main.read_bytes() == original_main
        assert owner.read_bytes() == original_owner
        assert not compiles
    else:
        generator.generate(tmp_path, module=modules[0], minecraft_version='1.21.1', loader='fabric')
        assert len(compiles) == 1
        source = owner.read_text(encoding='utf-8')
        assert source.count('boolean valid(') == 1
        assert source.count('LIMIT = 100') == 1
        harness = tmp_path / 'Probe.java'
        harness.write_text(
            'package example; public class Probe { public static void main(String[] args) {'
            'if (AuthoredFailureLimits.allowed(-1) || !AuthoredFailureLimits.allowed(100) '
            '|| AuthoredFailureLimits.allowed(101)) throw new AssertionError(); }}', encoding='utf-8',
        )
        subprocess.run([shutil.which('javac'), '-cp', str(tmp_path / 'classes'), '-d',
                        str(tmp_path / 'classes'), str(harness)], check=True, capture_output=True)
        subprocess.run([shutil.which('java'), '-cp', str(tmp_path / 'classes'), 'example.Probe'],
                       check=True, capture_output=True)
    assert len(calls) == 4
    assert all(p['host_selected_class'] == 'AuthoredFailureLimits' for p, _ in calls)
    assert request == original_request
    assert not list(main.parent.glob('*Part*.java'))
    assert not router.calls


def test_real_adapter_sse_output_limit_enters_same_owner_java_paging(monkeypatch):
    """Controlled transport evidence; this does not run a live Qwen model."""
    import threading
    from contextlib import nullcontext
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from minecraft_mod_ai import llama_exact_context, llama_lora_runtime
    from minecraft_mod_ai import llama_stream_efficiency_contract as streaming
    from minecraft_mod_ai.custom_module_generator import _call_coder
    from minecraft_mod_ai.model_adapters.base import AdapterConfig
    from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
    from minecraft_mod_ai.model_router import ModelRouter

    requests = []
    responses = [
        ('private static int unfinished(', 'length'),
        ('private static final int LIMIT = 100;\n// MMM_REGION_MORE', 'stop'),
        ('public static boolean allowed(int count) { return count <= LIMIT; }\n// MMM_REGION_DONE', 'stop'),
    ]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            index = len(requests)
            requests.append(request)
            text, finish = responses[index]
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Connection', 'close')
            self.end_headers()
            for event in (
                {'choices': [{'delta': {'content': text}}]},
                {'choices': [{'delta': {}, 'finish_reason': finish}],
                 'usage': {'completion_tokens': 4096 if finish == 'length' else 40}},
            ):
                self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_port}/v1'
    config = AdapterConfig(role='coder', adapter='llama_cpp', model_id='controlled-text', max_new_tokens=4096)
    adapter = LlamaCppAdapter(config)
    monkeypatch.setattr(adapter, '_server_url', lambda _: endpoint)
    monkeypatch.setattr(llama_exact_context, 'capacity_safe_payload', lambda _url, payload, **_: payload)
    monkeypatch.setattr(llama_lora_runtime, 'apply_request_lora', lambda *_: None)
    monkeypatch.setattr(streaming, '_report_server_connection', lambda _: None)

    class Router(ModelRouter):
        def _generation_adapter(self, role):
            assert role == 'coder'
            return config, adapter

        def _generation_scope(self, _config):
            return nullcontext()

    try:
        router = Router()
        executor = _executor([])
        executor.call_coder = lambda messages: _call_coder(
            router, messages, output_token_ceiling=4096, force_non_thinking=True, tool_stage='atomic_java',
        )
        source = executor.run()['source']
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        client = streaming._CLIENTS.pop(endpoint, None)
        if client:
            client.close()
    assert len(requests) == 3
    assert all(not request.get('tools') for request in requests)
    assert 'unfinished' not in source
    assert source.count('LIMIT = 100') == 1
    assert 'boolean allowed(' in source
