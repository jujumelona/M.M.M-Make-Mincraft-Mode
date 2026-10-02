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
    ('\n'.join(f'private static int v{i};' for i in range(65)) + '\n// MMM_REGION_DONE', 'UNIT_LIMIT'),
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
        target={'minecraft_version': '1.21.1', 'loader': 'fabric', 'mappings': '1.21.1+build.3'},
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
            if payload.get('completion_phase') == 'select_next_unit':
                return {'done': False, 'next_work': 'declare LIMIT = 100', 'target': 'private static final int LIMIT'}
            page = payload['current_page_source']
            if 'LIMIT = 100' in page:
                return {'done': False, 'next_work': 'implement allowed(int count)',
                        'target': 'public static boolean allowed(int count)'}
            return {'done': True, 'next_work': '', 'target': ''}

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
    assert len(calls) == (5 if reject_page else 4)
    assert all(p['host_selected_class'] == 'AuthoredFailureLimits' for p, _ in calls)
    assert request == original_request
    assert not list(main.parent.glob('*Part*.java'))
    assert not router.calls


@pytest.mark.parametrize('native_completion', [False, True])
def test_real_adapter_sse_output_limit_enters_same_owner_java_paging(monkeypatch, native_completion):
    """Controlled transport evidence; this does not run a live Qwen model."""
    import threading
    from contextlib import nullcontext
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from minecraft_mod_ai import llama_exact_context, llama_lora_runtime
    from minecraft_mod_ai import llama_stream_efficiency_contract as streaming
    from minecraft_mod_ai.atomic_region_paging import decide_region_completion
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
    if native_completion:
        responses = [
            ('private static int unfinished(', 'length'),
            ({'done': False, 'next_work': 'declare limit constants', 'target': 'private static final int LIMIT'}, 'tool_calls'),
            ('private static final int LIMIT = 100; private static final int FLOOR = 0;', 'stop'),
            ({'done': False, 'next_work': 'implement allowed(int count)',
              'target': 'public static boolean allowed(int count)'}, 'tool_calls'),
            ('public static boolean allowed(int count) { return count >= FLOOR && count <= LIMIT; }', 'stop'),
            ({'done': True, 'next_work': '', 'target': ''}, 'tool_calls'),
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
            if isinstance(text, dict):
                tool_name = request['tools'][0]['function']['name']
                delta = {'tool_calls': [{'index': 0, 'id': f'call_{index}', 'type': 'function',
                                        'function': {'name': tool_name, 'arguments': json.dumps(text)}}]}
            else:
                delta = {'content': text}
            for event in (
                {'choices': [{'delta': delta}]},
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
        if native_completion:
            executor.completion_decider = lambda payload: decide_region_completion(router, payload)
        source = executor.run()['source']
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        client = streaming._CLIENTS.pop(endpoint, None)
        if client:
            client.close()
    assert len(requests) == (6 if native_completion else 3)
    if native_completion:
        assert [bool(request.get('tools')) for request in requests] == [False, True, False, True, False, True]
        assert 'FLOOR = 0' in source
    else:
        assert all(not request.get('tools') for request in requests)
    assert 'unfinished' not in source
    assert source.count('LIMIT = 100') == 1
    assert 'boolean allowed(' in source


# Exact six-declaration response retained from the supplied Colab failure.
DECISION_CONSTANTS = '''private static final String DECISION_SHIP_POSITION_AUTHORITATIVE_SIDE = "Server";
private static final String DECISION_ECONOMY_BALANCE_AUTHORITATIVE_SIDE = "Server";
private static final String DECISION_SHIP_POSITION_TRIGGER = "ShipPosition";
private static final String DECISION_ECONOMY_BALANCE_TRIGGER = "EconomyBalance";
private static final String DECISION_SHIP_POSITION_FROM_STATE = "Idle";
private static final String DECISION_ECONOMY_BALANCE_FROM_STATE = "Idle";'''


def test_colab_six_declarations_are_preserved_before_executable_completion(tmp_path):
    calls, decisions = [], []

    def coder(messages):
        payload = json.loads(messages[-1]['content'])
        calls.append(payload)
        if 'region_page' not in payload:
            raise ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED')
        assert payload['region_page']['next_work']
        if payload['region_page']['index'] == 0:
            return '```java\n' + DECISION_CONSTANTS + '\n```'
        assert len(payload['region_page']['accepted_api']) == 6
        return '''public static String authority(String trigger, String state) {
            if (DECISION_SHIP_POSITION_TRIGGER.equals(trigger)
                    && DECISION_SHIP_POSITION_FROM_STATE.equals(state))
                return DECISION_SHIP_POSITION_AUTHORITATIVE_SIDE;
            if (DECISION_ECONOMY_BALANCE_TRIGGER.equals(trigger)
                    && DECISION_ECONOMY_BALANCE_FROM_STATE.equals(state))
                return DECISION_ECONOMY_BALANCE_AUTHORITATIVE_SIDE;
            return "Rejected";
        }'''

    def decide(payload):
        decisions.append(deepcopy(payload))
        if payload['completion_phase'] == 'select_next_unit':
            return {'done': False, 'next_work': 'declare backing decision constants'}
        if payload['page_index'] == 0:
            return {'done': False, 'next_work': 'implement authority(String trigger, String state)'}
        return {'done': True, 'next_work': ''}

    executor = _executor([])
    executor.call_coder = coder
    executor.completion_decider = decide
    source = executor.run()['source']
    for declaration in DECISION_CONSTANTS.splitlines():
        assert source.count(declaration) == 1
    assert len(calls) == 3
    assert [p['completion_phase'] for p in decisions] == [
        'select_next_unit', 'assess_completion', 'assess_completion',
    ]
    assert all(p['task_authority'] == calls[0]['task_authority'] for p in calls + decisions)
    javac, java = shutil.which('javac'), shutil.which('java')
    if not javac or not java:
        pytest.skip('JDK required for executable verification')
    (tmp_path / 'Test.java').write_text(source, encoding='utf-8')
    (tmp_path / 'Probe.java').write_text('''package example;
        public class Probe { public static void main(String[] args) {
            if (!Test.authority("ShipPosition", "Idle").equals("Server")
                || !Test.authority("EconomyBalance", "Idle").equals("Server")
                || !Test.authority("Unknown", "Idle").equals("Rejected")
                || !Test.authority("ShipPosition", "Running").equals("Rejected"))
                throw new AssertionError();
        }}''', encoding='utf-8')
    subprocess.run([javac, '-d', str(tmp_path), str(tmp_path / 'Test.java'),
                    str(tmp_path / 'Probe.java')], check=True, capture_output=True)
    subprocess.run([java, '-cp', str(tmp_path), 'example.Probe'], check=True, capture_output=True)


def test_exhausted_unit_refines_only_pending_work_and_retains_accepted_members():
    calls, decisions = [], []
    responses = iter([
        ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED'),
        'private static int value = 7;',
        ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED'),
        'private static int read() { return value; }',
        'public static int result() { return read(); }',
    ])

    def coder(messages):
        calls.append(json.loads(messages[-1]['content']))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    def decide(payload):
        decisions.append(deepcopy(payload))
        phase = payload['completion_phase']
        if phase == 'select_next_unit':
            return {'done': False, 'next_work': 'declare value'}
        if phase == 'refine_next_unit':
            return {'done': False, 'next_work': 'implement private read() helper'}
        if 'int result()' in payload['current_page_source']:
            return {'done': True, 'next_work': ''}
        return {'done': False, 'next_work': 'implement result()'}

    executor = _executor([])
    executor.call_coder, executor.completion_decider = coder, decide
    source = executor.run()['source']
    assert source.count('int value = 7') == 1
    assert 'int read()' in source and 'int result()' in source
    assert calls[2]['region_page']['accepted_sha256'] == calls[3]['region_page']['accepted_sha256']
    assert calls[3]['region_page']['next_work'] == 'implement private read() helper'
    assert calls[3]['region_page']['exhausted_work'] == ['implement result()']
    assert all(p['host_selected_class'] == 'Test' for p in calls + decisions)


@pytest.mark.parametrize('decision', [
    {'done': True, 'next_work': ''},
    {'done': False, 'next_work': ''},
    {'done': 'false', 'next_work': 'declare value'},
    {'done': False, 'next_work': 'x' * 257},
])
def test_unit_selection_cannot_claim_completion_or_bypass_native_contract(decision):
    calls = []

    def coder(messages):
        calls.append(messages)
        raise ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED')

    executor = _executor([])
    executor.call_coder = coder
    executor.completion_decider = lambda _: decision
    executor.write_source = lambda *_: pytest.fail('invalid selection must not mutate source')
    with pytest.raises(CustomModuleGenerationError, match='COMPLETION_DECISION_INVALID'):
        executor.run()
    assert len(calls) == 1


@pytest.mark.parametrize('repeat', [True, False])
def test_unit_refinement_is_bounded_and_never_retries_identical_work(repeat):
    calls, decisions = [], []

    def coder(messages):
        calls.append(json.loads(messages[-1]['content']))
        raise ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED')

    def decide(payload):
        decisions.append(payload)
        return {'done': False, 'next_work': 'same work' if repeat else f'helper {len(decisions)}'}

    executor = _executor([])
    executor.call_coder, executor.completion_decider = coder, decide
    executor.write_source = lambda *_: pytest.fail('exhausted source must not be written')
    with pytest.raises(CustomModuleGenerationError, match='NO_PROGRESS' if repeat else 'UNIT_TOO_LARGE'):
        executor.run()
    assert len(calls) == (2 if repeat else 4)
    assert executor.state == {}


@pytest.mark.parametrize('bad_tail', [
    'private static int stable = 99;',
    'package escaped; public class Other {}',
    'private static void broken( {',
])
def test_multi_unit_page_is_transactional_and_keeps_ownership_guards(bad_tail):
    responses = iter([
        ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED'),
        'private static int stable = 7;\n// MMM_REGION_MORE',
        'private static int newValue = 3;\n' + bad_tail + '\n// MMM_REGION_DONE',
    ])

    def coder(_):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    executor = _executor([])
    executor.call_coder = coder
    executor.write_source = lambda *_: pytest.fail('rejected batch must not be written')
    with pytest.raises(CustomModuleGenerationError, match='OWNERSHIP_VIOLATION|SCOPE_ESCAPE'):
        executor.run()
    assert executor.state == {}


def test_initialize_batch_preserves_local_scope_order_and_repeated_side_effects(tmp_path):
    responses = iter([
        'public static int value;',
        ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED'),
        'int initial = 4; value = initial; value++;\n// MMM_REGION_MORE',
        'value++; value += initial;\n// MMM_REGION_DONE',
    ])
    calls = []

    def coder(messages):
        calls.append(json.loads(messages[-1]['content']))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    executor = _executor([], section='integration', require_initialize=True)
    executor.call_coder = coder
    source = executor.run()['source']
    assert 'int initial = 4;' in calls[-1]['region_page']['accepted_source']
    assert source.count('value++;') == 2
    javac, java = shutil.which('javac'), shutil.which('java')
    if not javac or not java:
        pytest.skip('JDK required')
    (tmp_path / 'Test.java').write_text(source, encoding='utf-8')
    (tmp_path / 'Probe.java').write_text(
        'package example; public class Probe { public static void main(String[] a) {'
        'Test.initialize(); if (Test.value != 10) throw new AssertionError(Test.value); }}', encoding='utf-8',
    )
    subprocess.run([javac, '-d', str(tmp_path), str(tmp_path / 'Test.java'),
                    str(tmp_path / 'Probe.java')], check=True, capture_output=True)
    subprocess.run([java, '-cp', str(tmp_path), 'example.Probe'], check=True, capture_output=True)


def test_continuation_preserves_accepted_body_and_admits_only_new_declarations():
    original = 'private static Object restoreShipComponents(java.util.Map<String, Object> nbtMap) { return nbtMap.get("ship_components"); }'
    echo = '''// copied earlier helper
        private static Object restoreShipComponents(java.util.Map<String, Object> nbtMap) {
            /* layout is immaterial */ return nbtMap.get("ship_components");
        }'''
    new_member = 'public static Object restore(java.util.Map<String, Object> data) { return restoreShipComponents(data); }'
    responses = iter([ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED'), original, echo + new_member])
    calls, decisions = [], []

    def coder(messages):
        calls.append(json.loads(messages[-1]['content']))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    def decide(payload):
        decisions.append(deepcopy(payload))
        if payload['completion_phase'] == 'select_next_unit':
            return {'done': False, 'next_work': 'implement restoreShipComponents'}
        if payload['page_index'] == 0:
            return {'done': False, 'next_work': 'implement restore using restoreShipComponents'}
        return {'done': True, 'next_work': ''}

    executor = _executor([])
    executor.call_coder, executor.completion_decider = coder, decide
    source = executor.run()['source']
    assert source.count(original) == 1
    assert source.count('Object restoreShipComponents(') == 1
    assert new_member in source
    assert decisions[-1]['current_page_source'] == new_member
    assert decisions[-1]['echoed_member_keys'] == ['method:restoreShipComponents(java.util.Map)']
    page = calls[-1]
    assert page['phase'] == 'append_atomic_concern_units'
    assert page['scope']['generation_mode'] == 'append_only'
    assert page['region_page']['accepted_member_keys'] == ['method:restoreShipComponents(java.util.Map)']
    assert 'regenerate the whole' not in json.dumps(page['generation_recipe'])


@pytest.mark.parametrize('omit_pending', [False, True])
def test_changed_redeclaration_corrects_only_pending_page_without_replacing_accepted_source(omit_pending):
    original = 'private static int stored() { return 7; }'
    pending = 'public static int result() { return stored(); }'
    corrected = 'public static int result() { return addOne(); }'
    helper = 'private static int addOne() { return stored() + 1; }'
    responses = iter([
        ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED'),
        original,
        'private static int stored() { return 99; }' + pending,
        helper if omit_pending else corrected + helper,
    ])
    calls, decisions = [], []

    def coder(messages):
        calls.append(json.loads(messages[-1]['content']))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    def decide(payload):
        decisions.append(deepcopy(payload))
        if payload['completion_phase'] == 'select_next_unit':
            return {'done': False, 'next_work': 'implement stored()'}
        if payload['page_index'] == 0:
            return {'done': False, 'next_work': 'implement result()'}
        return {'done': True, 'next_work': ''}

    executor = _executor([])
    executor.call_coder, executor.completion_decider = coder, decide
    if omit_pending:
        executor.write_source = lambda *_: pytest.fail('omitted pending declaration must not commit')
        with pytest.raises(CustomModuleGenerationError, match='selected declaration missing'):
            executor.run()
    else:
        source = executor.run()['source']
        assert source.count(original) == 1
        assert 'return 99' not in source
        assert corrected in source and helper in source
    correction = calls[-1]['page_correction']
    assert correction['immutable_conflicts'][0]['immutable_source'] == original
    assert correction['pending_declarations'] == [pending]
    assert calls[-1]['region_page']['accepted_sha256'] == calls[-2]['region_page']['accepted_sha256']
    assert all(p['page_index'] != 1 for p in decisions if p['completion_phase'] == 'assess_completion')


def test_identical_echo_cannot_claim_new_progress():
    original = 'private static int stored() { return 7; }'
    responses = iter([ir.OutputBudgetExhausted('OUTPUT_BUDGET_EXHAUSTED'), original, original])

    def coder(_):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    executor = _executor([])
    executor.call_coder = coder
    executor.completion_decider = lambda _: {'done': False, 'next_work': 'implement missing result()'}
    executor.write_source = lambda *_: pytest.fail('no-progress region must not commit')
    with pytest.raises(CustomModuleGenerationError, match='NO_PROGRESS'):
        executor.run()


@pytest.mark.parametrize('changed', [
    'private static String key() { return "a b"; }',
    'private static String key() { return "a\\tb"; }',
    'public static String key() { return "ab"; }',
    'private static String key() { return "ab".trim(); }',
])
def test_echo_identity_preserves_literals_modifiers_and_executable_tokens(changed):
    from minecraft_mod_ai.atomic_region_paging import _member_page_delta

    original = 'private static String key() { return "ab"; }'
    additions, echoed, conflicts = _member_page_delta([changed], [original])
    assert not additions and not echoed
    assert conflicts[0]['immutable_source'] == original
    assert conflicts[0]['rejected_source'] == changed
