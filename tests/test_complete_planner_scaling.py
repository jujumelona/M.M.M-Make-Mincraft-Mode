from __future__ import annotations
import json
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
import minecraft_mod_ai.complete_planner as planner_module
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner, _ProductionBatch, _extract_json, _implementation_prompt, _remove_bootstrap_duplicates
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.spec import ContentKind, ContentSpec, SpecValidationError

def _module(module_id: str) -> dict[str, object]:
    return {'module_id': module_id, 'kind': 'custom_java', 'config': {}, 'depends_on': [], 'required_gates': []}

class _LongPaginationRouter:

    def __init__(self, page_count: int) -> None:
        self.page_count = page_count
        self.requests: list[dict[str, object]] = []
        self.request_sizes: list[int] = []
        self.media_paths: list[tuple[str, ...]] = []
        self.expected_cursor = ''

    def generate_text(self, role, messages, **kwargs):
        assert role == 'planner'
        user_content = messages[-1]['content']
        request = json.loads(user_content)
        assert request['cursor'] == self.expected_cursor
        self.requests.append(request)
        self.request_sizes.append(len(user_content.encode('utf-8')))
        self.media_paths.append(tuple((str(path) for path in kwargs['media_paths'])))
        index = len(self.requests) - 1
        complete = index == self.page_count - 1
        next_cursor = '' if complete else f'opaque/{index + 1:08d}?token=unchanged-width'
        self.expected_cursor = next_cursor
        return json.dumps({'modules': [_module(f'feature_{index:08d}')], 'complete': complete, 'next_cursor': next_cursor})

def test_module_pagination_request_stays_bounded_across_many_pages() -> None:
    router = _LongPaginationRouter(page_count=600)
    planner = CompleteGameDesignPlanner(router)
    huge_evidence_marker = 'must-not-be-resent-' * 20000
    modules = planner._expand_batches(prompt='Build every system in this self-contained batch.', game_design={'title': 'Unbounded production graph', 'pitch': 'Compile the complete requested graph.', 'modules': [{'plugin_id': 'custom', 'reason': 'requested'}], '_technical_evidence': {'schema_version': 'test/evidence-v1', 'untrusted_excerpt': huge_evidence_marker}}, batches=[{'batch_id': 'all_requested_systems', 'scope': 'Implement all 600 requested independent systems and their observable validation hooks.', 'depends_on_batches': []}], media_paths=('reference.png',))
    assert len(modules) == 600
    assert len({module.module_id for module in modules}) == 600
    assert 'planning_context' in router.requests[0]
    assert all(('planning_context' not in request for request in router.requests[1:]))
    assert all(('known_module_ids' not in request for request in router.requests))
    assert all((huge_evidence_marker not in json.dumps(request) for request in router.requests))
    for index, request in enumerate(router.requests):
        catalog = request['known_module_catalog']
        assert catalog['count'] == index
        assert len(catalog['sha256']) == 64
        assert len(catalog['recent_ids']) <= 32
        assert catalog['recent_limit'] == 32
    assert len(set((request['known_module_catalog']['sha256'] for request in router.requests))) == 600
    steady_state_sizes = router.request_sizes[64:]
    assert max(steady_state_sizes) - min(steady_state_sizes) <= 2
    assert max(router.request_sizes) < 5000
    assert router.media_paths[0] == ('reference.png',)
    assert all((not paths for paths in router.media_paths[1:]))

class _ResponseRouter:

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict[str, object]] = []
        self.media_paths: list[tuple[str, ...]] = []

    def generate_text(self, role, messages, **kwargs):
        self.requests.append(json.loads(messages[-1]['content']))
        self.media_paths.append(tuple((str(path) for path in kwargs['media_paths'])))
        return json.dumps(next(self.responses))

class _RawResponseRouter:

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict[str, object] | str] = []
        self.messages: list[list[dict[str, str]]] = []
        self.media_paths: list[tuple[str, ...]] = []

    def generate_text(self, role, messages, **kwargs):
        assert role == 'planner'
        user_content = messages[-1]['content']
        try:
            request = json.loads(user_content)
        except json.JSONDecodeError:
            request = user_content
        self.requests.append(request)
        self.messages.append([dict(message) for message in messages])
        self.media_paths.append(tuple((str(path) for path in kwargs['media_paths'])))
        return next(self.responses)

class _SessionRawResponseRouter(_RawResponseRouter):

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.session_events: list[str] = []

    @contextmanager
    def generation_session(self, role: str):
        self.session_events.append(f'enter:{role}')
        try:
            yield self
        finally:
            self.session_events.append(f'exit:{role}')

class _ThinkingModulePageRouter:

    def generate_text(self, role, messages, **kwargs):
        assert role == 'planner'
        assert kwargs['response_format'] == 'json'
        assert json.loads(messages[-1]['content'])['cursor'] == ''
        return '\n'.join((json.dumps({'analysis': 'draft the module page first'}), json.dumps({'modules': [_module('final_module')], 'complete': True, 'next_cursor': '', 'notes': 'Generated after internal reasoning.'})))

def test_module_pagination_skips_qwen_scratch_before_final_page() -> None:
    modules = CompleteGameDesignPlanner(_ThinkingModulePageRouter())._expand_batches(prompt='Create the requested module.', game_design={'title': 'Scratch-safe module'}, batches=[{'batch_id': 'core', 'scope': 'Create the final module.', 'depends_on_batches': []}], media_paths=())
    assert [module.module_id for module in modules] == ['final_module']

def test_legacy_module_batch_repairs_only_the_cut_page() -> None:
    router = _RawResponseRouter(['{"modules":[', json.dumps({'modules': [_module('repaired_module')], 'complete': True, 'next_cursor': ''})])
    modules = CompleteGameDesignPlanner(router)._expand_batches(prompt='Create the repaired module.', game_design={'title': 'Legacy page repair'}, batches=[{'batch_id': 'core', 'scope': 'Create the repaired module.', 'depends_on_batches': []}], media_paths=('reference.png',))
    assert [module.module_id for module in modules] == ['repaired_module']
    assert len(router.requests) == 2
    assert router.requests[0] == router.requests[1]
    assert router.requests[0]['cursor'] == ''
    assert router.media_paths == [('reference.png',), ()]

def test_module_pagination_rejects_duplicate_ids_within_page() -> None:
    planner = CompleteGameDesignPlanner(_ResponseRouter([{'modules': [_module('same_id'), _module('same_id')], 'complete': True, 'next_cursor': ''}]))
    with pytest.raises(SpecValidationError, match='duplicate module ID'):
        planner._expand_batches(prompt='Duplicate test', game_design={'title': 'Duplicate test'}, batches=[{'batch_id': 'one', 'scope': 'one', 'depends_on_batches': []}], media_paths=())

def test_module_pagination_rejects_duplicate_ids_across_batches() -> None:
    planner = CompleteGameDesignPlanner(_ResponseRouter([{'modules': [_module('shared_id')], 'complete': True, 'next_cursor': ''}, {'modules': [_module('shared_id')], 'complete': True, 'next_cursor': ''}]))
    with pytest.raises(SpecValidationError, match='duplicate module ID'):
        planner._expand_batches(prompt='Global duplicate test', game_design={'title': 'Global duplicate test'}, batches=[{'batch_id': 'one', 'scope': 'first', 'depends_on_batches': []}, {'batch_id': 'two', 'scope': 'second', 'depends_on_batches': ['one']}], media_paths=())

def test_full_planning_context_and_media_are_sent_once_across_batches() -> None:
    router = _ResponseRouter([{'modules': [_module(f'module_{index}')], 'complete': True, 'next_cursor': ''} for index in range(3)])
    planner = CompleteGameDesignPlanner(router)
    modules = planner._expand_batches(prompt='Build the complete requested design.', game_design={'title': 'Many batches', 'modules': [{'plugin_id': f'feature_{index}', 'reason': 'requested'} for index in range(1000)], '_technical_evidence': {'excerpt': 'untrusted-' * 10000}}, batches=[{'batch_id': f'batch_{index}', 'scope': f'Self-contained implementation scope {index}', 'depends_on_batches': [] if index == 0 else [f'batch_{index - 1}']} for index in range(3)], media_paths=('large-reference.png',))
    assert len(modules) == 3
    assert 'planning_context' in router.requests[0]
    assert all(('planning_context' not in request for request in router.requests[1:]))
    assert len({request['planning_context_receipt']['sha256'] for request in router.requests}) == 1
    assert router.media_paths == [('large-reference.png',), (), ()]

def test_production_batch_stops_after_one_page_local_repair() -> None:
    router = _RawResponseRouter(['{"modules":[', '{"modules":'])
    with pytest.raises(SpecValidationError, match='failed after one page-local repair'):
        CompleteGameDesignPlanner(router)._expand_production_batches(batches=(_ProductionBatch('core', 'Implement core.', (), ('core',), ('core_module',)),), prompt='Build core.', game_design={'title': 'Bounded repair'}, media_paths=())
    assert len(router.requests) == 2
    assert router.requests[0] == router.requests[1]

def test_production_outline_paginates_without_repeating_full_evidence() -> None:
    router = _ResponseRouter([{'production_batches': [{'batch_id': 'second', 'scope': 'Second implementation scope.', 'depends_on_batches': ['first'], 'deliverables': ['second work'], 'exports': ['second_module']}], 'complete': True, 'next_cursor': ''}])
    planner = CompleteGameDesignPlanner(router)
    batches = planner._collect_production_batches(first_page={'production_batches': [{'batch_id': 'first', 'scope': 'First implementation scope.', 'depends_on_batches': [], 'deliverables': ['first work'], 'exports': ['first_module']}], 'complete': False, 'next_cursor': 'outline-page-2'}, prompt='Build a large project.', game_design={'title': 'Large', '_technical_evidence': {'huge': 'not-forwarded' * 1000}, '_research_brief': {'huge': 'brief-not-forwarded' * 1000}}, media_paths=('reference.png',))
    assert [item.batch_id for item in batches] == ['first', 'second']
    assert router.requests[0]['cursor'] == 'outline-page-2'
    assert '_technical_evidence' not in router.requests[0]['planning_context']['game_design']
    assert all((not key.startswith('_') for key in router.requests[0]['planning_context']['game_design']))
    assert 'brief-not-forwarded' not in json.dumps(router.requests[0])
    assert router.media_paths == [()]

def test_initial_outline_uses_compact_research_facts_and_full_receipt() -> None:
    candidates = [{'candidate_id': f'huggingface:owner/model-{index}', 'provider': 'huggingface_models', 'resource_kind': 'ai_model', 'license_id': 'apache-2.0', 'license_policy': 'reviewable_model_license', 'compatibility': 'unverified', 'reuse_status': 'candidate_only_metadata_not_weights', 'evidence_sha256': 'sha256:' + 'f' * 64, 'metadata': {'revision_sha': f'{index:040x}', 'pipeline_tag': 'text-generation', 'library_name': 'transformers', 'private': False, 'gated': False, 'disabled': False, 'card': {'license_evidence': 'model_card', 'datasets': [f'owner/dataset-{index}', 'bounded-dataset-' + 'x' * 2048], 'languages': ['ko', 'en-' + 'y' * 2048]}, 'format_inventory': {'has_safetensors': True, 'has_gguf': False, 'has_onnx': False, 'unsafe_serialization_files': ['weights.bin', 'model.pkl'], 'repository_code_files': ['modeling.py']}, 'untrusted_description': 'never-forward-this-' * 1000}} for index in range(200)]
    design = {'title': 'Large researched mod', 'pitch': 'Preserve all requested systems through pages.', '_research_brief': {'internal': 'PRIVATE_RESEARCH_BRIEF_MUST_NOT_DUPLICATE-' * 1000}, '_ecosystem_discovery': {'schema_version': 'mmm/ecosystem-seed-bundle-v1', 'status': 'available', 'route_sha256': 'a' * 64, 'route_count': 200, 'coverage': 'complete', 'pages': [{'provider': 'huggingface_models', 'candidates': candidates}]}}
    rendered = _implementation_prompt('Build the researched mod.', design)
    assert 'huggingface:owner/model-0' in rendered
    assert 'huggingface:owner/model-199' not in rendered
    assert 'never-forward-this' not in rendered
    assert 'PRIVATE_RESEARCH_BRIEF_MUST_NOT_DUPLICATE' not in rendered
    assert 'candidate_only_metadata_not_weights' in rendered
    assert '"evidence": "model_card"' in rendered
    assert '"has_safetensors": true' in rendered
    assert '"unsafe_serialization_file_count": 2' in rendered
    assert '"repository_code_file_count": 1' in rendered
    assert 'x' * 300 not in rendered
    assert 'y' * 300 not in rendered
    assert 'full_context_receipt' in rendered
    assert len(rendered.encode('utf-8')) < 12000

def test_initial_outline_keeps_one_typed_technical_query_per_domain() -> None:
    design = {'title': 'Evidence-backed systems', '_technical_evidence': {'schema_version': 'mmm/central-evidence-graph-v1', 'brief_sha256': 'sha256:' + 'a' * 64, 'evidence_sha256': 'sha256:' + 'b' * 64, 'unresolved_official_domains': ['unresolved_runtime'], 'domains': [{'domain_id': 'fabric_runtime', 'strategy': 'adaptive_per_query', 'queries': [{'query_sha256': 'sha256:' + 'c' * 64, 'strategy': 'corrective_multi_hop', 'primary': {'query_hash': 'sha256:' + 'd' * 64, 'corpus_snapshot_hash': 'sha256:' + 'e' * 64, 'quality': 'strong', 'coverage': 0.875, 'correction_required': True, 'hits': [{'document_id': 'fabric-api-events', 'excerpt': 'UNTRUSTED EXCERPT MUST NOT PASS'}, {'document_id': 'fabric-networking'}]}, 'corrections': [{'quality': 'strong'}]}, {'query_sha256': 'SECOND_QUERY_MUST_STAY_OUT', 'primary': {'quality': 'weak', 'hits': []}, 'corrections': []}]}]}}
    rendered = _implementation_prompt('Build the evidence-backed mod.', design)
    encoded_context = rendered.split('Compact authoritative planning context:\n', 1)[1].split('\n\nCreate only the paginated production outline.', 1)[0]
    technical = json.loads(encoded_context)['research_outline']['technical_evidence']
    domain = technical['domains'][0]
    query = domain['representative_query']
    assert technical['domain_count'] == 1
    assert technical['unresolved_official_domain_count'] == 1
    assert domain['query_count'] == 2
    assert query['quality'] == 'strong'
    assert query['coverage'] == 0.875
    assert query['correction_required'] is True
    assert query['correction_count'] == 1
    assert query['hit_ids'] == ['fabric-api-events', 'fabric-networking']
    assert 'SECOND_QUERY_MUST_STAY_OUT' not in rendered
    assert 'UNTRUSTED EXCERPT MUST NOT PASS' not in rendered
    assert len(rendered.encode('utf-8')) < 12000

def test_production_outline_repairs_only_the_cut_continuation_page() -> None:
    router = _RawResponseRouter(['{"production_batches":[', json.dumps({'production_batches': [{'batch_id': 'second', 'scope': 'Second implementation scope.', 'depends_on_batches': ['first'], 'deliverables': ['second work'], 'exports': ['second_module']}], 'complete': True, 'next_cursor': ''})])
    batches = CompleteGameDesignPlanner(router)._collect_production_batches(first_page={'production_batches': [{'batch_id': 'first', 'scope': 'First implementation scope.', 'depends_on_batches': [], 'deliverables': ['first work'], 'exports': ['first_module']}], 'complete': False, 'next_cursor': 'outline-page-2'}, prompt='Build a large project.', game_design={'title': 'Outline repair'}, media_paths=())
    assert [item.batch_id for item in batches] == ['first', 'second']
    assert len(router.requests) == 2
    retry_request = dict(router.requests[1])
    diagnostic = retry_request.pop('missing_fragment_reason')
    assert retry_request == router.requests[0]
    assert 'no complete production-outline JSON page' in diagnostic
    assert router.requests[0]['cursor'] == 'outline-page-2'
    assert router.requests[0]['known_batch_catalog']['count'] == 1
    assert all(([message['role'] for message in attempt] == ['system', 'user'] for attempt in router.messages))

def _bootstrap_base():
    content = ContentSpec(content_id='bootstrap_relic', kind=ContentKind.ITEM, display_name_en='Bootstrap Relic', display_name_ko='Bootstrap Relic KO', color='#123456', recipe=True)
    return SimpleNamespace(spec=SimpleNamespace(contents=(content,)))

def test_equivalent_bootstrap_duplicate_remains_backward_compatible() -> None:
    modules = _remove_bootstrap_duplicates((ProductionModule(module_id='bootstrap_relic', kind='item', config={'display_name_en': 'Bootstrap Relic', 'display_name_ko': 'Bootstrap Relic KO', 'color': '#123456', 'recipe': True}, required_gates=('registry', 'resource', 'recipe')),), _bootstrap_base())
    assert len(modules) == 1
    assert modules[0].module_id == 'bootstrap_integration'
    assert modules[0].config == {'uses_base_content': ['bootstrap_relic']}

@pytest.mark.parametrize('module', (ProductionModule(module_id='bootstrap_relic', kind='item', config={'attack_damage': 12}), ProductionModule(module_id='bootstrap_relic', kind='item', depends_on=('progression_core',)), ProductionModule(module_id='bootstrap_relic', kind='item', required_gates=('multiplayer runtime proof',))))
def test_richer_bootstrap_duplicate_routes_to_custom_extension(module: ProductionModule) -> None:
    modules = _remove_bootstrap_duplicates((module,), _bootstrap_base())
    assert len(modules) == 1
    extension = modules[0]
    assert extension.module_id == module.module_id
    assert extension.kind == 'custom_java'
    assert extension.config == {**module.config, 'requested_kind': 'item', 'extends_bootstrap': 'bootstrap_relic'}
    assert extension.depends_on == module.depends_on
    assert extension.required_gates == module.required_gates

def test_bootstrap_duplicate_kind_mismatch_still_fails_closed() -> None:
    with pytest.raises(SpecValidationError, match='collides with bootstrap item'):
        _remove_bootstrap_duplicates((ProductionModule(module_id='bootstrap_relic', kind='block'),), _bootstrap_base())
