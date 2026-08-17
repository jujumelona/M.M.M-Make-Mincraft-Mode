from __future__ import annotations
import hashlib
import json
from contextlib import contextmanager
import pytest
import minecraft_mod_ai.complete_planner as complete_planner_module
import minecraft_mod_ai.game_design as game_design_module
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.game_design import GameDesignPlanner
from minecraft_mod_ai.spec import SpecValidationError
EARLY_REQUIREMENT = 'EARLY_GRAVITY_LANTERN_REQUIREMENT'
LATE_REQUIREMENT = 'LATE_TIDAL_COMPASS_REQUIREMENT'

def _page_design(text: str) -> dict[str, object]:
    text_snippet = text[:100].replace('\n', ' ').strip()
    markers = [marker for marker in (EARLY_REQUIREMENT, LATE_REQUIREMENT) if marker in text]
    return {'game_design': {'title': 'Bounded request page', 'pitch': 'Preserve ' + ', '.join(markers) if markers else f'Preserve this bounded request page: {text_snippet}', 'core_loop': [f'implement {marker}' for marker in markers or [text_snippet]], 'progression': ['initial milestone'], 'combat': {}, 'mod_context': {}, 'modules': [{'plugin_id': 'custom', 'status': 'custom', 'reason': marker} for marker in markers or ['custom_feature']], 'assets': [], 'acceptance_tests': [f'{marker} is observable' for marker in markers or ['feature is observable']]}}

def _patch_research_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    brief = {'schema_version': 'minecraft-mod-ai/research-brief-v1', 'domains': []}
    monkeypatch.setattr(game_design_module, 'normalize_research_brief', lambda prompt, design: brief)
    monkeypatch.setattr(complete_planner_module, 'normalize_research_brief', lambda prompt, design: brief)
    monkeypatch.setattr(complete_planner_module, 'collect_technology_radar', lambda *args, **kwargs: {'requirements': []})
    monkeypatch.setattr(complete_planner_module, '_retrieve_implementation_evidence', lambda *args, **kwargs: None)
    monkeypatch.setattr(complete_planner_module, 'collect_ecosystem_seed_bundle', lambda *args, **kwargs: None)

def test_more_than_6144_word_request_is_losslessly_paged_through_production(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_research_to_empty(monkeypatch)
    prompt = f'  \n{EARLY_REQUIREMENT}: lantern gravity must invert on redstone.\n' + 'neutral_requirement_context ' * 6500 + f'\n{LATE_REQUIREMENT}: compass must track the current tide.\n  '
    assert len(prompt.split()) > 6144
    router = _LosslessWorkflowRouter()
    proposal = CompleteGameDesignPlanner(router).plan(prompt)
    ingestion = proposal.game_design['_request_ingestion']
    production_ingestion = proposal.game_design['_request_production_ingestion']
    assert proposal.requested_prompt == prompt
    assert ingestion['prompt_sha256'] == hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    assert ingestion['page_count'] > 5
    assert production_ingestion['page_count'] == ingestion['page_count']
    assert production_ingestion['batch_count'] == 2
    assert len(router.design_pages) == ingestion['page_count']
    assert len(router.outline_pages) == ingestion['page_count']
    assert ''.join((page['authoritative_request_text'] for page in router.design_pages)) == prompt
    assert ''.join((page['request_ingestion_page']['authoritative_request_text'] for page in router.outline_pages)) == prompt
    rendered_design = json.dumps(proposal.game_design, ensure_ascii=False)
    rendered_outline = json.dumps(proposal.game_design['production_outline'], ensure_ascii=False)
    assert EARLY_REQUIREMENT in rendered_design
    assert LATE_REQUIREMENT in rendered_design
    assert EARLY_REQUIREMENT in rendered_outline
    assert LATE_REQUIREMENT in rendered_outline
    assert any((EARLY_REQUIREMENT in scope for scope in router.expansion_scopes))
    assert any((LATE_REQUIREMENT in scope for scope in router.expansion_scopes))
    outline = proposal.game_design['production_outline']
    assert outline[0]['depends_on_batches'] == []
    assert all((item['depends_on_batches'] for item in outline[1:]))
    gameplay_modules = [module for module in proposal.modules if module.kind == 'custom_java']
    gameplay_ids = {module.module_id for module in gameplay_modules}
    assert not set(gameplay_modules[0].depends_on) & gameplay_ids
    assert all((set(module.depends_on) & gameplay_ids for module in gameplay_modules[1:]))
    assert any((module.config['observed_early'] for module in gameplay_modules))
    assert any((module.config['observed_late'] for module in gameplay_modules))
    assert max(router.message_bytes) < 40000
    assert router.session_events == ['enter:planner', 'exit:planner']

class _MalformedSecondDesignPageRouter:

    def generate_text(self, role, messages, **kwargs):
        del role, kwargs
        content = messages[-1]['content']
        try:
            request = json.loads(content)
        except Exception:
            request = {'authoritative_request_text': content, 'page': {'page_index': 0}}
        if isinstance(request, dict) and request.get('page', {}).get('page_index') == 1:
            return '{"game_design":'
        req_text = request.get('authoritative_request_text', content) if isinstance(request, dict) else content
        return json.dumps(_page_design(req_text), ensure_ascii=False)

def test_malformed_large_request_page_fails_closed_at_exact_no_progress() -> None:
    prompt = 'first requirement ' + 'bounded filler ' * 2500 + 'last requirement'
    with pytest.raises(SpecValidationError, match='exact no-progress cycle'):
        GameDesignPlanner(_MalformedSecondDesignPageRouter()).plan(prompt)

class _ValidDesignPageRouter:

    def generate_text(self, role, messages, **kwargs):
        del role, kwargs
        content = messages[-1]['content']
        try:
            request = json.loads(content)
        except Exception:
            request = {'authoritative_request_text': content, 'page': {'page_index': 0}}
        req_text = request.get('authoritative_request_text', content) if isinstance(request, dict) else content
        return json.dumps(_page_design(req_text), ensure_ascii=False)

def test_large_request_research_classification_is_itself_losslessly_paged() -> None:
    prompt = 'generic capability requirement ' * 2500
    design, proposal = GameDesignPlanner(_ValidDesignPageRouter()).plan(prompt)
    brief = design['_research_brief']
    assert brief['schema_version'] in ('mmm/central-research-brief-v1', 'minecraft-mod-ai/research-brief-v1')
    research_ingestion = brief['request_ingestion']
    assert proposal.requested_prompt == prompt
    assert research_ingestion['page_count'] > 1
    assert research_ingestion['prompt_sha256'] == hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    assert len(brief['domains']) >= research_ingestion['page_count']
    assert len({domain['domain_id'] for domain in brief['domains']}) == len(brief['domains'])

def test_empty_prompt_has_readable_error() -> None:
    with pytest.raises(SpecValidationError, match='프롬프트를 입력해 주세요'):
        GameDesignPlanner(_MalformedSecondDesignPageRouter()).plan('   ')
