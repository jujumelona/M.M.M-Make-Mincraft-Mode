from __future__ import annotations

from copy import deepcopy

import pytest

from minecraft_mod_ai.central_research import (
    normalize_research_brief,
    retrieve_domain_evidence,
)
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.retrieval import RetrievalHit, RetrievalReceipt
from minecraft_mod_ai.spec import SpecValidationError


def _domain(domain_id: str, *, query: str | None=None, providers: list[str] | None=None, depends_on: list[str] | None=None) -> dict[str, object]:
    return {'domain_id': domain_id, 'objective': f'Research {domain_id} without assuming a genre template.', 'requirements': [f'Preserve the requested {domain_id} capability.'], 'evidence_kinds': ['dependency', 'compatibility', 'license'], 'queries': [query or f'{domain_id} implementation evidence'], 'providers': providers or ['official_docs', 'github'], 'depends_on': depends_on or []}

def _candidate(domains: list[dict[str, object]]) -> dict[str, object]:
    return {'summary': 'A generic request-derived research DAG.', 'domains': domains, 'unresolved_questions': []}

def _selected_design(version: str='1.20.1', loader: str='fabric') -> dict[str, object]:
    return {'_platform_selection': {'target': {'minecraft_version': version, 'loader': loader}}}

def test_normalize_research_brief_preserves_host_selected_target() -> None:
    normalized = normalize_research_brief('Build the requested simulation.', _selected_design('1.21.1', 'fabric'), _candidate([_domain('request')]))
    assert normalized['_mmm_platform_target'] == {'minecraft_version': '1.21.1', 'loader': 'fabric'}

def test_normalize_research_brief_rejects_cycles_and_unknown_providers() -> None:
    cyclic = _candidate([_domain('first', depends_on=['second']), _domain('second', depends_on=['first'])])
    with pytest.raises(SpecValidationError, match='cycle'):
        normalize_research_brief('generic request', {}, cyclic)
    unknown_provider = deepcopy(cyclic)
    unknown_provider['domains'][0]['depends_on'] = []
    unknown_provider['domains'][1]['depends_on'] = ['first']
    unknown_provider['domains'][1]['providers'] = ['unreviewed_catalog']
    with pytest.raises(SpecValidationError, match='unknown providers'):
        normalize_research_brief('generic request', {}, unknown_provider)

@pytest.mark.parametrize(('prompt', 'systems'), [('Build a competitive handball league with passing, scoring, and seasons.', ('passing', 'scoring', 'seasons')), ('Build social deduction horror with trust voting, radio whispers, and hiding.', ('trust voting', 'radio whispers', 'hiding')), ('Build a cozy farming game with planting, watering, harvesting, and a market.', ('planting', 'watering', 'harvesting', 'market'))])
def test_fallback_is_request_derived_without_genre_content_injection(prompt: str, systems: tuple[str, ...]) -> None:
    design = {'core_loop': list(systems[:2]), 'progression': [systems[2]], 'combat': {}, 'world': {}, 'modules': [{'plugin_id': 'custom', 'reason': f'Implement {systems[-1]} exactly as requested.'}], 'assets': [], 'acceptance_tests': [f'Players can complete {systems[0]}.']}
    normalized = normalize_research_brief(prompt, design)
    serialized = ' '.join(requirement for domain in normalized['domains'] for requirement in domain['requirements']).casefold()
    assert normalized['origin'] == 'deterministic_fallback'
    assert prompt.casefold() in serialized
    assert all(system.casefold() in serialized for system in systems)
    assert not {'boss', 'arena', 'village', 'dungeon'} & set(serialized.replace('.', '').replace(',', '').split())

def test_generic_request_does_not_invent_visual_media_provider() -> None:
    brief = normalize_research_brief('Add a server command that reports the current tick count.', {'assets': []})
    assert all('openverse_images' not in domain['providers'] and 'visual_reference' not in domain['evidence_kinds'] for domain in brief['domains'])

def _receipt(query: str, *, correction_queries: tuple[str, ...]=()) -> RetrievalReceipt:
    hit = RetrievalHit(evidence_id='sha256:' + '1' * 64, document_id='fabric-api-1201', title='Fabric API 1.20.1', url='https://maven.fabricmc.net/', excerpt=f'Evidence for {query}', content_sha256='sha256:' + '2' * 64, revision='fabric-api-0.92.11+1.20.1', minecraft_versions=('1.20.1',), score=1.0, channels=('test',))
    return RetrievalReceipt(schema_version='minecraft-mod-ai/retrieval-receipt-v1', query=query, canonical_query=query, query_family='project', minecraft_version='1.20.1', loader='fabric', mappings='1.20.1+build.1', query_hash='sha256:' + '3' * 64, corpus_snapshot_hash='sha256:' + '4' * 64, quality='strong', coverage=1.0, correction_required=bool(correction_queries), correction_queries=correction_queries, hits=(hit,))

def test_targetless_official_research_is_deferred_without_retrieval() -> None:
    brief = normalize_research_brief('Research all routed facts.', {}, _candidate([_domain('official_one', providers=['official_docs'])]))
    calls: list[str] = []

    def fake_retrieve(query: str, **_kwargs: object) -> RetrievalReceipt:
        calls.append(query)
        return _receipt(query)
    evidence = retrieve_domain_evidence(brief, retrieve=fake_retrieve)
    assert calls == []
    assert evidence['target'] is None
    assert evidence['deferred_official_domains'] == ['official_one']
    assert evidence['unresolved_official_domains'] == []
    assert evidence['domains'][0]['strategy'] == 'deferred_until_platform_selected'

def test_retrieve_domain_evidence_covers_authored_and_declared_criteria_without_speculative_corrections() -> None:
    official_queries = ('official query alpha', 'official query beta', 'official query gamma')
    brief = normalize_research_brief('Research all routed facts.', _selected_design(), _candidate([{**_domain('official_one', providers=['official_docs']), 'queries': list(official_queries[:2])}, {**_domain('official_two', providers=['official_docs', 'github']), 'queries': [official_queries[2]], 'depends_on': ['official_one']}, {**_domain('external_only', query='must not use official retrieval', providers=['github'], depends_on=['official_one']), 'evidence_kinds': ['source_code']}]))
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_retrieve(query: str, **kwargs: object) -> RetrievalReceipt:
        calls.append((query, kwargs))
        if query in official_queries:
            # Baseline authored queries are seeds only; their correction suggestions
            # must not trigger speculative fan-out under the coverage-driven policy.
            return _receipt(query, correction_queries=(f'{query} correction one', f'{query} correction two'))
        return _receipt(query)
    evidence = retrieve_domain_evidence(brief, retrieve=fake_retrieve)
    called_queries = [query for query, _ in calls]
    assert all(called_queries.count(query) == 1 for query in official_queries)
    assert not any(' correction ' in query for query in called_queries)
    assert 'Preserve the requested official_one capability.' in called_queries
    assert 'Preserve the requested official_two capability.' in called_queries
    assert any('dependency official_one' in query for query in called_queries)
    assert any('compatibility official_one' in query for query in called_queries)
    assert any('license official_one' in query for query in called_queries)
    adapter = adapter_for_target('1.20.1', 'fabric')
    assert all((kwargs['minecraft_version'] == adapter.minecraft_version and kwargs['loader'] == adapter.loader and kwargs['mappings'] == adapter.yarn_mappings and kwargs['limit'] == 8 for _, kwargs in calls))
    assert evidence['target'] == {'minecraft_version': adapter.minecraft_version, 'loader': adapter.loader, 'mappings': adapter.yarn_mappings}
    assert evidence['deferred_official_domains'] == []
    assert evidence['unresolved_official_domains'] == []
    assert evidence['retrieval_is_authority'] is False

def test_selected_target_drives_every_rag_route_without_model_choice() -> None:
    brief = normalize_research_brief('Research exact target evidence.', _selected_design('1.21.1', 'fabric'), _candidate([_domain('official_one', providers=['official_docs'])]))
    calls: list[dict[str, object]] = []

    def fake_retrieve(query: str, **kwargs: object) -> RetrievalReceipt:
        calls.append(kwargs)
        return _receipt(query)
    evidence = retrieve_domain_evidence(brief, retrieve=fake_retrieve)
    adapter = adapter_for_target('1.21.1', 'fabric')
    expected = {'minecraft_version': adapter.minecraft_version, 'loader': adapter.loader, 'mappings': adapter.yarn_mappings, 'limit': 8}
    assert calls
    assert all(call == expected for call in calls)
    assert evidence['target'] == {'minecraft_version': adapter.minecraft_version, 'loader': adapter.loader, 'mappings': adapter.yarn_mappings}
