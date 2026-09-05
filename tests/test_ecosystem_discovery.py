import base64
import hashlib
import json

import httpx
import pytest

from minecraft_mod_ai.ecosystem_discovery import (
    EcosystemDiscoveryClient,
    EcosystemDiscoveryUnavailable,
    discover_seed_bundle,
)
from minecraft_mod_ai.spec import SpecValidationError


def _client(handler, **kwargs) -> EcosystemDiscoveryClient:
    return EcosystemDiscoveryClient(transport=httpx.MockTransport(handler), **kwargs)

def test_modrinth_search_uses_exact_facets_and_bound_offset_pagination() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == 'api.modrinth.com'
        assert request.url.path == '/v2/search'
        assert json.loads(request.url.params['facets']) == [['versions:1.20.1'], ['categories:fabric'], ['project_type:mod'], ['open_source:true']]
        offset = int(request.url.params['offset'])
        return httpx.Response(200, json={'total_hits': 5, 'hits': [{'project_id': f'project-{offset}', 'slug': f'project-{offset}', 'project_type': 'mod', 'title': f'Project {offset}', 'description': 'A Fabric mod', 'license': 'MIT', 'versions': ['1.20.1'], 'categories': ['fabric'], 'icon_url': 'https://cdn.modrinth.com/data/icon.png'}]})
    client = _client(handler)
    first = client.search('modrinth', 'ship navigation', limit=2, minecraft_version='1.20.1', loader='fabric')
    assert first['returned'] == 1
    assert first['provider_total_estimate'] == 5
    assert first['next_cursor'].startswith('offset:2:')
    assert first['download_performed'] is False
    second = client.search('modrinth', 'ship navigation', limit=2, cursor=first['next_cursor'], minecraft_version='1.20.1', loader='fabric')
    assert len(requests) == 2
    assert requests[1].url.params['offset'] == '2'
    assert second['next_cursor'].startswith('offset:4:')

def test_discovery_cursor_is_bound_to_query_and_provider() -> None:

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == 'api.github.com':
            return httpx.Response(200, json={'total_count': 2, 'items': [{'id': 1, 'full_name': 'example/fabric-library', 'html_url': 'https://github.com/example/fabric-library', 'description': 'Library', 'default_branch': 'main', 'archived': False, 'license': {'spdx_id': 'MIT'}, 'topics': ['fabric']}]})
        raise AssertionError('A rejected cursor must not make another request')
    client = _client(handler)
    page = client.search('github', 'navigation', limit=1)
    cursor = page['next_cursor']
    assert cursor.startswith('page:2:')
    with pytest.raises(SpecValidationError, match='provider and query'):
        client.search('github', 'unrelated economy', cursor=cursor, limit=1)
    with pytest.raises(SpecValidationError, match='provider and query'):
        client.search('openverse_images', 'navigation', cursor=cursor, limit=1)

def test_modrinth_project_inspection_preserves_exact_hashes_dependencies_and_license() -> None:
    requests: list[httpx.Request] = []
    sha1 = '1' * 40
    sha512 = 'a' * 128

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == '/v2/project/exact-project':
            return httpx.Response(200, json={'id': 'abc123', 'slug': 'exact-project', 'title': 'Exact Project', 'license': {'id': 'Apache-2.0', 'url': 'https://www.apache.org/licenses/LICENSE-2.0'}})
        assert request.url.path == '/v2/project/exact-project/version'
        assert json.loads(request.url.params['loaders']) == ['fabric']
        assert json.loads(request.url.params['game_versions']) == ['1.20.1']
        assert request.url.params['include_changelog'] == 'false'
        return httpx.Response(200, json=[{'id': 'version-1', 'version_number': '4.2.0+1.20.1', 'version_type': 'release', 'status': 'listed', 'date_published': '2026-07-01T00:00:00Z', 'game_versions': ['1.20.1'], 'loaders': ['fabric'], 'dependencies': [{'project_id': 'fabric-api', 'version_id': None, 'dependency_type': 'required'}, {'project_id': 'incompatible-mod', 'version_id': 'bad-version', 'dependency_type': 'incompatible'}], 'files': [{'filename': 'exact-project.jar', 'url': 'https://cdn.modrinth.com/data/abc/versions/v1/exact.jar', 'size': 123456, 'primary': True, 'hashes': {'sha1': sha1, 'sha512': sha512}}]}])
    result = _client(handler).inspect_modrinth_project('exact-project', minecraft_version='1.20.1', loader='fabric')
    assert len(requests) == 2
    assert result['license_id'] == 'Apache-2.0'
    assert result['license_policy'].startswith('permissive_candidate')
    assert result['exact_compatible_version_found'] is True
    assert result['download_performed'] is False
    version = result['versions'][0]
    assert version['dependencies'] == [{'project_id': 'fabric-api', 'version_id': None, 'dependency_type': 'required'}, {'project_id': 'incompatible-mod', 'version_id': 'bad-version', 'dependency_type': 'incompatible'}]
    assert version['files'][0]['sha1'] == sha1
    assert version['files'][0]['sha512'] == sha512
    assert 'verify the advertised SHA-512' in result['required_next_gate']

def test_modrinth_inspection_does_not_accept_off_target_or_ambiguous_files() -> None:

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == '/v2/project/not-exact':
            return httpx.Response(200, json={'id': 'not-exact', 'slug': 'not-exact', 'title': 'Not Exact', 'license': {'id': 'MIT', 'url': ''}})
        return httpx.Response(200, json=[{'id': 'off-target-version', 'version_number': '1.0.0', 'version_type': 'release', 'status': 'listed', 'game_versions': ['1.20.4'], 'loaders': ['fabric'], 'dependencies': [], 'files': [{'filename': 'one.jar', 'url': 'https://cdn.modrinth.com/data/one.jar', 'size': 1, 'primary': True, 'hashes': {'sha1': '1' * 40, 'sha512': 'a' * 128}}, {'filename': 'two.jar', 'url': 'https://cdn.modrinth.com/data/two.jar', 'size': 1, 'primary': True, 'hashes': {'sha1': '2' * 40, 'sha512': 'b' * 128}}]}])
    result = _client(handler).inspect_modrinth_project('not-exact', minecraft_version='1.20.1', loader='fabric')
    assert result['exact_compatible_version_found'] is False
    assert result['eligible_version_ids'] == []
    version = result['versions'][0]
    assert version['eligible_for_selection'] is False
    assert 'off_target_minecraft_version_or_loader' in version['unresolved_or_rejected_gates']
    assert 'exactly_one_primary_file_is_required' in version['unresolved_or_rejected_gates']

def test_github_search_skips_missing_and_unasserted_licenses() -> None:

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == 'api.github.com'
        return httpx.Response(200, json={'total_count': 3, 'items': [{'id': 1, 'full_name': 'example/no-license', 'html_url': 'https://github.com/example/no-license', 'archived': False, 'license': None}, {'id': 2, 'full_name': 'example/noassertion', 'html_url': 'https://github.com/example/noassertion', 'archived': False, 'license': {'spdx_id': 'NOASSERTION'}}, {'id': 3, 'full_name': 'example/reviewable', 'html_url': 'https://github.com/example/reviewable', 'description': 'Reviewable source', 'default_branch': 'main', 'archived': False, 'license': {'spdx_id': 'BSD-3-Clause'}, 'topics': ['minecraft', 'fabric']}]})
    page = _client(handler).search('github', 'trade economy', limit=10)
    assert [item['candidate_id'] for item in page['candidates']] == ['github:example/reviewable']
    assert page['candidates'][0]['reuse_status'] == 'candidate_only_not_cloned'
    assert page['download_performed'] is False

def test_github_inspection_pins_commit_and_hashes_license_without_cloning() -> None:
    license_bytes = b'MIT License\n\nPermission is hereby granted...'
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == '/repos/example/reviewable':
            return httpx.Response(200, json={'full_name': 'example/reviewable', 'html_url': 'https://github.com/example/reviewable', 'default_branch': 'main', 'private': False, 'archived': False})
        if request.url.path == '/repos/example/reviewable/commits/main':
            return httpx.Response(200, json={'sha': 'a' * 40, 'commit': {'verification': {'verified': True}}})
        assert request.url.path == '/repos/example/reviewable/license'
        return httpx.Response(200, json={'encoding': 'base64', 'content': base64.b64encode(license_bytes).decode('ascii'), 'html_url': 'https://github.com/example/reviewable/blob/main/LICENSE', 'license': {'spdx_id': 'MIT'}})
    result = _client(handler).inspect_github_repository('example/reviewable')
    assert len(requests) == 3
    assert result['commit_sha'] == 'a' * 40
    assert result['commit_verified'] is True
    assert result['license_id'] == 'MIT'
    assert result['license_text_sha256'] == 'sha256:' + hashlib.sha256(license_bytes).hexdigest()
    assert result['download_performed'] is False
    assert result['compatibility'].startswith('unverified')

def test_openverse_keeps_license_attribution_and_origin_verification_without_download() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == 'api.openverse.org'
        assert request.url.path == '/v1/images/'
        assert request.url.params['license'] == 'cc0,pdm,by,by-sa'
        assert request.url.params['license_type'] == 'modification'
        assert request.url.params['mature'] == 'false'
        return httpx.Response(200, json={'result_count': 1, 'page_count': 1, 'results': [{'id': 'museum-ship', 'title': 'Historic sailing ship', 'creator': 'Example Museum', 'license': 'by', 'license_version': '4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/', 'foreign_landing_url': 'https://museum.example/items/ship', 'detail_url': 'https://api.openverse.org/v1/images/museum-ship/', 'thumbnail': 'https://images.openverse.org/thumb.jpg', 'attribution': 'Historic sailing ship by Example Museum, CC BY 4.0', 'provider': 'wikimedia', 'source': 'wikimedia', 'mature': False}]})
    page = _client(handler).search('openverse_images', 'historic ship', limit=5)
    assert len(requests) == 1
    candidate = page['candidates'][0]
    assert candidate['license_id'] == 'CC-BY-4.0'
    assert candidate['source_url'] == 'https://museum.example/items/ship'
    assert candidate['attribution'].endswith('CC BY 4.0')
    assert candidate['reuse_status'] == 'origin_license_verification_required'
    assert candidate['license_policy'] == 'origin_verification_and_attribution_required'
    assert page['download_performed'] is False
    assert page['authorization'] == 'none'

def test_discovery_rejects_api_host_escape_before_transport() -> None:

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError('Disallowed hosts must never reach the transport')
    client = _client(handler)
    with pytest.raises(SpecValidationError, match='API allowlist'):
        client._get_json('https://attacker.example/v2/search')

def test_discovery_does_not_follow_redirects() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={'Location': 'https://attacker.example/collect'})
    client = _client(handler)
    with pytest.raises(EcosystemDiscoveryUnavailable, match='HTTP 302'):
        client.search('modrinth', 'navigation')
    assert len(requests) == 1
    assert requests[0].url.host == 'api.modrinth.com'

def test_discovery_rejects_response_over_byte_limit_before_json_decode() -> None:
    oversized = b'{' * (4 * 1024 * 1024 + 1)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized)
    with pytest.raises(EcosystemDiscoveryUnavailable, match='byte policy'):
        _client(handler).search('modrinth', 'navigation')

def test_wikipedia_is_reference_only_and_uses_language_bound_pagination() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == 'ko.wikipedia.org'
        assert request.url.params['list'] == 'search'
        assert request.url.params['srsearch'] == '축구 경기 규칙'
        return httpx.Response(200, json={'query': {'searchinfo': {'totalhits': 3}, 'search': [{'pageid': 123, 'title': '축구', 'snippet': '<span>팀 스포츠</span> 경기 규칙', 'timestamp': '2026-01-01T00:00:00Z'}]}, 'continue': {'sroffset': 1, 'continue': '-||'}})
    page = _client(handler).search('wikipedia', '축구 경기 규칙', limit=1)
    assert page['next_cursor'].startswith('offset:1:')
    candidate = page['candidates'][0]
    assert candidate['provider'] == 'wikipedia'
    assert candidate['summary'] == '팀 스포츠 경기 규칙'
    assert candidate['reuse_status'] == 'reference_only_not_code_or_asset'
    assert candidate['preview_urls'] == []
    assert page['download_performed'] is False
    assert len(requests) == 1

def test_openalex_search_is_cursor_bound_research_evidence_only() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == 'api.openalex.org'
        assert request.url.path == '/works'
        cursor = request.url.params['cursor']
        return httpx.Response(200, json={'meta': {'count': 2, 'next_cursor': 'next-page' if cursor == '*' else None}, 'results': [{'id': 'https://openalex.org/W123', 'doi': 'https://doi.org/10.1000/example', 'display_name': 'Execution feedback for game agents', 'publication_year': 2026, 'type': 'article', 'cited_by_count': 7, 'authorships': [{'author': {'display_name': 'Ada Researcher'}}], 'primary_location': {}, 'best_oa_location': {'license': 'cc-by'}, 'open_access': {'is_oa': True}}]})
    client = _client(handler)
    first = client.search('openalex_works', 'game agent execution feedback', limit=1, target_profile='general_reference')
    second = client.search('openalex_works', 'game agent execution feedback', cursor=first['next_cursor'], limit=1, target_profile='general_reference')
    assert requests[0].url.params['cursor'] == '*'
    assert requests[1].url.params['cursor'] == 'next-page'
    assert second['next_cursor'] == ''
    candidate = first['candidates'][0]
    assert candidate['provider'] == 'openalex_works'
    assert candidate['reuse_status'] == 'research_evidence_only_not_implementation_authority'
    assert candidate['metadata']['doi'] == '10.1000/example'

def test_crossref_search_cross_checks_paper_metadata_without_full_text_reuse() -> None:

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == 'api.crossref.org'
        assert request.url.params['cursor'] == '*'
        return httpx.Response(200, json={'message': {'total-results': 1, 'next-cursor': None, 'items': [{'DOI': '10.5555/agent-test', 'title': ['Repository agents with executable tests'], 'abstract': '<jats:p>Fresh evidence matters.</jats:p>', 'type': 'proceedings-article', 'published': {'date-parts': [[2025, 7, 2]]}, 'author': [{'given': 'Grace', 'family': 'Tester'}], 'publisher': 'Example Society', 'is-referenced-by-count': 3, 'license': [{'URL': 'https://creativecommons.org/licenses/by/4.0/'}]}]}})
    page = _client(handler).search('crossref_works', 'repository agents executable tests', target_profile='general_reference')
    candidate = page['candidates'][0]
    assert candidate['source_url'] == 'https://doi.org/10.5555/agent-test'
    assert candidate['metadata']['abstract'] == 'Fresh evidence matters.'
    assert candidate['metadata']['published'] == '2025-07-02'
    assert candidate['license_policy'].startswith('bibliographic_metadata_only')

def test_seed_bundle_pages_a_large_route_catalog_without_a_global_cap(monkeypatch) -> None:

    class Discovery:

        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, provider, query, **kwargs):
            self.queries.append(query)
            return {'schema_version': 'mmm/ecosystem-discovery-page-v1', 'provider': provider, 'query': query, 'returned': 1, 'provider_total_estimate': 1, 'next_cursor': '', 'page_sha256': 'a' * 64, 'candidates': []}
    brief = {'domains': [{'domain_id': f'system_{index:03d}', 'objective': f'Research system {index}', 'requirements': [f'Implement system {index}'], 'evidence_kinds': ['source_code'], 'queries': [f'unique system {index}'], 'providers': ['github'], 'depends_on': []} for index in range(40)]}
    discovery = Discovery()
    monkeypatch.setenv('MMM_ECOSYSTEM_DISCOVERY', 'auto')
    first = discover_seed_bundle('Build every requested system', {'title': 'Large'}, research_brief=brief, client=discovery, route_limit=7)
    second = discover_seed_bundle('Build every requested system', {'title': 'Large'}, research_brief=brief, client=discovery, route_cursor=first['next_route_cursor'], route_limit=7)
    assert first['route_count'] == 40
    assert first['processed_route_count'] == 7
    assert first['remaining_route_count'] == 33
    assert first['routes_complete'] is False
    assert second['route_offset'] == 7
    assert second['processed_route_count'] == 7
    assert len(discovery.queries) == 14
    with pytest.raises(SpecValidationError, match='route catalog and page size'):
        discover_seed_bundle('Build every requested system', {'title': 'Large'}, research_brief=brief, client=discovery, route_cursor=first['next_route_cursor'], route_limit=8)
