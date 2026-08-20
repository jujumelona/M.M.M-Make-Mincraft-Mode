from __future__ import annotations
import base64
import binascii
import hashlib
import html
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping
from urllib.parse import parse_qs, quote, urlparse
import httpx
from .central_research import external_discovery_routes
from .platform_catalog import adapter_for_target
from .spec import SpecValidationError, canonical_json
_PROVIDERS = frozenset({'modrinth', 'github', 'openverse_images', 'wikipedia', 'huggingface_models', 'openalex_works', 'crossref_works'})
_TARGET_PROFILES = frozenset({'minecraft_mod', 'ai_runtime', 'media', 'general_reference'})
_MODRINTH_ID = re.compile('^[A-Za-z0-9_-]{3,64}$')
_SHA1_HEX = re.compile('^[0-9a-f]{40}$')
_SHA512_HEX = re.compile('^[0-9a-f]{128}$')
_GITHUB_REPOSITORY = re.compile('^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$')
_GIT_COMMIT = re.compile('^[0-9a-f]{40,64}$')
_CURSOR = re.compile('^(offset|page):(\\d+):([0-9a-f]{16})$')
_TOKEN_CURSOR = re.compile('^token:([A-Za-z0-9_-]{1,2048}):([0-9a-f]{16})$')
_SEED_ROUTE_CURSOR = re.compile('^routes:(\\d+):([0-9a-f]{16})$')
_HF_REVISION = re.compile('^[0-9a-f]{40,64}$')
_HF_MODEL_COMPONENT = re.compile('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
_MAX_QUERY_BYTES = 16 * 1024
_MAX_PAGE_ITEMS = 100
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_USER_AGENT = 'M.M.M-Make-Mincraft-Mode/0.8.0 (https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode)'
_PERMISSIVE_CODE_LICENSES = frozenset({'MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'ISC', 'Zlib', 'Unlicense', 'CC0-1.0'})
_REVIEWABLE_MODEL_LICENSES = frozenset({'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 'cc-by-2.0', 'cc-by-3.0', 'cc-by-4.0', 'cc-by-sa-3.0', 'cc-by-sa-4.0', 'cc0-1.0', 'mit', 'mpl-2.0', 'unlicense'})
_RESTRICTED_MODEL_LICENSES = frozenset({'bigscience-openrail-m', 'creativeml-openrail-m', 'gemma', 'llama2', 'llama3', 'llama3.1', 'openrail', 'openrail++', 'research-only'})
_SAFE_MODEL_SUFFIXES = frozenset({'.safetensors', '.gguf', '.onnx'})
_UNSAFE_SERIALIZATION_SUFFIXES = frozenset({'.bin', '.ckpt', '.joblib', '.pickle', '.pkl', '.pt', '.pth'})

class EcosystemDiscoveryUnavailable(RuntimeError):
    pass

@dataclass(frozen=True)
class EcosystemCandidate:
    candidate_id: str
    provider: str
    resource_kind: str
    title: str
    summary: str
    source_url: str
    api_url: str
    license_id: str
    license_url: str
    license_policy: str
    minecraft_version: str
    loader: str
    compatibility: str
    attribution: str
    preview_urls: tuple[str, ...]
    reuse_status: str
    evidence_sha256: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {**asdict(self), 'preview_urls': list(self.preview_urls)}
        if self.metadata is None:
            payload.pop('metadata')
        return payload

@dataclass(frozen=True)
class _DiscoveryTarget:
    minecraft_version: str
    loader: str
    exact: bool

def _normalize_discovery_target(minecraft_version: str | None, loader: str | None, *, target_profile: str, exact_required: bool=False) -> _DiscoveryTarget:
    if target_profile != 'minecraft_mod':
        return _DiscoveryTarget('not_applicable', 'not_applicable', False)
    version = str(minecraft_version or '').strip()
    loader_name = str(loader or '').strip().casefold()
    if bool(version) != bool(loader_name):
        raise SpecValidationError('Minecraft ecosystem target requires both minecraft_version and loader.')
    if not version:
        if exact_required:
            raise SpecValidationError('Exact ecosystem inspection requires the host-selected Minecraft target.')
        return _DiscoveryTarget('unresolved', 'unresolved', False)
    try:
        adapter = adapter_for_target(version, loader_name)
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc
    return _DiscoveryTarget(adapter.minecraft_version, adapter.loader, True)

class EcosystemDiscoveryClient:
    """Read-only paginated discovery over reviewed public ecosystem APIs.

    Targetless Minecraft searches are intentionally shallow and platform-neutral.
    Exact compatibility claims and project inspection are available only after the
    host has selected an executable provider target.
    """

    def __init__(self, *, transport: httpx.BaseTransport | None=None, timeout_seconds: float=12.0, github_token: str | None=None, openverse_token: str | None=None) -> None:
        if timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be positive.')
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN', '')
        self.openverse_token = openverse_token or os.environ.get('MMM_OPENVERSE_TOKEN', '')

    def search(self, provider: str, query: str, *, cursor: str='', limit: int=20, minecraft_version: str | None=None, loader: str | None=None, target_profile: str='minecraft_mod') -> dict[str, Any]:
        provider = provider.strip().lower()
        query = query.strip()
        target_profile = target_profile.strip().lower()
        if provider not in _PROVIDERS:
            raise SpecValidationError(f'Unsupported ecosystem discovery provider: {provider!r}')
        if target_profile not in _TARGET_PROFILES:
            raise SpecValidationError(f'Unsupported ecosystem discovery target profile: {target_profile!r}')
        if not query or len(query.encode('utf-8')) > _MAX_QUERY_BYTES:
            raise SpecValidationError('Discovery query must be non-empty and within the query byte policy.')
        if type(limit) is not int or not 1 <= limit <= _MAX_PAGE_ITEMS:
            raise SpecValidationError(f'limit must be between 1 and {_MAX_PAGE_ITEMS}.')
        target = _normalize_discovery_target(minecraft_version, loader, target_profile=target_profile)
        if provider in {'huggingface_models', 'openalex_works', 'crossref_works'}:
            position_kind = 'token'
            provider_cursor = _decode_token_cursor(cursor, provider=provider, query=query, target_profile=target_profile, minecraft_version=target.minecraft_version, loader=target.loader)
        else:
            position_kind = 'offset' if provider in {'modrinth', 'wikipedia'} else 'page'
            position = _decode_cursor(cursor, provider=provider, query=query, position_kind=position_kind, target_profile=target_profile, minecraft_version=target.minecraft_version, loader=target.loader)
        if provider == 'huggingface_models':
            candidates, total, next_position = self._search_huggingface_models(query, cursor=provider_cursor, limit=limit)
        elif provider == 'modrinth':
            candidates, total, next_position = self._search_modrinth(query, offset=position, limit=limit, target=target)
        elif provider == 'github':
            candidates, total, next_position = self._search_github(query, page=position or 1, limit=limit, target_profile=target_profile, target=target)
        elif provider == 'openverse_images':
            candidates, total, next_position = self._search_openverse(query, media_type='images', page=position or 1, limit=limit)
        elif provider == 'wikipedia':
            candidates, total, next_position = self._search_wikipedia(query, offset=position, limit=limit)
        elif provider == 'openalex_works':
            candidates, total, next_position = self._search_openalex_works(query, cursor=provider_cursor, limit=limit)
        else:
            candidates, total, next_position = self._search_crossref_works(query, cursor=provider_cursor, limit=limit)
        if provider in {'huggingface_models', 'openalex_works', 'crossref_works'}:
            next_cursor = _encode_token_cursor(provider=provider, query=query, target_profile=target_profile, token=next_position, minecraft_version=target.minecraft_version, loader=target.loader) if next_position is not None else ''
        else:
            next_cursor = _encode_cursor(provider=provider, query=query, position_kind=position_kind, position=next_position, target_profile=target_profile, minecraft_version=target.minecraft_version, loader=target.loader) if next_position is not None else ''
        payload = {'schema_version': 'mmm/ecosystem-discovery-page-v2', 'provider': provider, 'query': query, 'query_sha256': _sha256_text(query), 'minecraft_version': target.minecraft_version, 'loader': target.loader, 'target_exact': target.exact, 'target_profile': target_profile, 'candidates': [candidate.to_dict() for candidate in candidates], 'returned': len(candidates), 'provider_total_estimate': total, 'provider_truncated': bool(provider == 'github' and isinstance(total, int) and (total > 1000)), 'provider_result_limit': 1000 if provider == 'github' else None, 'next_cursor': next_cursor, 'download_performed': False, 'authorization': 'none', 'selection_policy': 'Targetless results are metadata hypotheses only. Exact-target candidates are still not selected or downloaded until origin-license, dependency closure and immutable-file-hash gates pass.'}
        payload['page_sha256'] = _sha256_text(canonical_json(payload))
        return payload

    def inspect_modrinth_project(self, project_id: str, *, minecraft_version: str | None=None, loader: str | None=None) -> dict[str, Any]:
        if not _MODRINTH_ID.fullmatch(project_id):
            raise SpecValidationError('Invalid Modrinth project ID or slug.')
        target = _normalize_discovery_target(minecraft_version, loader, target_profile='minecraft_mod', exact_required=True)
        project_url = f'https://api.modrinth.com/v2/project/{project_id}'
        version_url = project_url + '/version'
        project = self._get_json(project_url)
        versions = self._get_json(version_url, params={'loaders': json.dumps([target.loader]), 'game_versions': json.dumps([target.minecraft_version]), 'include_changelog': 'false'})
        if not isinstance(project, dict) or not isinstance(versions, list):
            raise EcosystemDiscoveryUnavailable('Modrinth returned an invalid project inspection response.')
        license_value = project.get('license')
        if isinstance(license_value, dict):
            license_id = str(license_value.get('id', '')).strip()
            license_url = str(license_value.get('url', '') or '').strip()
        else:
            license_id = str(license_value or '').strip()
            license_url = ''
        normalized_versions = [_normalize_modrinth_version(version, minecraft_version=target.minecraft_version, loader=target.loader) for version in versions if isinstance(version, dict)]
        compatible = [version for version in normalized_versions if version['eligible_for_selection']]
        payload = {'schema_version': 'mmm/modrinth-inspection-v2', 'project_id': str(project.get('id', project_id)), 'slug': str(project.get('slug', project_id)), 'title': str(project.get('title', '')), 'license_id': license_id, 'license_url': _safe_https_url(license_url, allow_empty=True), 'license_policy': _code_license_policy(license_id), 'minecraft_version': target.minecraft_version, 'loader': target.loader, 'versions': normalized_versions, 'exact_compatible_version_found': bool(compatible), 'eligible_version_ids': [version['version_id'] for version in compatible], 'compatibility_gate': 'candidate_requires_dependency_closure_and_verified_download' if compatible else 'blocked_no_exact_version_with_one_primary_strong_digest_file', 'download_performed': False, 'required_next_gate': 'Select one listed exact-target version, resolve required/incompatible dependencies, download only after approval, and verify the advertised SHA-512.'}
        payload['inspection_sha256'] = _sha256_text(canonical_json(payload))
        return payload

    def inspect_github_repository(self, full_name: str) -> dict[str, Any]:
        if not _GITHUB_REPOSITORY.fullmatch(full_name):
            raise SpecValidationError('Invalid GitHub owner/repository name.')
        repository_url = f'https://api.github.com/repos/{full_name}'
        repository = self._get_json(repository_url, provider='github')
        if not isinstance(repository, dict):
            raise EcosystemDiscoveryUnavailable('GitHub returned an invalid repository response.')
        if repository.get('private') or repository.get('archived'):
            raise EcosystemDiscoveryUnavailable('GitHub repository is private or archived.')
        default_branch = str(repository.get('default_branch') or '').strip()
        if not default_branch or len(default_branch.encode('utf-8')) > 512:
            raise EcosystemDiscoveryUnavailable('GitHub repository has no usable default branch.')
        commit = self._get_json(repository_url + '/commits/' + quote(default_branch, safe=''), provider='github')
        if not isinstance(commit, dict):
            raise EcosystemDiscoveryUnavailable('GitHub returned invalid commit evidence.')
        commit_sha = str(commit.get('sha') or '').strip().lower()
        if not _GIT_COMMIT.fullmatch(commit_sha):
            raise EcosystemDiscoveryUnavailable('GitHub did not return an immutable commit SHA.')
        license_data = self._get_json(repository_url + '/license', params={'ref': commit_sha}, provider='github')
        if not isinstance(license_data, dict):
            raise EcosystemDiscoveryUnavailable('GitHub returned invalid commit or license evidence.')
        license_value = license_data.get('license')
        license_id = str(license_value.get('spdx_id') or '').strip() if isinstance(license_value, dict) else ''
        if not license_id or license_id == 'NOASSERTION':
            raise EcosystemDiscoveryUnavailable('GitHub did not detect a reviewable SPDX license.')
        encoded_license = str(license_data.get('content') or '').replace('\n', '')
        if license_data.get('encoding') != 'base64' or not encoded_license:
            raise EcosystemDiscoveryUnavailable('GitHub license evidence has no base64 content.')
        try:
            license_bytes = base64.b64decode(encoded_license, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise EcosystemDiscoveryUnavailable('GitHub license evidence is invalid base64.') from exc
        if not license_bytes or len(license_bytes) > _MAX_RESPONSE_BYTES:
            raise EcosystemDiscoveryUnavailable('GitHub license text exceeds the byte policy.')
        verification = commit.get('commit')
        verification = verification.get('verification') if isinstance(verification, dict) else None
        source_url = _safe_https_url(repository.get('html_url'))
        if urlparse(source_url).hostname != 'github.com':
            raise EcosystemDiscoveryUnavailable('GitHub repository origin URL is invalid.')
        license_path = str(license_data.get('path') or 'LICENSE').replace('\\', '/')
        if not license_path or license_path.startswith('/') or '..' in license_path.split('/'):
            raise EcosystemDiscoveryUnavailable('GitHub license path is invalid.')
        payload = {'schema_version': 'mmm/github-repository-inspection-v1', 'full_name': str(repository.get('full_name') or full_name), 'source_url': source_url, 'default_branch': default_branch, 'commit_sha': commit_sha, 'commit_verified': bool(isinstance(verification, dict) and verification.get('verified') is True), 'license_id': license_id, 'license_url': f'https://github.com/{full_name}/blob/{commit_sha}/' + quote(license_path, safe='/_.-'), 'license_blob_sha': str(license_data.get('sha') or ''), 'license_text_sha256': 'sha256:' + hashlib.sha256(license_bytes).hexdigest(), 'license_policy': _code_license_policy(license_id), 'download_performed': False, 'compatibility': 'unverified_until_pinned_metadata_and_build_pass', 'required_next_gate': 'Read platform metadata and build files at this commit, bind an exact host-selected target, resolve dependencies, then verify any selected release file with its own immutable SHA-256 or stronger digest.'}
        payload['inspection_sha256'] = _sha256_text(canonical_json(payload))
        return payload

    def inspect_huggingface_model(self, repo_id: str) -> dict[str, Any]:
        normalized_repo_id = _normalize_hf_repo_id(repo_id)
        if not normalized_repo_id:
            raise SpecValidationError('Invalid Hugging Face model repository ID.')
        encoded_repo_id = '/'.join((quote(part, safe='._-') for part in normalized_repo_id.split('/')))
        model_url = f'https://huggingface.co/api/models/{encoded_repo_id}'
        current = self._get_json(model_url, params={'full': 'true', 'cardData': 'true'}, provider='huggingface')
        normalized_current = _normalize_hf_model(current)
        if normalized_current is None:
            raise EcosystemDiscoveryUnavailable('Hugging Face returned model metadata without an immutable revision.')
        if normalized_current['model_id'].lower() != normalized_repo_id.lower():
            raise EcosystemDiscoveryUnavailable('Hugging Face returned metadata for a different model repository.')
        revision_sha = normalized_current['revision_sha']
        pinned_url = model_url + '/revision/' + quote(revision_sha, safe='')
        pinned = self._get_json(pinned_url, params={'full': 'true', 'cardData': 'true', 'blobs': 'true'}, provider='huggingface')
        normalized = _normalize_hf_model(pinned)
        if normalized is None or normalized['model_id'].lower() != normalized_repo_id.lower() or normalized['revision_sha'] != revision_sha:
            raise EcosystemDiscoveryUnavailable('Hugging Face did not return matching metadata for the pinned revision.')
        files = normalized.pop('files')
        card = normalized.pop('card')
        model_license_policy = _model_license_policy(card['license_id'])
        access_blocked = bool(normalized['private'] or normalized['gated'] or normalized['disabled'])
        license_blocked = model_license_policy.startswith('reject_')
        format_inventory = _hf_format_inventory(files)
        payload = {'schema_version': 'mmm/huggingface-model-inspection-v1', 'model_id': normalized_repo_id, 'source_url': f'https://huggingface.co/{encoded_repo_id}/tree/{revision_sha}', 'api_url': pinned_url, 'revision_sha': revision_sha, 'model': normalized, 'card': card, 'files': files, 'format_inventory': format_inventory, 'gates': {'access': {'status': 'blocked' if access_blocked else 'metadata_only', 'reason': 'private, gated or disabled model metadata' if access_blocked else 'public metadata does not authorize an artifact download'}, 'model_license': {'status': 'blocked_unresolved' if license_blocked else 'manual_review_required', 'license_id': card['license_id'], 'license_url': card['license_url'], 'policy': model_license_policy}, 'code_license': {'status': 'unresolved_separate_review_required', 'reason': 'The model artifact license does not license the runtime, client library, sidecar or generated integration code.'}, 'dataset_provenance': {'status': 'manual_review_required', 'declared_datasets': card['datasets'], 'reason': 'Card declarations are claims, not proof of training-data rights, consent, quality or permitted downstream use.'}, 'runtime_compatibility': {'status': 'unverified', 'reason': 'Benchmark the exact revision against the host-selected executable Minecraft target, runtime, hardware, latency and memory budgets.'}}, 'selected': False, 'download_performed': False, 'weights_downloaded': False, 'code_executed': False, 'required_next_gates': ['Resolve access and the exact model-artifact license terms.', 'Verify licenses for runtime and integration code separately.', 'Benchmark the pinned revision in the intended local, sidecar or remote topology.', 'Approve an artifact and verify its immutable digest before any download.']}
        payload['inspection_sha256'] = _sha256_text(canonical_json(payload))
        return payload

    def _search_modrinth(self, query: str, *, offset: int, limit: int, target: _DiscoveryTarget) -> tuple[list[EcosystemCandidate], int, int | None]:
        facets: list[list[str]] = []
        if target.exact:
            facets.extend([[f'versions:{target.minecraft_version}'], [f'categories:{target.loader}']])
        facets.extend([['project_type:mod'], ['open_source:true']])
        raw = self._get_json('https://api.modrinth.com/v2/search', params={'query': query, 'facets': json.dumps(facets, separators=(',', ':')), 'index': 'relevance', 'offset': str(offset), 'limit': str(limit)})
        if not isinstance(raw, dict) or not isinstance(raw.get('hits'), list):
            raise EcosystemDiscoveryUnavailable('Modrinth returned an invalid search response.')
        total = _nonnegative_int(raw.get('total_hits'))
        candidates: list[EcosystemCandidate] = []
        for hit in raw['hits']:
            if not isinstance(hit, dict):
                continue
            project_id = str(hit.get('project_id', '')).strip()
            slug = str(hit.get('slug', '')).strip()
            license_id = str(hit.get('license', '')).strip()
            if not project_id or not slug or (not license_id):
                continue
            project_type = str(hit.get('project_type', 'mod'))
            stable = {'project_id': project_id, 'slug': slug, 'title': str(hit.get('title', '')), 'description': str(hit.get('description', '')), 'license': license_id, 'versions': sorted((str(v) for v in hit.get('versions', []))), 'categories': sorted((str(v) for v in hit.get('categories', [])))}
            candidates.append(EcosystemCandidate(candidate_id=f'modrinth:{project_id}', provider='modrinth', resource_kind=project_type, title=stable['title'], summary=stable['description'], source_url=f'https://modrinth.com/{project_type}/{slug}', api_url=f'https://api.modrinth.com/v2/project/{project_id}', license_id=license_id, license_url='', license_policy=_code_license_policy(license_id), minecraft_version=target.minecraft_version, loader=target.loader, compatibility='search_metadata_exact; version_file_inspection_required' if target.exact else 'platform_neutral_metadata; target_hypothesis_required', attribution='', preview_urls=_safe_preview_urls([hit.get('icon_url'), *hit.get('gallery', [])], allowed_hosts={'cdn.modrinth.com'}), reuse_status='candidate_only_not_downloaded', evidence_sha256=_sha256_text(canonical_json(stable))))
        next_offset = offset + limit if offset + limit < total else None
        return (candidates, total, next_offset)

    def _search_github(self, query: str, *, page: int, limit: int, target_profile: str, target: _DiscoveryTarget) -> tuple[list[EcosystemCandidate], int, int | None]:
        scoped_query = query
        if target_profile == 'minecraft_mod':
            scoped_query += f' minecraft {target.loader}' if target.exact else ' minecraft mod'
        raw = self._get_json('https://api.github.com/search/repositories', params={'q': f'{scoped_query} in:name,description,readme fork:false archived:false', 'sort': 'stars', 'order': 'desc', 'page': str(page), 'per_page': str(limit)}, provider='github')
        if not isinstance(raw, dict) or not isinstance(raw.get('items'), list):
            raise EcosystemDiscoveryUnavailable('GitHub returned an invalid repository search response.')
        total = _nonnegative_int(raw.get('total_count'))
        candidates: list[EcosystemCandidate] = []
        for item in raw['items']:
            if not isinstance(item, dict) or item.get('archived'):
                continue
            license_value = item.get('license')
            license_id = str(license_value.get('spdx_id', '')).strip() if isinstance(license_value, dict) else ''
            if not license_id or license_id == 'NOASSERTION':
                continue
            full_name = str(item.get('full_name', '')).strip()
            source_url = _safe_https_url(item.get('html_url'))
            if not full_name or urlparse(source_url).hostname != 'github.com':
                continue
            stable = {'id': item.get('id'), 'full_name': full_name, 'description': str(item.get('description') or ''), 'default_branch': str(item.get('default_branch', '')), 'license': license_id, 'topics': sorted((str(value) for value in item.get('topics', [])))}
            is_minecraft = target_profile == 'minecraft_mod'
            candidates.append(EcosystemCandidate(candidate_id=f'github:{full_name.lower()}', provider='github', resource_kind='repository', title=full_name, summary=stable['description'], source_url=source_url, api_url=f'https://api.github.com/repos/{full_name}', license_id=license_id, license_url=f'https://api.github.com/repos/{full_name}/license', license_policy=_code_license_policy(license_id), minecraft_version=target.minecraft_version if is_minecraft else 'not_applicable', loader=target.loader if is_minecraft else 'not_applicable', compatibility='unverified; exact commit, platform metadata and build inspection required' if is_minecraft and target.exact else 'platform_neutral_metadata; target_hypothesis_required' if is_minecraft else 'unverified; pin commit, inspect license and benchmark runtime', attribution='', preview_urls=(), reuse_status='candidate_only_not_cloned', evidence_sha256=_sha256_text(canonical_json(stable))))
        next_page = page + 1 if page * limit < min(total, 1000) else None
        return (candidates, total, next_page)

    def _search_huggingface_models(self, query: str, *, cursor: str, limit: int) -> tuple[list[EcosystemCandidate], None, str | None]:
        params = {'search': query, 'limit': str(limit), 'full': 'true', 'cardData': 'true', 'sort': 'lastModified', 'direction': '-1'}
        if cursor:
            params['cursor'] = cursor
        raw, next_url = self._get_json('https://huggingface.co/api/models', params=params, provider='huggingface', include_next_url=True)
        if not isinstance(raw, list):
            raise EcosystemDiscoveryUnavailable('Hugging Face returned an invalid model search response.')
        candidates: list[EcosystemCandidate] = []
        for item in raw:
            normalized = _normalize_hf_model(item)
            if normalized is None:
                continue
            files = normalized.pop('files')
            card = normalized.pop('card')
            license_policy = _model_license_policy(card['license_id'])
            access_blocked = bool(normalized['private'] or normalized['gated'] or normalized['disabled'])
            license_blocked = license_policy.startswith('reject_')
            model_id = normalized['model_id']
            encoded_model_id = '/'.join((quote(part, safe='._-') for part in model_id.split('/')))
            metadata = {**normalized, 'card': card, 'file_count': len(files), 'format_inventory': _hf_format_inventory(files)}
            stable = {'model': metadata, 'file_paths': [file['path'] for file in files]}
            if access_blocked:
                reuse_status = 'blocked_gated_private_or_disabled'
            elif license_blocked:
                reuse_status = 'rejected_until_model_license_is_verified'
            else:
                reuse_status = 'candidate_only_metadata_not_weights'
            pipeline_tag = normalized['pipeline_tag'] or 'unspecified pipeline'
            library_name = normalized['library_name'] or 'unspecified runtime'
            candidates.append(EcosystemCandidate(candidate_id=f'huggingface:{model_id.lower()}', provider='huggingface_models', resource_kind='ai_model', title=model_id, summary=f'{pipeline_tag} model metadata for {library_name}; exact runtime, performance, provenance and license remain unverified.', source_url=f'https://huggingface.co/{encoded_model_id}', api_url=f'https://huggingface.co/api/models/{encoded_model_id}', license_id=card['license_id'], license_url=card['license_url'], license_policy=license_policy, minecraft_version='not_applicable', loader='not_applicable', compatibility='unverified; exact runtime, hardware, latency, memory and Minecraft integration benchmark required', attribution='Hugging Face Hub model-card metadata; author claims require review', preview_urls=(), reuse_status=reuse_status, evidence_sha256=_sha256_text(canonical_json(stable)), metadata=metadata))
        return (candidates, None, _huggingface_cursor_from_next_url(next_url))

    def _search_openalex_works(self, query: str, *, cursor: str, limit: int) -> tuple[list[EcosystemCandidate], int, str | None]:
        raw = self._get_json('https://api.openalex.org/works', params={'search': query, 'per-page': str(limit), 'cursor': cursor or '*', 'select': 'id,doi,display_name,publication_year,type,cited_by_count,authorships,primary_location,best_oa_location,open_access'}, provider='openalex')
        results = raw.get('results') if isinstance(raw, dict) else None
        meta = raw.get('meta') if isinstance(raw, dict) else None
        if not isinstance(results, list) or not isinstance(meta, dict):
            raise EcosystemDiscoveryUnavailable('OpenAlex returned an invalid works search response.')
        candidates: list[EcosystemCandidate] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            openalex_id = str(item.get('id') or '').strip()
            work_key = openalex_id.rsplit('/', 1)[-1]
            title = ' '.join(str(item.get('display_name') or '').split())
            if not re.fullmatch('W\\d+', work_key) or not title:
                continue
            doi = _normalized_doi(item.get('doi'))
            source_url = f"https://doi.org/{quote(doi, safe='/():._-')}" if doi else f'https://openalex.org/{work_key}'
            authors = _openalex_authors(item.get('authorships'))
            location = item.get('best_oa_location')
            if not isinstance(location, dict):
                location = item.get('primary_location')
            location = location if isinstance(location, dict) else {}
            paper_license = str(location.get('license') or '').strip()
            metadata = {'openalex_id': work_key, 'doi': doi, 'publication_year': _nonnegative_int(item.get('publication_year')), 'work_type': str(item.get('type') or ''), 'cited_by_count': _nonnegative_int(item.get('cited_by_count')), 'authors': authors, 'open_access': dict(item.get('open_access')) if isinstance(item.get('open_access'), dict) else {}, 'asserted_paper_license': paper_license}
            stable = {'title': title, **metadata}
            candidates.append(EcosystemCandidate(candidate_id=f'openalex:{work_key.lower()}', provider='openalex_works', resource_kind='scholarly_work', title=title, summary=_scholarly_summary(metadata), source_url=source_url, api_url=f'https://api.openalex.org/works/{work_key}', license_id=paper_license or 'paper-license-unverified', license_url='', license_policy='bibliographic_metadata_only; inspect the paper license before full-text reuse', minecraft_version='not_applicable', loader='not_applicable', compatibility='research candidate only; translate claims to the host-selected Minecraft target and reproduce relevant measurements', attribution='OpenAlex bibliographic metadata', preview_urls=(), reuse_status='research_evidence_only_not_implementation_authority', evidence_sha256=_sha256_text(canonical_json(stable)), metadata=metadata))
        next_cursor = str(meta.get('next_cursor') or '').strip() or None
        return (candidates, _nonnegative_int(meta.get('count')), next_cursor)

    def _search_crossref_works(self, query: str, *, cursor: str, limit: int) -> tuple[list[EcosystemCandidate], int, str | None]:
        raw = self._get_json('https://api.crossref.org/works', params={'query.bibliographic': query, 'rows': str(limit), 'cursor': cursor or '*', 'select': 'DOI,title,abstract,type,published,author,publisher,URL,is-referenced-by-count,license'}, provider='crossref')
        message = raw.get('message') if isinstance(raw, dict) else None
        items = message.get('items') if isinstance(message, dict) else None
        if not isinstance(message, dict) or not isinstance(items, list):
            raise EcosystemDiscoveryUnavailable('Crossref returned an invalid works search response.')
        candidates: list[EcosystemCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            doi = _normalized_doi(item.get('DOI'))
            raw_title = item.get('title')
            title = ' '.join(str(raw_title[0]).split()) if isinstance(raw_title, list) and raw_title else ''
            if not doi or not title:
                continue
            license_items = item.get('license')
            license_url = ''
            if isinstance(license_items, list):
                for license_item in license_items:
                    if isinstance(license_item, dict):
                        license_url = _safe_https_url(license_item.get('URL'), allow_empty=True)
                        if license_url:
                            break
            metadata = {'doi': doi, 'published': _crossref_date(item.get('published')), 'work_type': str(item.get('type') or ''), 'publisher': str(item.get('publisher') or ''), 'cited_by_count': _nonnegative_int(item.get('is-referenced-by-count')), 'authors': _crossref_authors(item.get('author')), 'abstract': _plain_bounded_text(item.get('abstract'), 1200)}
            stable = {'title': title, 'license_url': license_url, **metadata}
            encoded_doi = quote(doi, safe='/():._-')
            candidates.append(EcosystemCandidate(candidate_id='crossref:' + _sha256_text(doi).split(':', 1)[1], provider='crossref_works', resource_kind='scholarly_work', title=title, summary=_scholarly_summary(metadata), source_url=f'https://doi.org/{encoded_doi}', api_url=f'https://api.crossref.org/works/{encoded_doi}', license_id='paper-license-link-present' if license_url else 'paper-license-unverified', license_url=license_url, license_policy='bibliographic_metadata_only; inspect the paper license before full-text reuse', minecraft_version='not_applicable', loader='not_applicable', compatibility='research candidate only; translate claims to the host-selected Minecraft target and reproduce relevant measurements', attribution='Crossref bibliographic metadata', preview_urls=(), reuse_status='research_evidence_only_not_implementation_authority', evidence_sha256=_sha256_text(canonical_json(stable)), metadata=metadata))
        return (candidates, _nonnegative_int(message.get('total-results')), str(message.get('next-cursor') or '').strip() or None)

    def _search_openverse(self, query: str, *, media_type: str, page: int, limit: int) -> tuple[list[EcosystemCandidate], int, int | None]:
        raw = self._get_json(f'https://api.openverse.org/v1/{media_type}/', params={'q': query, 'license': 'cc0,pdm,by,by-sa', 'license_type': 'modification', 'mature': 'false', 'page': str(page), 'page_size': str(limit)}, provider='openverse')
        if not isinstance(raw, dict) or not isinstance(raw.get('results'), list):
            raise EcosystemDiscoveryUnavailable('Openverse returned an invalid media search response.')
        total = _nonnegative_int(raw.get('result_count'))
        candidates: list[EcosystemCandidate] = []
        for item in raw['results']:
            if not isinstance(item, dict) or item.get('mature') is True:
                continue
            identifier = str(item.get('id', '')).strip()
            license_name = str(item.get('license', '')).strip().lower()
            license_version = str(item.get('license_version', '')).strip()
            if not identifier or license_name not in {'cc0', 'pdm', 'by', 'by-sa'}:
                continue
            license_id = _creative_commons_id(license_name, license_version)
            source_url = _safe_https_url(item.get('foreign_landing_url') or item.get('detail_url'))
            if not source_url:
                continue
            title = str(item.get('title') or 'Untitled').strip() or 'Untitled'
            creator = str(item.get('creator') or '').strip()
            attribution = str(item.get('attribution') or '').strip()
            if not attribution:
                attribution = f'{title} by {creator}, {license_id}' if creator else f'{title}, {license_id}'
            stable = {'id': identifier, 'title': title, 'creator': creator, 'source_url': source_url, 'license_id': license_id, 'provider': str(item.get('provider') or ''), 'source': str(item.get('source') or '')}
            candidates.append(EcosystemCandidate(candidate_id=f'openverse:{identifier}', provider='openverse_images', resource_kind='image_asset', title=title, summary=f'Openverse image metadata by {creator}.' if creator else 'Openverse image metadata.', source_url=source_url, api_url=_safe_https_url(item.get('detail_url'), allow_empty=True), license_id=license_id, license_url=_safe_https_url(item.get('license_url'), allow_empty=True), license_policy=_media_license_policy(license_name), minecraft_version='not_applicable', loader='not_applicable', compatibility='visual_reference_only; verify origin, dimensions, and exact license before reuse', attribution=attribution, preview_urls=_safe_preview_urls([item.get('thumbnail')], allowed_hosts=None), reuse_status='origin_license_verification_required', evidence_sha256=_sha256_text(canonical_json(stable)), metadata={'creator': creator, 'provider': stable['provider'], 'source': stable['source']}))
        page_count = _nonnegative_int(raw.get('page_count'))
        return (candidates, total, page + 1 if page_count and page < page_count else None)

    def _search_wikipedia(self, query: str, *, offset: int, limit: int) -> tuple[list[EcosystemCandidate], int, int | None]:
        language = 'ko' if re.search('[가-힣]', query) else 'en'
        host = f'{language}.wikipedia.org'
        url = f'https://{host}/w/api.php'
        raw = self._get_json(url, params={'action': 'query', 'list': 'search', 'srsearch': query, 'srnamespace': '0', 'sroffset': str(offset), 'srlimit': str(limit), 'format': 'json', 'formatversion': '2'})
        query_data = raw.get('query') if isinstance(raw, dict) else None
        search = query_data.get('search') if isinstance(query_data, dict) else None
        if not isinstance(search, list):
            raise EcosystemDiscoveryUnavailable('Wikipedia returned an invalid search response.')
        search_info = query_data.get('searchinfo')
        total = _nonnegative_int(search_info.get('totalhits')) if isinstance(search_info, dict) else 0
        candidates: list[EcosystemCandidate] = []
        for item in search:
            if not isinstance(item, dict):
                continue
            page_id = _nonnegative_int(item.get('pageid'))
            title = str(item.get('title') or '').strip()
            if not page_id or not title:
                continue
            source_url = f'https://{host}/wiki/' + quote(title.replace(' ', '_'), safe='()_-.')
            snippet = html.unescape(re.sub('<[^>]*>', ' ', str(item.get('snippet') or '')))
            snippet = ' '.join(snippet.split())
            stable = {'language': language, 'page_id': page_id, 'title': title, 'snippet': snippet, 'timestamp': str(item.get('timestamp') or '')}
            candidates.append(EcosystemCandidate(candidate_id=f'wikipedia:{language}:{page_id}', provider='wikipedia', resource_kind='gameplay_or_domain_reference', title=title, summary=snippet, source_url=source_url, api_url=url, license_id='CC-BY-SA-4.0', license_url='https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use', license_policy='reference_only; page media and external links have separate rights', minecraft_version='not_applicable', loader='not_applicable', compatibility='design_reference_only; verify primary sources', attribution='Wikipedia contributors; exact revision required if quoted', preview_urls=(), reuse_status='reference_only_not_code_or_asset', evidence_sha256=_sha256_text(canonical_json(stable))))
        continuation = raw.get('continue') if isinstance(raw, dict) else None
        next_offset = _nonnegative_int(continuation.get('sroffset')) if isinstance(continuation, dict) and type(continuation.get('sroffset')) is int else None
        return (candidates, total, next_offset)

    def _get_json(self, url: str, *, params: dict[str, str] | None=None, provider: str='', include_next_url: bool=False) -> Any:
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname not in {'api.modrinth.com', 'api.github.com', 'api.openverse.org', 'en.wikipedia.org', 'ko.wikipedia.org', 'huggingface.co', 'api.openalex.org', 'api.crossref.org'}:
            raise SpecValidationError('Discovery request escaped the API allowlist.')
        if parsed.hostname == 'huggingface.co' and (not (parsed.path == '/api/models' or parsed.path.startswith('/api/models/'))):
            raise SpecValidationError('Hugging Face discovery is restricted to metadata API paths.')
        if parsed.hostname == 'api.openalex.org' and (not (parsed.path == '/works' or parsed.path.startswith('/works/'))):
            raise SpecValidationError('OpenAlex discovery is restricted to works metadata paths.')
        if parsed.hostname == 'api.crossref.org' and (not (parsed.path == '/works' or parsed.path.startswith('/works/'))):
            raise SpecValidationError('Crossref discovery is restricted to works metadata paths.')
        headers = {'Accept': 'application/json', 'User-Agent': _USER_AGENT}
        if provider == 'github':
            headers['X-GitHub-Api-Version'] = '2022-11-28'
            headers['Accept'] = 'application/vnd.github+json'
            if self.github_token:
                headers['Authorization'] = f'Bearer {self.github_token}'
        elif provider == 'openverse' and self.openverse_token:
            headers['Authorization'] = f'Bearer {self.openverse_token}'
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=False, transport=self.transport, headers=headers) as client:
                response = client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise EcosystemDiscoveryUnavailable(f'{parsed.hostname} discovery request failed: {type(exc).__name__}.') from exc
        if response.status_code != 200:
            raise EcosystemDiscoveryUnavailable(f'{parsed.hostname} discovery returned HTTP {response.status_code}.')
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise EcosystemDiscoveryUnavailable(f'{parsed.hostname} discovery response exceeded the byte policy.')
        try:
            payload = response.json()
        except ValueError as exc:
            raise EcosystemDiscoveryUnavailable(f'{parsed.hostname} discovery returned invalid JSON.') from exc
        if include_next_url:
            next_link = response.links.get('next')
            next_url = str(next_link.get('url') or '') if isinstance(next_link, dict) else ''
            return (payload, next_url)
        return payload

def discover_seed_bundle(prompt: str, game_design: dict[str, Any], *, research_brief: dict[str, Any] | None=None, client: EcosystemDiscoveryClient | None=None, route_cursor: str='', route_limit: int=12) -> dict[str, Any]:
    mode = os.environ.get('MMM_ECOSYSTEM_DISCOVERY', 'auto').strip().lower()
    if mode not in {'auto', 'on', 'off'}:
        raise SpecValidationError('MMM_ECOSYSTEM_DISCOVERY must be auto, on or off.')
    if type(route_limit) is not int or not 1 <= route_limit <= 100:
        raise SpecValidationError('route_limit must be between 1 and 100.')
    query = _seed_query(prompt, game_design)
    if research_brief is None:
        routes = [{'domain_id': 'request', 'provider': provider, 'query': query, 'target_profile': 'media' if provider in {'openverse_images'} else 'minecraft_mod'} for provider in ['modrinth', 'openverse_images']]
        if client is not None and client.github_token or os.environ.get('GITHUB_TOKEN'):
            routes.append({'domain_id': 'request', 'provider': 'github', 'query': query, 'target_profile': 'minecraft_mod'})
    else:
        routes = list(external_discovery_routes(research_brief))
    selected_target: Mapping[str, Any] = {}
    if isinstance(research_brief, Mapping):
        raw_target = research_brief.get('_mmm_platform_target')
        if isinstance(raw_target, Mapping):
            selected_target = raw_target
    minecraft_version = str(selected_target.get('minecraft_version') or '') or None
    loader = str(selected_target.get('loader') or '') or None
    route_receipt = _sha256_text(canonical_json({'routes': routes, 'minecraft_version': minecraft_version or 'unresolved', 'loader': loader or 'unresolved'}))
    route_offset = _decode_seed_route_cursor(route_cursor, route_sha256=route_receipt, route_limit=route_limit)
    if route_offset > len(routes):
        raise SpecValidationError('Seed route cursor is beyond the route catalog.')
    selected_routes = routes[route_offset:route_offset + route_limit]
    next_route_offset = route_offset + len(selected_routes)
    next_route_cursor = _encode_seed_route_cursor(next_route_offset, route_sha256=route_receipt, route_limit=route_limit) if next_route_offset < len(routes) else ''
    if mode == 'off':
        return {'schema_version': 'mmm/ecosystem-seed-bundle-v2', 'status': 'disabled', 'query_sha256': _sha256_text(query), 'route_sha256': route_receipt, 'route_count': len(routes), 'route_offset': route_offset, 'processed_route_count': 0, 'remaining_route_count': len(routes) - route_offset, 'next_route_cursor': next_route_cursor, 'routes_complete': not next_route_cursor, 'candidate_count': 0, 'pages': [], 'errors': [], 'coverage': 'specialist discovery still required per production batch', 'authorization': 'none', 'download_performed': False}
    discovery = client or EcosystemDiscoveryClient()
    pages: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for route in selected_routes:
        provider = route['provider']
        provider_query = route['query']
        if provider == 'openverse_images':
            provider_query += ' visual reference texture architecture objects'
        target_profile = str(route.get('target_profile', 'minecraft_mod'))
        try:
            page = discovery.search(provider, provider_query, limit=10, minecraft_version=minecraft_version if target_profile == 'minecraft_mod' else None, loader=loader if target_profile == 'minecraft_mod' else None, target_profile=target_profile)
            pages.append({**page, 'research_domain_id': route['domain_id'], 'route_query_sha256': _sha256_text(route['query'])})
        except EcosystemDiscoveryUnavailable as exc:
            errors.append({'domain_id': route['domain_id'], 'provider': provider, 'query_sha256': _sha256_text(route['query']), 'error_type': type(exc).__name__, 'message': str(exc)})
    candidate_count = sum((int(page.get('returned', 0)) for page in pages if isinstance(page, dict)))
    status = 'available' if candidate_count else 'empty' if pages else 'unavailable'
    if mode == 'on' and (not pages):
        raise EcosystemDiscoveryUnavailable('Required ecosystem discovery providers were unavailable.')
    return {'schema_version': 'mmm/ecosystem-seed-bundle-v2', 'status': status, 'query_sha256': _sha256_text(query), 'route_sha256': route_receipt, 'route_count': len(routes), 'route_offset': route_offset, 'processed_route_count': len(selected_routes), 'remaining_route_count': len(routes) - next_route_offset, 'next_route_cursor': next_route_cursor, 'routes_complete': not next_route_cursor, 'candidate_count': candidate_count, 'pages': pages, 'errors': errors, 'coverage': 'seed pages only; targetless Minecraft pages are shallow metadata hypotheses; after host target selection continue provider cursors and run exact project inspection for every dependency or third-party asset considered', 'authorization': 'none', 'download_performed': False}

def _encode_seed_route_cursor(offset: int, *, route_sha256: str, route_limit: int) -> str:
    payload = f'{route_sha256}\x00{route_limit}\x00{offset}'
    checksum = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
    return f'routes:{offset}:{checksum}'

def _decode_seed_route_cursor(cursor: str, *, route_sha256: str, route_limit: int) -> int:
    if not cursor:
        return 0
    if not isinstance(cursor, str) or len(cursor) > 96:
        raise SpecValidationError('Seed route cursor is invalid.')
    match = _SEED_ROUTE_CURSOR.fullmatch(cursor)
    if match is None:
        raise SpecValidationError('Seed route cursor is invalid.')
    offset = int(match.group(1))
    if _encode_seed_route_cursor(offset, route_sha256=route_sha256, route_limit=route_limit) != cursor:
        raise SpecValidationError('Seed route cursor does not match this route catalog and page size.')
    return offset

def _seed_query(prompt: str, game_design: dict[str, Any]) -> str:
    parts = [prompt, str(game_design.get('title', '')), str(game_design.get('pitch', ''))]
    for item in game_design.get('modules', []):
        if isinstance(item, dict):
            parts.append(str(item.get('reason') or item.get('name') or ''))
    for item in game_design.get('assets', []):
        if isinstance(item, dict):
            parts.append(str(item.get('brief') or ''))
    return ' '.join(part.strip() for part in parts if part.strip())

def _encode_cursor(*, provider: str, query: str, position_kind: str, position: int, target_profile: str='minecraft_mod', minecraft_version: str='unresolved', loader: str='unresolved') -> str:
    payload = f'{provider}\x00{_sha256_text(query)}\x00{target_profile}\x00{minecraft_version}\x00{loader}\x00{position_kind}\x00{position}'
    checksum = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
    return f'{position_kind}:{position}:{checksum}'

def _decode_cursor(cursor: str, *, provider: str, query: str, position_kind: str, target_profile: str='minecraft_mod', minecraft_version: str='unresolved', loader: str='unresolved') -> int:
    if not cursor:
        return 0
    if not isinstance(cursor, str) or len(cursor) > 96:
        raise SpecValidationError('Discovery cursor is invalid.')
    match = _CURSOR.fullmatch(cursor)
    if match is None or match.group(1) != position_kind:
        raise SpecValidationError('Discovery cursor is invalid.')
    position = int(match.group(2))
    if _encode_cursor(provider=provider, query=query, position_kind=position_kind, position=position, target_profile=target_profile, minecraft_version=minecraft_version, loader=loader) != cursor:
        raise SpecValidationError('Discovery cursor does not match this provider and query or target profile.')
    return position

def _encode_token_cursor(*, provider: str, query: str, target_profile: str, token: str, minecraft_version: str='not_applicable', loader: str='not_applicable') -> str:
    if not isinstance(token, str) or not token or len(token.encode('utf-8')) > 1024:
        raise SpecValidationError('Provider pagination token is invalid.')
    encoded = base64.urlsafe_b64encode(token.encode('utf-8')).decode('ascii').rstrip('=')
    payload = f'{provider}\x00{_sha256_text(query)}\x00{target_profile}\x00{minecraft_version}\x00{loader}\x00token\x00{token}'
    checksum = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
    return f'token:{encoded}:{checksum}'

def _decode_token_cursor(cursor: str, *, provider: str, query: str, target_profile: str, minecraft_version: str='not_applicable', loader: str='not_applicable') -> str:
    if not cursor:
        return ''
    if not isinstance(cursor, str) or len(cursor) > 2200:
        raise SpecValidationError('Discovery cursor is invalid.')
    match = _TOKEN_CURSOR.fullmatch(cursor)
    if match is None:
        raise SpecValidationError('Discovery cursor is invalid.')
    encoded = match.group(1)
    padded = encoded + '=' * (-len(encoded) % 4)
    try:
        token_bytes = base64.b64decode(padded, altchars=b'-_', validate=True)
        token = token_bytes.decode('utf-8')
    except (UnicodeDecodeError, ValueError, binascii.Error) as exc:
        raise SpecValidationError('Discovery cursor is invalid.') from exc
    if _encode_token_cursor(provider=provider, query=query, target_profile=target_profile, token=token, minecraft_version=minecraft_version, loader=loader) != cursor:
        raise SpecValidationError('Discovery cursor does not match this provider and query or target profile.')
    return token

def _huggingface_cursor_from_next_url(next_url: str) -> str | None:
    if not next_url:
        return None
    parsed = urlparse(next_url)
    if parsed.scheme != 'https' or parsed.netloc != 'huggingface.co' or parsed.path.rstrip('/') != '/api/models' or parsed.fragment:
        raise EcosystemDiscoveryUnavailable('Hugging Face returned an unsafe pagination link.')
    values = parse_qs(parsed.query, keep_blank_values=True).get('cursor', [])
    if len(values) != 1:
        raise EcosystemDiscoveryUnavailable('Hugging Face returned an invalid pagination cursor.')
    token = values[0]
    if not token or len(token.encode('utf-8')) > 1024:
        raise EcosystemDiscoveryUnavailable('Hugging Face returned an invalid pagination cursor.')
    return token

def _normalize_hf_repo_id(value: Any) -> str:
    if not isinstance(value, str):
        return ''
    repo_id = value.strip()
    if not repo_id or len(repo_id.encode('utf-8')) > 256:
        return ''
    parts = repo_id.split('/')
    if len(parts) not in {1, 2}:
        return ''
    if any((_HF_MODEL_COMPONENT.fullmatch(part) is None for part in parts)):
        return ''
    return repo_id

def _normalize_hf_model(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    model_id = _normalize_hf_repo_id(value.get('id') or value.get('modelId'))
    revision_sha = str(value.get('sha') or '').strip().lower()
    if not model_id or _HF_REVISION.fullmatch(revision_sha) is None:
        return None
    tags = sorted(_normalized_string_list(value.get('tags')))
    card = _normalize_hf_card(value.get('cardData'), tags=tags)
    gated_value = value.get('gated', False)
    if isinstance(gated_value, bool):
        gated = gated_value
        gated_status = str(gated_value).lower()
    elif isinstance(gated_value, str):
        gated_status = gated_value.strip().lower()
        gated = gated_status not in {'', 'false', 'none', 'no'}
    else:
        gated = bool(gated_value)
        gated_status = 'unrecognized' if gated else 'false'
    return {'model_id': model_id, 'revision_sha': revision_sha, 'author': _bounded_metadata_text(value.get('author')), 'pipeline_tag': _bounded_metadata_text(value.get('pipeline_tag')), 'library_name': _bounded_metadata_text(value.get('library_name')), 'last_modified': _bounded_metadata_text(value.get('lastModified')), 'created_at': _bounded_metadata_text(value.get('createdAt')), 'downloads': _nonnegative_int(value.get('downloads')), 'likes': _nonnegative_int(value.get('likes')), 'private': bool(value.get('private', False)), 'gated': gated, 'gated_status': gated_status, 'disabled': bool(value.get('disabled', False)), 'tags': tags, 'card': card, 'files': _normalize_hf_files(value.get('siblings'))}

def _normalize_hf_card(value: Any, *, tags: list[str]) -> dict[str, Any]:
    card = value if isinstance(value, dict) else {}
    card_license = str(card.get('license') or '').strip() if isinstance(card.get('license'), str) else ''
    tag_licenses = sorted({tag.split(':', 1)[1].strip() for tag in tags if tag.lower().startswith('license:') and tag.split(':', 1)[1].strip()})
    declared = sorted({license_id for license_id in [card_license, *tag_licenses] if license_id})
    license_conflict = len({item.lower() for item in declared}) > 1
    if license_conflict:
        license_id = ''
        license_evidence = 'conflicting_card_and_tag_metadata'
    elif card_license:
        license_id = card_license
        license_evidence = 'model_card'
    elif len(tag_licenses) == 1:
        license_id = tag_licenses[0]
        license_evidence = 'hub_tag_only'
    else:
        license_id = ''
        license_evidence = 'missing'
    return {'license_id': license_id, 'license_url': _safe_https_url(card.get('license_link'), allow_empty=True), 'license_evidence': license_evidence, 'declared_license_ids': declared, 'license_conflict': license_conflict, 'base_models': sorted(_normalized_string_list(card.get('base_model'))), 'datasets': sorted(_normalized_string_list(card.get('datasets'))), 'languages': sorted(_normalized_string_list(card.get('language'))), 'pipeline_tag': _bounded_metadata_text(card.get('pipeline_tag')), 'library_name': _bounded_metadata_text(card.get('library_name'))}

def _normalize_hf_files(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    suffixes = sorted(_SAFE_MODEL_SUFFIXES | _UNSAFE_SERIALIZATION_SUFFIXES, key=len, reverse=True)
    for item in value:
        if not isinstance(item, dict):
            continue
        path = _safe_repo_path(item.get('rfilename'))
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        suffix = next((candidate for candidate in suffixes if path.lower().endswith(candidate)), '')
        lfs = item.get('lfs') if isinstance(item.get('lfs'), dict) else {}
        lfs_sha256 = str(lfs.get('sha256') or '').strip().lower()
        if not re.fullmatch('[0-9a-f]{64}', lfs_sha256):
            lfs_sha256 = ''
        blob_id = str(item.get('blobId') or '').strip().lower()
        if not re.fullmatch('[0-9a-f]{40,64}', blob_id):
            blob_id = ''
        size = _nonnegative_int(item.get('size')) or _nonnegative_int(lfs.get('size'))
        files.append({'path': path, 'size': size, 'blob_id': blob_id, 'lfs_sha256': lfs_sha256, 'safe_data_format': suffix in _SAFE_MODEL_SUFFIXES, 'unsafe_serialization': suffix in _UNSAFE_SERIALIZATION_SUFFIXES, 'is_weight_artifact': bool(suffix), 'is_repository_code': path.lower().endswith(('.bat', '.cmd', '.jar', '.js', '.ps1', '.py', '.sh'))})
    return sorted(files, key=lambda item: item['path'])

def _hf_format_inventory(files: list[dict[str, Any]]) -> dict[str, Any]:
    safe_paths = [file['path'] for file in files if file['safe_data_format']]
    unsafe_paths = [file['path'] for file in files if file['unsafe_serialization']]
    code_paths = [file['path'] for file in files if file['is_repository_code']]
    return {'has_safetensors': any((path.lower().endswith('.safetensors') for path in safe_paths)), 'has_gguf': any((path.lower().endswith('.gguf') for path in safe_paths)), 'has_onnx': any((path.lower().endswith('.onnx') for path in safe_paths)), 'safe_data_format_files': safe_paths, 'unsafe_serialization_files': unsafe_paths, 'repository_code_files': code_paths, 'safe_format_is_not_execution_authorization': True}

def _safe_repo_path(value: Any) -> str:
    if not isinstance(value, str):
        return ''
    path = value.strip().replace('\\', '/')
    if not path or len(path.encode('utf-8')) > 1024 or path.startswith('/') or any((part in {'', '.', '..'} for part in path.split('/'))):
        return ''
    return path

def _normalized_string_list(value: Any) -> list[str]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or len(text.encode('utf-8')) > 2048 or text in result:
            continue
        result.append(text)
    return result

def _bounded_metadata_text(value: Any) -> str:
    if not isinstance(value, str):
        return ''
    text = value.strip()
    if len(text.encode('utf-8')) > 4096:
        return ''
    return text

def _model_license_policy(license_id: str) -> str:
    normalized = license_id.strip().lower()
    if normalized in _REVIEWABLE_MODEL_LICENSES:
        return 'model_artifact_license_manual_review_required; code, datasets and outputs have separate rights'
    if normalized in _RESTRICTED_MODEL_LICENSES:
        return 'restricted_model_license_manual_review_required; verify use, distribution and acceptable-use terms'
    return 'reject_until_a_recognized_model_license_is_verified; custom, missing or conflicting metadata is not permission'

def _normalize_modrinth_version(version: dict[str, Any], *, minecraft_version: str, loader: str) -> dict[str, Any]:
    reasons: list[str] = []
    version_id = str(version.get('id') or '').strip()
    version_number = _bounded_metadata_text(version.get('version_number'))
    version_type = str(version.get('version_type') or '').strip().lower()
    status = str(version.get('status') or '').strip().lower()
    game_versions = _normalized_string_list(version.get('game_versions'))
    loaders = [value.lower() for value in _normalized_string_list(version.get('loaders'))]
    if _MODRINTH_ID.fullmatch(version_id) is None:
        reasons.append('missing_or_invalid_immutable_version_id')
    if not version_number:
        reasons.append('missing_version_number')
    if minecraft_version not in game_versions or loader not in loaders:
        reasons.append('off_target_minecraft_version_or_loader')
    if status and status != 'listed':
        reasons.append('version_is_not_listed')
    if version_type and version_type not in {'release', 'beta', 'alpha'}:
        reasons.append('unrecognized_version_type')
    files: list[dict[str, Any]] = []
    for item in version.get('files', []):
        if not isinstance(item, dict):
            continue
        filename = str(item.get('filename') or '').strip()
        url = _safe_https_url(item.get('url'))
        host = (urlparse(url).hostname or '').lower() if url else ''
        hashes = item.get('hashes') if isinstance(item.get('hashes'), dict) else {}
        sha1 = str(hashes.get('sha1') or '').strip().lower()
        sha512 = str(hashes.get('sha512') or '').strip().lower()
        safe_filename = bool(filename and len(filename.encode('utf-8')) <= 512 and ('/' not in filename) and ('\\' not in filename) and filename.lower().endswith('.jar'))
        safe_origin = bool(host == 'cdn.modrinth.com' or host.endswith('.modrinth.com'))
        size = _nonnegative_int(item.get('size'))
        files.append({'filename': filename, 'url': url, 'size': size, 'primary': item.get('primary') is True, 'sha1': sha1 if _SHA1_HEX.fullmatch(sha1) else '', 'sha512': sha512 if _SHA512_HEX.fullmatch(sha512) else '', 'safe_filename': safe_filename, 'safe_origin': safe_origin, 'strong_digest_valid': bool(_SHA512_HEX.fullmatch(sha512))})
    primary_files = [file for file in files if file['primary']]
    if len(primary_files) != 1:
        reasons.append('exactly_one_primary_file_is_required')
    elif not (primary_files[0]['safe_filename'] and primary_files[0]['safe_origin'] and (primary_files[0]['size'] > 0) and primary_files[0]['strong_digest_valid']):
        reasons.append('primary_file_requires_safe_origin_size_and_sha512')
    dependencies: list[dict[str, Any]] = []
    allowed_dependency_types = {'required', 'optional', 'incompatible', 'embedded'}
    for item in version.get('dependencies', []):
        if not isinstance(item, dict):
            reasons.append('malformed_dependency_record')
            continue
        dependency_type = str(item.get('dependency_type') or '').strip().lower()
        project_id = item.get('project_id')
        dependency_version_id = item.get('version_id')
        if dependency_type not in allowed_dependency_types:
            reasons.append('unrecognized_dependency_type')
        if project_id is None and dependency_version_id is None:
            reasons.append('dependency_has_no_project_or_version_id')
        dependencies.append({'project_id': project_id, 'version_id': dependency_version_id, 'dependency_type': dependency_type})
    return {'version_id': version_id, 'version_number': version_number, 'version_type': version_type, 'status': status, 'date_published': _bounded_metadata_text(version.get('date_published')), 'game_versions': game_versions, 'loaders': loaders, 'dependencies': dependencies, 'files': files, 'eligible_for_selection': not reasons, 'unresolved_or_rejected_gates': list(dict.fromkeys(reasons))}

def _normalized_doi(value: Any) -> str:
    doi = str(value or '').strip()
    for prefix in ('https://doi.org/', 'http://doi.org/', 'doi:'):
        if doi.casefold().startswith(prefix):
            doi = doi[len(prefix):]
            break
    if not doi or len(doi.encode('utf-8')) > 512 or (not doi.startswith('10.')):
        return ''
    if any((character.isspace() or ord(character) < 32 for character in doi)):
        return ''
    return doi

def _openalex_authors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for authorship in value[:100]:
        author = authorship.get('author') if isinstance(authorship, dict) else None
        name = ' '.join(str(author.get('display_name') or '').split()) if isinstance(author, dict) else ''
        if name and name not in authors:
            authors.append(name[:256])
    return authors

def _crossref_authors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for author in value[:100]:
        if not isinstance(author, dict):
            continue
        name = ' '.join((part.strip() for part in (str(author.get('given') or ''), str(author.get('family') or '')) if part.strip()))
        if name and name not in authors:
            authors.append(name[:256])
    return authors

def _crossref_date(value: Any) -> str:
    parts = value.get('date-parts') if isinstance(value, dict) else None
    first = parts[0] if isinstance(parts, list) and parts else None
    if not isinstance(first, list) or not first:
        return ''
    integers = [part for part in first[:3] if type(part) is int and part >= 1]
    return '-'.join((str(part) if index == 0 else f'{part:02d}' for index, part in enumerate(integers)))

def _plain_bounded_text(value: Any, limit: int) -> str:
    text = html.unescape(re.sub('<[^>]*>', ' ', str(value or '')))
    return ' '.join(text.split())[:limit]

def _scholarly_summary(metadata: dict[str, Any]) -> str:
    year = metadata.get('publication_year') or metadata.get('published') or 'undated'
    work_type = str(metadata.get('work_type') or 'research work')
    citations = _nonnegative_int(metadata.get('cited_by_count'))
    abstract = str(metadata.get('abstract') or '').strip()
    base = f'{work_type}, {year}; cited-by metadata: {citations}.'
    return (base + ' ' + abstract)[:1600] if abstract else base

def _code_license_policy(license_id: str) -> str:
    if license_id in _PERMISSIVE_CODE_LICENSES:
        return 'permissive_candidate; preserve notices and inspect dependencies'
    if not license_id or license_id.upper() in {'ARR', 'NOASSERTION'}:
        return 'reject_until_a_recognized_license_is_verified'
    return 'copyleft_or_custom_review_required; dependency use is not permission to copy source or assets'

def _media_license_policy(license_name: str) -> str:
    return {'cc0': 'origin_verification_required; attribution_optional', 'pdm': 'origin_public_domain_verification_required', 'by': 'origin_verification_and_attribution_required', 'by-sa': 'origin_verification_attribution_and_share_alike_required'}[license_name]

def _creative_commons_id(name: str, version: str) -> str:
    if name == 'pdm':
        return 'PDM-1.0'
    if name == 'cc0':
        return 'CC0-1.0'
    return f"CC-{name.upper()}-{version or 'unknown'}"

def _safe_https_url(value: Any, *, allow_empty: bool=False) -> str:
    text = str(value or '').strip()
    if not text and allow_empty:
        return ''
    parsed = urlparse(text)
    if parsed.scheme != 'https' or not parsed.hostname:
        return ''
    return text

def _safe_preview_urls(values: list[Any], *, allowed_hosts: set[str] | None) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        url = _safe_https_url(value, allow_empty=True)
        if not url:
            continue
        host = urlparse(url).hostname
        if allowed_hosts is not None and host not in allowed_hosts:
            continue
        if url not in result:
            result.append(url)
    return tuple(result)

def _nonnegative_int(value: Any) -> int:
    return value if type(value) is int and value >= 0 else 0

def _sha256_text(value: str) -> str:
    return 'sha256:' + hashlib.sha256(value.encode('utf-8')).hexdigest()
