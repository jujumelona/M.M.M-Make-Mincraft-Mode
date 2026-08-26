import httpx
import pytest

from minecraft_mod_ai.ecosystem_discovery import (
    EcosystemDiscoveryClient,
    EcosystemDiscoveryUnavailable,
)
from minecraft_mod_ai.spec import SpecValidationError


def _client(handler) -> EcosystemDiscoveryClient:
    return EcosystemDiscoveryClient(transport=httpx.MockTransport(handler))

def test_target_profile_binds_numeric_cursors_too() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={'total_count': 2, 'items': []})
    client = _client(handler)
    first = client.search('github', 'runtime', target_profile='ai_runtime', limit=1)
    assert first['next_cursor'].startswith('page:2:')
    with pytest.raises(SpecValidationError, match='target profile'):
        client.search('github', 'runtime', target_profile='minecraft_mod', cursor=first['next_cursor'], limit=1)
    assert len(requests) == 1

def test_huggingface_inspection_rejects_revision_mismatch() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        revision = 'a' * 40 if calls == 1 else 'b' * 40
        return httpx.Response(200, json={'id': 'example/model', 'sha': revision, 'cardData': {'license': 'mit'}})
    with pytest.raises(EcosystemDiscoveryUnavailable, match='pinned revision'):
        _client(handler).inspect_huggingface_model('example/model')
    assert calls == 2

def test_huggingface_metadata_api_cannot_be_used_to_download_weights() -> None:

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError('Non-metadata paths must not reach the transport')
    client = _client(handler)
    with pytest.raises(SpecValidationError, match='metadata API paths'):
        client._get_json('https://huggingface.co/example/model/resolve/main/model.safetensors')
