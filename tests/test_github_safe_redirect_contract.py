"""Changed repository names may redirect only within the trusted GitHub API."""
from __future__ import annotations

import httpx
import pytest

from minecraft_mod_ai.ecosystem_discovery import (
    EcosystemDiscoveryClient,
    EcosystemDiscoveryUnavailable,
)
from minecraft_mod_ai.source_transplant import (
    SourceTransplantError,
    _github_json,
)


def _route(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/repos/old/name":
        return httpx.Response(
            301,
            headers={"Location": "https://api.github.com/repositories/123"},
            request=request,
        )
    assert request.url.path == "/repositories/123"
    return httpx.Response(200, json={"id": 123}, request=request)


def test_ecosystem_accepts_same_origin_canonical_repository_redirect():
    with EcosystemDiscoveryClient(transport=httpx.MockTransport(_route), github_token="test-token") as c:
        receipt = c._get_json("https://api.github.com/repos/old/name", provider="github")
    assert receipt == {"id": 123}


def test_source_transplant_accepts_same_origin_canonical_repository_redirect():
    with httpx.Client(transport=httpx.MockTransport(_route), follow_redirects=False) as c:
        assert _github_json(c, "https://api.github.com/repos/old/name") == {"id": 123}


@pytest.mark.parametrize("target", [
    "https://example.org/steal",
    "http://api.github.com/repositories/123",
    "https://api.github.com.evil.org/repositories/123",
])
def test_discovery_rejects_untrusted_redirect(target):
    def handler(request):
        return httpx.Response(301, headers={"Location": target}, request=request)

    with EcosystemDiscoveryClient(transport=httpx.MockTransport(handler), github_token="secret") as c:
        with pytest.raises(EcosystemDiscoveryUnavailable, match="redirect"):
            c._get_json("https://api.github.com/repos/old/name", provider="github")


def test_source_transplant_rejects_untrusted_redirect():
    def handler(request):
        return httpx.Response(
            301,
            headers={"Location": "https://example.org/steal"},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as c:
        with pytest.raises(SourceTransplantError, match="redirect"):
            _github_json(c, "https://api.github.com/repos/old/name")
