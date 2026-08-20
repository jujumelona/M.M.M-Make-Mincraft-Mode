from __future__ import annotations

import threading

import httpx
import pytest

from minecraft_mod_ai import ecosystem_discovery as discovery


def test_persistent_discovery_pool_keeps_native_policy_and_allows_parallel_requests() -> None:
    assert getattr(
        discovery.EcosystemDiscoveryClient.__init__,
        "_mmm_persistent_http_pool_v2",
        False,
    )
    assert not getattr(
        discovery.EcosystemDiscoveryClient._get_json,
        "_mmm_persistent_http_pool_v1",
        False,
    )

    barrier = threading.Barrier(2)
    requests: list[httpx.Request] = []
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            requests.append(request)
        barrier.wait(timeout=2)
        return httpx.Response(
            200,
            json={"items": [], "total_count": 0},
            request=request,
        )

    client = discovery.EcosystemDiscoveryClient(
        transport=httpx.MockTransport(handler),
        github_token="test-token",
    )
    errors: list[BaseException] = []

    def fetch(query: str) -> None:
        try:
            client._get_json(
                "https://api.github.com/search/repositories",
                params={"q": query},
                provider="github",
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=fetch, args=(f"query-{index}",))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert errors == []
    assert len(requests) == 2
    assert all(request.headers["X-GitHub-Api-Version"] == "2022-11-28" for request in requests)
    assert all(request.headers["Authorization"] == "Bearer test-token" for request in requests)
    assert all(request.headers["Accept"] == "application/vnd.github+json" for request in requests)

    with pytest.raises(discovery.SpecValidationError, match="allowlist"):
        client._get_json("https://example.com/escape")
