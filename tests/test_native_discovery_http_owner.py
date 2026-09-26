from __future__ import annotations

import threading

import httpx
import pytest

from minecraft_mod_ai import ecosystem_discovery as discovery


def test_persistent_discovery_pool_keeps_native_policy_and_allows_parallel_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    pooled_client = client._http_client

    def unexpected_client_construction(*_args, **_kwargs):
        raise AssertionError("discovery request rebuilt httpx.Client instead of reusing its pool")

    monkeypatch.setattr(discovery.httpx, "Client", unexpected_client_construction)

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
    assert client._http_client is pooled_client
    assert len(requests) == 2
    assert all(request.headers["X-GitHub-Api-Version"] == "2022-11-28" for request in requests)
    assert all(request.headers["Authorization"] == "Bearer test-token" for request in requests)
    assert all(request.headers["Accept"] == "application/vnd.github+json" for request in requests)

    with pytest.raises(discovery.SpecValidationError, match="allowlist"):
        client._get_json("https://example.com/escape")
