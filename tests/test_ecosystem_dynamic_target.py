from __future__ import annotations

import json

import httpx
import pytest

from minecraft_mod_ai.ecosystem_discovery import EcosystemDiscoveryClient
from minecraft_mod_ai.spec import SpecValidationError


def _modrinth_hit() -> dict[str, object]:
    return {
        "project_id": "dynamic-target-project",
        "slug": "dynamic-target-project",
        "title": "Dynamic Target Project",
        "description": "Target-aware metadata candidate",
        "license": "MIT",
        "project_type": "mod",
        "versions": ["1.20.1", "1.21.1"],
        "categories": ["fabric"],
        "gallery": [],
    }


def test_targetless_modrinth_search_is_platform_neutral() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        facets = json.loads(request.url.params["facets"])
        assert ["versions:1.20.1"] not in facets
        assert ["categories:fabric"] not in facets
        assert ["project_type:mod"] in facets
        return httpx.Response(
            200,
            json={"hits": [_modrinth_hit()], "total_hits": 1},
        )

    page = EcosystemDiscoveryClient(
        transport=httpx.MockTransport(handler)
    ).search("modrinth", "inventory helper", limit=1)

    assert page["target_exact"] is False
    assert page["minecraft_version"] == "unresolved"
    assert page["loader"] == "unresolved"
    candidate = page["candidates"][0]
    assert candidate["minecraft_version"] == "unresolved"
    assert candidate["loader"] == "unresolved"
    assert "target_hypothesis_required" in candidate["compatibility"]


def test_exact_modrinth_search_uses_host_selected_target_facets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        facets = json.loads(request.url.params["facets"])
        assert ["versions:1.21.1"] in facets
        assert ["categories:fabric"] in facets
        return httpx.Response(
            200,
            json={"hits": [_modrinth_hit()], "total_hits": 1},
        )

    page = EcosystemDiscoveryClient(
        transport=httpx.MockTransport(handler)
    ).search(
        "modrinth",
        "inventory helper",
        limit=1,
        minecraft_version="1.21.1",
        loader="fabric",
    )

    assert page["target_exact"] is True
    assert page["minecraft_version"] == "1.21.1"
    assert page["loader"] == "fabric"
    candidate = page["candidates"][0]
    assert candidate["minecraft_version"] == "1.21.1"
    assert candidate["loader"] == "fabric"


def test_discovery_cursor_is_bound_to_platform_target() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"hits": [_modrinth_hit()], "total_hits": 2},
        )

    client = EcosystemDiscoveryClient(transport=httpx.MockTransport(handler))
    first = client.search(
        "modrinth",
        "inventory helper",
        limit=1,
        minecraft_version="1.20.1",
        loader="fabric",
    )
    assert first["next_cursor"]

    with pytest.raises(SpecValidationError, match="cursor"):
        client.search(
            "modrinth",
            "inventory helper",
            limit=1,
            cursor=first["next_cursor"],
            minecraft_version="1.21.1",
            loader="fabric",
        )


def test_exact_project_inspection_cannot_run_without_host_target() -> None:
    client = EcosystemDiscoveryClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(500, json={})
        )
    )
    with pytest.raises(SpecValidationError, match="host-selected"):
        client.inspect_modrinth_project("dynamic-target-project")
