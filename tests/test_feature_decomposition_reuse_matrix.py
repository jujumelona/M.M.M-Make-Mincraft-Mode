from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import reuse_discovery
from minecraft_mod_ai.evidence_first_planning import build_request_catalog
from minecraft_mod_ai.reuse_planner import decompose_capability_graph
from tests.planning_authority_fixtures import request_catalog


def _grounded_catalog(prompt: str, *capabilities: str) -> dict:
    return request_catalog(
        prompt,
        [
            {
                "requirement_id": f"req_{index:03d}",
                "capability": capability,
                "statement": f"Implement {capability} as part of the authored request.",
                "source_text": prompt,
                "implementation_capabilities": [capability],
                "search_queries": [capability.replace(".", " ")],
            }
            for index, capability in enumerate(capabilities, start=1)
        ],
    )


def test_grounded_feature_decomposition_preserves_multi_capability_authority() -> None:
    prompt = (
        "MapleStory-style mod -> mobs -> bosses -> items -> "
        "level progression -> upgrade system"
    )
    expected = (
        "entity.mob",
        "entity.boss",
        "item.equipment",
        "progression.level",
        "progression.upgrade",
    )
    frozen = _grounded_catalog(prompt, *expected)
    catalog = build_request_catalog(
        prompt,
        {"_evidence_request_catalog": frozen},
    )
    requirements = catalog.get("requirements", [])

    assert [req["capability"] for req in requirements] == list(expected)
    assert catalog["catalog_sha256"] == frozen["catalog_sha256"]


def test_capability_graph_decomposition_and_search_terms_use_frozen_catalog() -> None:
    prompt = (
        "MapleStory-style mod -> mobs -> bosses -> items -> "
        "level progression -> upgrade system"
    )
    expected = (
        "entity.mob",
        "entity.boss",
        "item.equipment",
        "progression.level",
        "progression.upgrade",
    )
    catalog = _grounded_catalog(prompt, *expected)
    graph = decompose_capability_graph(
        prompt,
        design={"_evidence_request_catalog": catalog},
    )

    assert graph.nodes == expected
    assert "semantic" not in set(graph.nodes)

    search_dict = dict(graph.search_terms)
    for capability in expected:
        assert capability in search_dict
        terms = search_dict[capability]
        assert terms
        assert any(len(term.strip()) > 2 for term in terms)


def test_unseen_grounded_capabilities_keep_stable_ascii_identifiers() -> None:
    prompt = "warp drive • fusion generator • cybernetic arm • dimension gate"
    expected = (
        "warp.drive",
        "fusion.generator",
        "cybernetic.arm",
        "dimension.gate",
    )
    catalog = build_request_catalog(
        prompt,
        {"_evidence_request_catalog": _grounded_catalog(prompt, *expected)},
    )
    capabilities = [req["capability"] for req in catalog["requirements"]]

    assert capabilities == list(expected)
    assert len(set(capabilities)) == len(expected)
    assert all(capability.isascii() and len(capability) > 2 for capability in capabilities)


def test_curseforge_search_query_sanitization_and_isolation(monkeypatch) -> None:
    monkeypatch.setenv("MMM_CURSEFORGE_API_KEY", "test-cf-key-12345")
    observed = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {
                        "name": "Epic Boss Mod",
                        "links": {"sourceUrl": "https://github.com/test-owner/epic-boss-mod"},
                    }
                ]
            }

    class FakeClient:
        def get(self, url, **kwargs):
            observed["url"] = url
            observed["params"] = kwargs.get("params")
            observed["headers"] = kwargs.get("headers")
            return FakeResponse()

    monkeypatch.setattr(reuse_discovery, "_pooled_http_client", lambda: FakeClient())

    results = reuse_discovery._search_curseforge("boss system 🚀", limit=10)

    assert len(results) == 1
    assert results[0] == ("test-owner/epic-boss-mod", 1.0)
    assert observed["headers"]["x-api-key"] == "test-cf-key-12345"
    search_filter = observed["params"]["searchFilter"]
    assert search_filter.isascii()
    assert "🚀" not in search_filter


def test_curseforge_failure_does_not_break_provider_discovery(monkeypatch) -> None:
    monkeypatch.setenv("MMM_CURSEFORGE_API_KEY", "broken-key")

    class FailingClient:
        def get(self, url, **kwargs):
            del url, kwargs
            raise ConnectionError("CurseForge endpoint unreachable")

    monkeypatch.setattr(reuse_discovery, "_pooled_http_client", lambda: FailingClient())

    client = SimpleNamespace(
        search=lambda provider, query, **kwargs: {
            "candidates": [{"repository": "owner/modrinth-boss"}]
        }
    )

    results = reuse_discovery.discover_repositories_for_graph(
        ["boss.entity"],
        client,
        capability_graph={
            "search_terms": [
                {"capability": "boss.entity", "terms": ["boss mod"]}
            ]
        },
    )

    assert "boss.entity" in results
    assert "owner/modrinth-boss" in results["boss.entity"]
