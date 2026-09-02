from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import reuse_discovery
from minecraft_mod_ai.evidence_first_planning import build_request_catalog
from minecraft_mod_ai.reuse_planner import decompose_capability_graph


def test_prompt_feature_decomposition_multi_capability() -> None:
    prompt = (
        "MapleStory-style mod -> mobs -> bosses -> items -> "
        "level progression -> upgrade system"
    )
    catalog = build_request_catalog(prompt, {})
    requirements = catalog.get("requirements", [])

    assert len(requirements) >= 4
    capabilities = [req["capability"] for req in requirements]

    # Must not collapse into a single generic semantic capability.
    assert "semantic" not in capabilities
    assert any("mob" in cap or "entity" in cap for cap in capabilities)
    assert any("boss" in cap for cap in capabilities)
    assert any("item" in cap or "equipment" in cap for cap in capabilities)
    assert any("level" in cap or "progression" in cap for cap in capabilities)
    assert any("upgrade" in cap for cap in capabilities)


def test_capability_graph_decomposition_and_search_terms() -> None:
    prompt = (
        "MapleStory-style mod -> mobs -> bosses -> items -> "
        "level progression -> upgrade system"
    )
    catalog = build_request_catalog(prompt, {})
    graph = decompose_capability_graph(
        prompt,
        design={"_evidence_request_catalog": catalog},
    )

    assert len(graph.nodes) >= 4
    node_set = set(graph.nodes)
    assert "semantic" not in node_set

    assert any("boss" in node for node in graph.nodes)
    assert any("mob" in node or "entity" in node for node in graph.nodes)
    assert any("upgrade" in node or "level" in node for node in graph.nodes)

    search_dict = dict(graph.search_terms)
    for cap in graph.nodes:
        assert cap in search_dict
        terms = search_dict[cap]
        assert len(terms) >= 1
        assert any(len(term.strip()) > 2 for term in terms)


def test_arbitrary_unseen_prompt_decomposition_uses_stable_identifiers() -> None:
    prompt = (
        "warp drive • fusion generator • cybernetic arm • dimension gate"
    )
    catalog = build_request_catalog(prompt, {})
    requirements = catalog.get("requirements", [])

    assert len(requirements) >= 3
    capabilities = [req["capability"] for req in requirements]

    assert "semantic" not in capabilities
    assert len(set(capabilities)) >= 3
    for capability in capabilities:
        assert capability.isascii()
        assert len(capability) > 2


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
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
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
        if provider == "modrinth"
        else {"candidates": []}
    )
    result = reuse_discovery.discover_repositories_for_graph(
        ("boss.combat",),
        client,
        capability_graph={
            "nodes": ["boss.combat"],
            "search_terms": [
                {"capability": "boss.combat", "terms": ["boss combat"]}
            ],
        },
    )
    assert result["boss.combat"] == ("owner/modrinth-boss",)
