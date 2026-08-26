from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import reuse_discovery
from minecraft_mod_ai.evidence_first_planning import (
    build_request_catalog,
    romanize_korean_universal,
)
from minecraft_mod_ai.reuse_planner import decompose_capability_graph


def test_korean_prompt_feature_decomposition_multi_capability() -> None:
    prompt = "메이플스토리 모드 → 잡몹 → 보스 → 아이템 → 레벨 성장 → 강화 시스템"
    catalog = build_request_catalog(prompt, {})
    requirements = catalog.get("requirements", [])

    assert len(requirements) >= 4
    capabilities = [req["capability"] for req in requirements]

    # Must not collapse into a single "semantic" capability
    assert "semantic" not in capabilities
    assert any("mob" in cap or "entity" in cap for cap in capabilities)
    assert any("boss" in cap for cap in capabilities)
    assert any("item" in cap or "equipment" in cap for cap in capabilities)
    assert any("level" in cap or "progression" in cap for cap in capabilities)
    assert any("upgrade" in cap for cap in capabilities)


def test_capability_graph_decomposition_and_search_terms() -> None:
    prompt = "메이플스토리 모드 → 잡몹 → 보스 → 아이템 → 레벨 성장 → 강화 시스템"
    graph = decompose_capability_graph(prompt)

    assert len(graph.nodes) >= 4
    node_set = set(graph.nodes)
    assert "semantic" not in node_set

    # Verify domain nodes exist
    assert any("boss" in node for node in graph.nodes)
    assert any("mob" in node or "entity" in node for node in graph.nodes)
    assert any("upgrade" in node or "level" in node for node in graph.nodes)

    # Verify search terms are populated with relevant Minecraft keywords
    search_dict = dict(graph.search_terms)
    for cap in graph.nodes:
        assert cap in search_dict
        terms = search_dict[cap]
        assert len(terms) >= 1
        assert any(len(t.strip()) > 2 for t in terms)


def test_arbitrary_unseen_prompt_decomposition_and_romanization() -> None:
    # Test completely arbitrary, non-hardcoded mod concepts
    prompt = "우주선 워프 엔진 • 핵융합 발전기 • 사이버네틱스 의수 • 디멘션 게이트"
    catalog = build_request_catalog(prompt, {})
    requirements = catalog.get("requirements", [])

    assert len(requirements) >= 3
    capabilities = [req["capability"] for req in requirements]

    # Must be distinct, romanized/slugified, and non-empty
    assert "semantic" not in capabilities
    assert len(set(capabilities)) >= 3
    for cap in capabilities:
        assert cap.isascii()
        assert len(cap) > 2

    # Direct test of universal romanization
    assert "ujuseon" in romanize_korean_universal("우주선")
    assert "haegyunghab" in romanize_korean_universal("핵융합")


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

    def fake_get(url, **kwargs):
        observed["url"] = url
        observed["params"] = kwargs.get("params")
        observed["headers"] = kwargs.get("headers")
        return FakeResponse()

    monkeypatch.setattr(reuse_discovery.httpx, "get", fake_get)

    results = reuse_discovery._search_curseforge("보스 시스템", limit=10)

    assert len(results) == 1
    assert results[0] == ("test-owner/epic-boss-mod", 1.0)
    assert observed["headers"]["x-api-key"] == "test-cf-key-12345"
    # Query must be sanitized to ascii/alphanumeric
    assert "보스" not in observed["params"]["searchFilter"]


def test_curseforge_failure_does_not_break_provider_discovery(monkeypatch) -> None:
    monkeypatch.setenv("MMM_CURSEFORGE_API_KEY", "broken-key")

    def failing_get(url, **kwargs):
        raise ConnectionError("CurseForge endpoint unreachable")

    monkeypatch.setattr(reuse_discovery.httpx, "get", failing_get)

    client = SimpleNamespace(
        search=lambda provider, query, **kwargs: {
            "candidates": [{"repository": "owner/modrinth-boss"}]
        }
    )

    results = reuse_discovery.discover_repositories_for_graph(
        ["boss.entity"],
        client,
        capability_graph={"search_terms": [{"capability": "boss.entity", "terms": ["boss mod"]}]},
    )

    # Even though CurseForge failed, Modrinth / GitHub results are intact
    assert "boss.entity" in results
    assert "owner/modrinth-boss" in results["boss.entity"]
