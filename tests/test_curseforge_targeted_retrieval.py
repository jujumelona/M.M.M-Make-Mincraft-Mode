"""CurseForge retrieval must be semantic, scoped, and provenance-safe."""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from minecraft_mod_ai import pre_design_grounded_rag, reuse_discovery
from minecraft_mod_ai.curseforge_query import (
    curseforge_catalogue_score, curseforge_search_queries,
)


def test_no_empty_queries_and_bilingual_intent():
    assert curseforge_search_queries("") == ()
    assert curseforge_search_queries("minecraft fabric mod game") == ()
    assert "spaceship" in curseforge_search_queries("우주선 조립 및 외계인 전투")
    assert "alien" in curseforge_search_queries("우주선 조립 및 외계인 전투")
    assert "trading" in curseforge_search_queries("자원 채굴 거래")
    assert "space" in curseforge_search_queries("launch into space")
    assert all(term.strip() for term in curseforge_search_queries("spaceship aliens"))


def test_title_relevance_outranks_unrelated_popularity():
    question = "spacecraft crew and alien combat"
    relevant = {"name": "Spaceship", "summary": "Crew combat and aliens"}
    irrelevant = {"name": "Farming Deluxe", "summary": "Crops and animals"}
    assert curseforge_catalogue_score(question, relevant) > 0
    assert curseforge_catalogue_score(question, irrelevant) == 0


def test_reuse_discovery_isolates_capabilities_and_cache(monkeypatch):
    monkeypatch.setattr(reuse_discovery, "_curseforge_api_key", lambda: "dummy")
    monkeypatch.setattr(reuse_discovery, "_query_variant_limit", lambda: 2)
    requests = []

    def fake_search(query, *, minecraft_version, loader, **_kwargs):
        assert query
        assert minecraft_version == "1.21.8"
        assert loader == "fabric"
        requests.append(query)
        if query == "spaceship":
            return [("", 10.0, {
                "source_id": "curseforge:100", "catalog_id": 100,
                "title": "Spaceship Construction", "summary": "Ship in space",
                "catalog_url": "https://www.curseforge.com/minecraft/mc-mods/ship",
            })]
        if query == "trading":
            return [("", 10.0, {
                "source_id": "curseforge:200", "catalog_id": 200,
                "title": "Trading", "summary": "Trade with villagers",
                "catalog_url": "https://www.curseforge.com/minecraft/mc-mods/trading",
            })]
        return []

    monkeypatch.setattr(reuse_discovery, "_search_curseforge", fake_search)
    diagnostics = []
    cache = {}
    graph = {"search_terms": [
        {"capability": "spaceship", "terms": ["ship"]},
        {"capability": "trading", "terms": ["economy"]},
    ]}
    result = reuse_discovery.discover_repositories_for_graph(
        ["spaceship", "trading"], None, capability_graph=graph,
        minecraft_version="1.21.8", loader="fabric", diagnostics=diagnostics,
        query_cache=cache,
    )
    assert result == {"spaceship": (), "trading": ()}
    assert len(diagnostics) == 2
    ship, trade = diagnostics
    assert ship["status"] == "catalog_only"
    assert trade["status"] == "catalog_only"
    assert [x["source_id"] for x in ship["ranked_catalogue_hits"]] == ["curseforge:100"]
    assert [x["source_id"] for x in trade["ranked_catalogue_hits"]] == ["curseforge:200"]
    assert not any("github" in item for item in requests)
    before = len(requests)
    second = []
    reuse_discovery.discover_repositories_for_graph(
        ["spaceship", "trading"], None, capability_graph=graph,
        minecraft_version="1.21.8", loader="fabric", diagnostics=second,
        query_cache=cache,
    )
    assert len(requests) == before
    assert all(q["cache_hit"] for row in second for q in row["query_receipts"])


def test_predesign_provider_search_keeps_version_loader_and_feature(monkeypatch):
    monkeypatch.setattr(pre_design_grounded_rag, "curseforge_api_key", lambda: "test")
    monkeypatch.setattr(pre_design_grounded_rag, "_provider_page_limit", lambda: 1)
    monkeypatch.setattr(pre_design_grounded_rag, "_provider_result_limit", lambda: 10)
    requests = []
    def fake_json(url, _headers=None):
        if "/categories?" in url:
            return {"data": [
                {"id": 6, "gameId": 432, "name": "Mods",
                 "slug": "mc-mods", "isClass": True},
            ]}
        if "/description" in url:
            return {"data": "<p>Assemble and pilot the spaceship.</p>"}
        if "/mods/search?" in url:
            params = parse_qs(urlsplit(url).query)
            assert params.get("classId") == ["6"]
            assert params.get("gameVersion") == ["1.21.8"]
            assert params.get("modLoaderType") == ["4"]
            term = params.get("searchFilter", [""])[0]
            assert term
            requests.append(term)
            return {"data": [
                {"id": 100, "classId": 6, "name": "Spaceship Construction",
                 "summary": "Pilot a ship into space", "links": {}},
                {"id": 101, "classId": 6, "name": "Unrelated Animals",
                 "summary": "Farming and animals", "links": {}},
            ], "pagination": {
                "index": 0, "pageSize": 50, "resultCount": 2, "totalCount": 2,
            }}
        raise AssertionError(url)
    monkeypatch.setattr(pre_design_grounded_rag, "_json", fake_json)
    records, receipt = pre_design_grounded_rag._search_curseforge(
        "spaceship", minecraft_version="1.21.8", loader="fabric"
    )
    assert requests == ["spaceship", "spacecraft", "space"]
    assert receipt["search_filter_used"] is True
    assert [r["source_id"] for r in records] == ["curseforge:100"]
    assert records[0]["body_retrieved"]
    assert records[0]["metadata"]["requested_minecraft_version"] == "1.21.8"
    assert records[0]["metadata"]["source_reuse_authority"] == "catalogue_only"


def test_direct_search_forbids_global_browse():
    with pytest.raises(ValueError, match="CURSEFORGE_EMPTY_SEARCH_FILTER_FORBIDDEN"):
        reuse_discovery._search_curseforge("", limit=5, minecraft_version="1.21.8",
                                           loader="fabric")
