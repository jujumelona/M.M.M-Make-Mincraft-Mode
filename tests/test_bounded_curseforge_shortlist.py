"""CurseForge discovery inspects a short list per topic, never the whole catalogue."""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from minecraft_mod_ai import pre_design_grounded_rag, reuse_discovery


def test_source_reuse_stops_after_first_sufficient_query(monkeypatch):
    monkeypatch.delenv("MMM_REUSE_DISCOVERY_LIMIT", raising=False)
    monkeypatch.delenv("MMM_REUSE_REPRESENTATIVE_LIMIT", raising=False)
    monkeypatch.delenv("MMM_REUSE_QUERY_VARIANTS", raising=False)
    monkeypatch.delenv("MMM_REUSE_MIN_CANDIDATES_PER_CAPABILITY", raising=False)
    monkeypatch.delenv("MMM_REUSE_DISCOVERY_PAGES", raising=False)
    assert reuse_discovery._query_limit() == 6
    assert reuse_discovery._representative_limit() == 3
    assert reuse_discovery._query_variant_limit() == 3
    assert reuse_discovery._discovery_pages() == 1

    monkeypatch.setattr(reuse_discovery, "_curseforge_api_key", lambda: "test")
    requests = []

    def search(query, *, minecraft_version, loader, limit, **kwargs):
        requests.append((query, minecraft_version, loader, limit))
        return [
            ("", float(5 - i), {
                "source_id": f"curseforge:{i}",
                "catalog_id": i,
                "title": f"Spaceship {i}",
                "summary": "Pilot a spaceship",
                "catalog_url": f"https://www.curseforge.com/minecraft/mc-mods/ship-{i}",
            }) for i in range(1, 5)
        ]

    monkeypatch.setattr(reuse_discovery, "_search_curseforge", search)
    diagnostics = []
    result = reuse_discovery.discover_repositories_for_graph(
        ["spaceship", "spaceship"], None,
        minecraft_version="1.21.8", loader="fabric",
        diagnostics=diagnostics,
    )
    assert result == {"spaceship": ()}
    assert len(requests) == 1
    assert requests[0][1:] == ("1.21.8", "fabric", 6)
    assert len(diagnostics[0]["ranked_catalogue_hits"]) == 3
    assert diagnostics[0]["queries"] == [requests[0][0]]
    assert diagnostics[0]["catalog_scanned_count"] == 4


def test_predesign_search_stops_after_first_shortlist(monkeypatch):
    monkeypatch.delenv("MMM_PREDESIGN_PROVIDER_RESULTS_PER_QUERY", raising=False)
    monkeypatch.delenv("MMM_PREDESIGN_PROVIDER_SEARCH_PAGES", raising=False)
    assert pre_design_grounded_rag._provider_result_limit() is None
    assert pre_design_grounded_rag._provider_page_limit() is None
    monkeypatch.setattr(pre_design_grounded_rag, "curseforge_api_key", lambda: "test")
    search_requests = []
    detail_requests = []

    def get_catalog(url, headers):
        if "/categories?" in url:
            return {"data": [{
                "id": 6, "gameId": 432, "name": "Mods",
                "slug": "mc-mods", "isClass": True,
            }]}
        if "/mods/search?" in url:
            q = parse_qs(urlsplit(url).query)
            search_requests.append(q)
            assert q["pageSize"] == ["8"]
            assert q["gameVersion"] == ["1.21.8"]
            assert q["modLoaderType"] == ["4"]
            assert q["classId"] == ["6"]
            assert q["searchFilter"][0]
            return {"data": [
                {"id": i, "classId": 6, "name": f"Spaceship {i}",
                 "summary": "Space ship piloting", "links": {}}
                for i in range(100, 104)
            ], "pagination": {
                "index": 0, "pageSize": 8, "resultCount": 4, "totalCount": 1000,
            }}
        raise AssertionError(url)

    def detail(url, *args, **kwargs):
        detail_requests.append(url)
        return {"data": "Spaceship piloting and construction."}

    monkeypatch.setattr(pre_design_grounded_rag, "_curseforge_browse_json", get_catalog)
    monkeypatch.setattr(pre_design_grounded_rag, "_json", detail)
    records, receipt = pre_design_grounded_rag._search_curseforge(
        "spaceship", minecraft_version="1.21.8", loader="fabric",
    )
    assert len(search_requests) == 1
    assert len(detail_requests) == 2
    assert len(records) == 2
    assert len(receipt["search_variants"]) == 1
    assert receipt["search_requests"] == 1
    assert receipt["retrieval_complete"] is False


def test_multifeature_request_checks_each_distinct_topic(monkeypatch):
    monkeypatch.setattr(reuse_discovery, "_curseforge_api_key", lambda: "test")
    calls = []

    def search(query, *, minecraft_version, loader, **kwargs):
        calls.append(query)
        assert (minecraft_version, loader) == ("1.21.8", "fabric")
        return [
            ("", 10.0, {
                "source_id": f"curseforge:{query}-{i}",
                "catalog_id": i, "title": f"{query} {i}",
                "summary": f"Mod for {query}",
                "catalog_url": f"https://www.curseforge.com/minecraft/mc-mods/{query}-{i}",
            }) for i in (1, 2)
        ]

    monkeypatch.setattr(reuse_discovery, "_search_curseforge", search)
    receipts = []
    reuse_discovery.discover_repositories_for_graph(
        ["spaceship weapons crew"], None,
        minecraft_version="1.21.8", loader="fabric", diagnostics=receipts,
    )
    assert calls == ["spaceship", "weapons", "crew"]
    assert len(receipts[0]["ranked_catalogue_hits"]) == 6
    assert receipts[0]["queries"] == calls


def test_predesign_reviews_each_topic_before_extra_details(monkeypatch):
    monkeypatch.setattr(pre_design_grounded_rag, "curseforge_api_key", lambda: "test")
    searches = []
    descriptions = []

    def catalog(url, headers):
        if "/categories?" in url:
            return {"data": [{"id": 6, "gameId": 432, "name": "Mods",
                              "slug": "mc-mods", "isClass": True}]}
        if "/mods/search?" in url:
            qs = parse_qs(urlsplit(url).query)
            topic = qs["searchFilter"][0]
            searches.append(topic)
            mod_start = {"spaceship": 100, "weapons": 200, "crew": 300}[topic]
            return {"data": [
                {"id": mod_start + i, "classId": 6, "name": f"{topic} {i}",
                 "summary": f"Provides {topic}", "links": {}} for i in (1, 2)
            ], "pagination": {
                "index": 0, "pageSize": int(qs["pageSize"][0]),
                "resultCount": 2, "totalCount": 2,
            }}
        raise AssertionError(url)

    def describe(url, *args, **kwargs):
        descriptions.append(url)
        return {"data": "<p>Mod details.</p>"}

    monkeypatch.setattr(pre_design_grounded_rag, "_curseforge_browse_json", catalog)
    monkeypatch.setattr(pre_design_grounded_rag, "_json", describe)
    records, receipt = pre_design_grounded_rag._search_curseforge(
        "spaceship weapons crew", minecraft_version="1.21.8", loader="fabric",
    )
    assert searches == ["spaceship", "weapons", "crew"]
    assert len(records) == 6
    assert receipt["search_requests"] == 3
    assert len(descriptions) == 3
    assert {u.split("/mods/")[1].split("/")[0] for u in descriptions} == {
        "101", "201", "301",
    }
