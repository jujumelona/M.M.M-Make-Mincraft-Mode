from __future__ import annotations

import json
import urllib.parse

from minecraft_mod_ai import pre_design_grounded_rag as rag


def _query(url: str) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)


def test_modrinth_paginates_and_bulk_fetches_full_projects(monkeypatch):
    calls: list[str] = []

    def fake_json(url, headers=None):
        calls.append(url)
        if "/v2/search?" in url:
            offset = int(_query(url).get("offset", ["0"])[0])
            if offset == 0:
                return {"offset": 0, "total_hits": 3, "hits": [
                    {"project_id": "a", "slug": "a", "title": "A", "description": "summary-a", "versions": ["1"]},
                    {"project_id": "b", "slug": "b", "title": "B", "description": "summary-b", "versions": ["1"]},
                ]}
            if offset == 2:
                return {"offset": 2, "total_hits": 3, "hits": [
                    {"project_id": "c", "slug": "c", "title": "C", "description": "summary-c", "versions": ["2"]},
                ]}
            return {"offset": offset, "total_hits": 3, "hits": []}
        if "/v2/projects?" in url:
            ids = json.loads(_query(url)["ids"][0])
            return [
                {"id": project_id, "slug": project_id, "title": project_id.upper(), "body": f"full-{project_id}", "source_url": f"https://github.com/x/{project_id}", "game_versions": ["9"]}
                for project_id in ids
            ]
        raise AssertionError(url)

    monkeypatch.setattr(rag, "_json", fake_json)
    records, receipt = rag._search_modrinth("space trading")
    assert [record["source_id"] for record in records] == ["modrinth:a", "modrinth:b", "modrinth:c"]
    assert [record["content"] for record in records] == ["full-a", "full-b", "full-c"]
    assert all(record["metadata"]["versions"] == ["9"] for record in records)
    assert receipt["search_requests"] == 2
    assert receipt["source_requests"] == 2
    assert receipt["provider_total"] == 3
    assert sum("/v2/projects?" in call for call in calls) == 2


def test_curseforge_paginates_targeted_queries_and_fetches_ranked_descriptions(monkeypatch):
    calls = []

    def fake_json(url, headers=None):
        calls.append(url)
        if "/categories?" in url:
            return {"data": [
                {"id": 6, "gameId": 432, "name": "Mods", "slug": "mc-mods", "isClass": True},
            ]}
        if "/mods/search?" in url:
            query = _query(url)
            assert query.get("searchFilter", [""])[0] in {"space", "trading", "trade"}
            assert query["classId"] == ["6"]
            index = int(query.get("index", ["0"])[0])
            if index == 0:
                return {"data": [
                    {"id": 1, "classId": 6, "name": "space one", "summary": "space trading", "links": {}},
                    {"id": 2, "classId": 6, "name": "space two", "summary": "space trading", "links": {}},
                ], "pagination": {"index": 0, "pageSize": int(query["pageSize"][0]), "resultCount": 2, "totalCount": 3}}
            if index == 2:
                return {"data": [
                    {"id": 3, "classId": 6, "name": "space three", "summary": "space trading", "links": {}},
                ], "pagination": {"index": 2, "pageSize": int(query["pageSize"][0]), "resultCount": 1, "totalCount": 3}}
        if "/description" in url:
            mod_id = url.split("/mods/", 1)[1].split("/", 1)[0]
            return {"data": f"<p>full-{mod_id}</p>"}
        raise AssertionError(url)

    monkeypatch.setenv("CURSEFORGE_API_KEY", "test")
    monkeypatch.setattr(rag, "_json", fake_json)
    monkeypatch.setattr(rag, "_provider_page_limit", lambda: 2)
    records, receipt = rag._search_curseforge("space trading")
    assert [record["source_id"] for record in records] == ["curseforge:1", "curseforge:2"]
    assert [record["content"] for record in records] == ["full-1", "full-2"]
    # Stop after the first keyword produced three plausible candidates.
    assert receipt["search_requests"] == 4  # two pages for space, two for trading
    assert receipt["search_variants"] == ["space", "trading"]
    assert receipt["source_requests"] == 2
    assert receipt["provider_total"] == 3


