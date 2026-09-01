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


def test_curseforge_paginates_and_fetches_all_descriptions(monkeypatch):
    def fake_json(url, headers=None):
        if "/mods/search?" in url:
            index = int(_query(url).get("index", ["0"])[0])
            if index == 0:
                return {"data": [
                    {"id": 1, "name": "one", "summary": "s1", "links": {}},
                    {"id": 2, "name": "two", "summary": "s2", "links": {}},
                ], "pagination": {"index": 0, "pageSize": 50, "resultCount": 2, "totalCount": 3}}
            if index == 2:
                return {"data": [
                    {"id": 3, "name": "three", "summary": "s3", "links": {}},
                ], "pagination": {"index": 2, "pageSize": 50, "resultCount": 1, "totalCount": 3}}
        if "/description" in url:
            mod_id = url.split("/mods/", 1)[1].split("/", 1)[0]
            return {"data": f"<p>full-{mod_id}</p>"}
        raise AssertionError(url)

    monkeypatch.setenv("CURSEFORGE_API_KEY", "test")
    monkeypatch.setattr(rag, "_json", fake_json)
    records, receipt = rag._search_curseforge("space trading")
    assert [record["source_id"] for record in records] == ["curseforge:1", "curseforge:2", "curseforge:3"]
    assert [record["content"] for record in records] == ["full-1", "full-2", "full-3"]
    assert receipt["search_requests"] == 2
    assert receipt["source_requests"] == 3
    assert receipt["provider_total"] == 3


def test_github_fallback_paginates_and_fetches_all_readmes(monkeypatch):
    def fake_json(url, headers=None):
        if "/search/repositories?" not in url:
            raise AssertionError(url)
        page = int(_query(url).get("page", ["1"])[0])
        if page == 1:
            return {"total_count": 3, "items": [
                {"full_name": "x/one", "name": "one", "description": "s1", "html_url": "https://github.com/x/one", "default_branch": "main"},
                {"full_name": "x/two", "name": "two", "description": "s2", "html_url": "https://github.com/x/two", "default_branch": "main"},
            ]}
        if page == 2:
            return {"total_count": 3, "items": [
                {"full_name": "x/three", "name": "three", "description": "s3", "html_url": "https://github.com/x/three", "default_branch": "main"},
            ]}
        return {"total_count": 3, "items": []}

    monkeypatch.setattr(rag, "_json", fake_json)
    monkeypatch.setattr(rag, "_text", lambda url, headers=None: "README-" + url.split("/repos/", 1)[1].split("/readme", 1)[0])
    records, receipt = rag._search_github("space trading")
    assert [record["source_id"] for record in records] == ["github:x/one", "github:x/two", "github:x/three"]
    assert [record["content"] for record in records] == ["README-x/one", "README-x/two", "README-x/three"]
    assert receipt["search_requests"] == 2
    assert receipt["source_requests"] == 3
    assert receipt["provider_total"] == 3
