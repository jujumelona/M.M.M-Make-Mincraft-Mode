"""Regression checks for provider search semantics and evidence budgets."""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import pytest

from minecraft_mod_ai import grounded_source_reuse, pre_design_grounded_rag, reuse_discovery
from minecraft_mod_ai.curseforge_catalog import (
    curseforge_browse_windows,
    curseforge_mods_class_id,
    curseforge_next_index,
)


def test_curseforge_mod_class_resolved_from_live_category_response():
    assert curseforge_mods_class_id([
        {"id": 6, "gameId": 432, "name": "Mods", "slug": "mc-mods", "isClass": True},
        {"id": 10, "gameId": 432, "name": "Modpacks", "slug": "modpacks", "isClass": True},
    ]) == 6
    assert curseforge_mods_class_id([
        {"id": 6, "gameId": 999, "name": "Mods", "slug": "mods", "isClass": True},
    ]) is None


def test_one_browse_plan_shared_by_reuse_and_predesign():
    categories = [
        {"id": 6, "gameId": 432, "name": "Mods",
         "slug": "mc-mods", "isClass": True},
        {"id": 24, "gameId": 432, "name": "World Gen",
         "isClass": False, "classId": 6},
        {"id": 25, "gameId": 432, "name": "Technology",
         "isClass": False, "classId": 6},
        {"id": 91, "gameId": 999, "isClass": False, "classId": 6},
        {"id": 94, "gameId": 432, "isClass": False, "classId": 7},
    ]
    assert curseforge_browse_windows(categories, 6) == (
        (2, None, 2), (3, None, 2), (11, None, 2),
        (3, 24, 1), (3, 25, 1),
    )


def test_provider_rank_fusion_does_not_multiply_repeated_single_provider_hits():
    single = {"catalog_evidence": [
        {"provider": "curseforge", "rank_position": 2},
        {"provider": "curseforge", "rank_position": 2},
    ]}
    multi = {"catalog_evidence": [
        {"provider": "curseforge", "rank_position": 2},
        {"provider": "sourcegraph", "rank_position": 7},
    ]}
    assert grounded_source_reuse._candidate_provider_rrf(single) == 1.0 / 62
    assert grounded_source_reuse._candidate_provider_rrf(multi) > (
        grounded_source_reuse._candidate_provider_rrf(single)
    )


def test_curseforge_pagination_advances_and_never_crosses_provider_window():
    assert curseforge_next_index(
        {"index": 0, "pageSize": 50, "resultCount": 50, "totalCount": 200},
        index=0, page_size=50, row_count=50,
    ) == 50
    assert curseforge_next_index(
        {"index": 0, "pageSize": 50, "resultCount": 4, "totalCount": 200},
        index=0, page_size=50, row_count=4,
    ) == 4
    assert curseforge_next_index(
        {"index": 9950, "pageSize": 50, "resultCount": 50, "totalCount": 12000},
        index=9950, page_size=50, row_count=50,
    ) is None
    with pytest.raises(ValueError):
        curseforge_next_index(
            {"index": 0, "pageSize": 50, "resultCount": 51, "totalCount": 100},
            index=0, page_size=50, row_count=50,
        )


def test_curseforge_reuse_discovers_second_page_without_name_search(monkeypatch):
    monkeypatch.setattr(reuse_discovery, "_curseforge_api_key", lambda: "dummy")
    monkeypatch.setattr(reuse_discovery, "_discovery_pages", lambda: 2)
    calls = []

    class Response:
        def __init__(self, url, params):
            self.url, self.params = url, params

        def raise_for_status(self):
            pass

        def json(self):
            if self.url.endswith("/categories"):
                return {"data": [
                    {"gameId": 432, "name": "Mods", "slug": "mc-mods",
                     "isClass": True, "id": 6},
                ]}
            position = int(self.params["index"])
            sort = int(self.params["sortField"])
            if sort == 2 and position == 0:
                mods = [{"id": i, "classId": 6, "name": f"Popular Mod {i}",
                         "links": {}} for i in range(1, 51)]
                total = 51
            elif sort == 2 and position == 50:
                mods = [{"id": 51, "classId": 6, "name": "Hidden Donor",
                         "summary": "Spacecraft assembly",
                         "links": {"sourceUrl": "https://github.com/owner/hidden-donor"}}]
                total = 51
            else:
                mods = []
                total = 0
            return {"data": mods, "pagination": {
                "index": position, "pageSize": 50,
                "resultCount": len(mods), "totalCount": total,
            }}

    class Client:
        def get(self, url, **kwargs):
            calls.append((url, dict(kwargs.get("params") or {})))
            return Response(url, kwargs.get("params") or {})

    monkeypatch.setattr(reuse_discovery, "_pooled_http_client", lambda: Client())
    results = reuse_discovery._search_curseforge(
        "build spaceship and fight aliens", limit=8,
        minecraft_version="1.21.1", loader="fabric", with_metadata=True,
    )
    assert any(repo == "" and row["source_id"] == "curseforge:51" for repo, _, row in results)
    searches = [params for url, params in calls if "/mods/search" in url]
    assert searches
    assert all("searchFilter" not in params for params in searches)
    assert all(params.get("gameVersion") == "1.21.1" and
               params.get("modLoaderType") == "4" for params in searches)
    assert all(params.get("classId") == "6" for params in searches)
    assert any(params["sortField"] == "2" and params["index"] == "50"
               for params in searches)



def test_later_curseforge_failure_keeps_completed_page_candidates(monkeypatch):
    monkeypatch.setattr(reuse_discovery, "_curseforge_api_key", lambda: "dummy")
    monkeypatch.setattr(reuse_discovery, "_discovery_pages", lambda: 2)

    class Response:
        def __init__(self, url, params):
            self.url, self.params = url, params

        def raise_for_status(self):
            pass

        def json(self):
            if "/categories" in self.url:
                return {"data": [
                    {"id": 6, "gameId": 432, "name": "Mods",
                     "slug": "mc-mods", "isClass": True},
                ]}
            if int(self.params["index"]) != 0:
                return {"data": None}  # Malformed second page.
            return {
                "data": [
                    {"id": 51, "classId": 6, "name": "Spaceship",
                     "summary": "Ship implementation",
                     "links": {"sourceUrl": "https://github.com/owner/ship"}},
                    *({"id": i, "classId": 6, "name": f"Mod {i}", "links": {}}
                      for i in range(52, 101)),
                ],
                "pagination": {"index": 0, "resultCount": 50,
                               "pageSize": 50, "totalCount": 51},
            }

    class Client:
        def get(self, url, **kwargs):
            return Response(url, kwargs.get("params") or {})

    monkeypatch.setattr(reuse_discovery, "_pooled_http_client", lambda: Client())
    errors = []
    found = reuse_discovery._search_curseforge(
        "spaceship", limit=8, with_metadata=True, errors=errors,
    )
    assert found and found[0][0] == ""
    assert found[0][2]["source_id"] == "curseforge:51"
    assert errors and errors[0]["stage"] == "later_page"


def test_predesign_browses_metadata_before_budgeted_descriptions(monkeypatch):
    monkeypatch.setattr(pre_design_grounded_rag, "curseforge_api_key", lambda: "dummy")
    monkeypatch.setattr(pre_design_grounded_rag, "_provider_page_limit", lambda: 1)
    monkeypatch.setattr(pre_design_grounded_rag, "_provider_result_limit", lambda: None)
    calls = []

    def fake_json(url, headers=None):
        calls.append(url)
        if "/categories?" in url:
            return {"data": [
                {"gameId": 432, "name": "Mods", "slug": "mc-mods",
                 "isClass": True, "id": 6},
            ]}
        if url.endswith("/description"):
            return {"data": "<p>Source-backed spaceship mechanic.</p>"}
        if "/mods/search?" in url:
            params = parse_qs(urlsplit(url).query)
            assert "searchFilter" not in params
            assert params.get("classId") == ["6"]
            assert params.get("gameVersion") == ["1.21.1"]
            assert params.get("modLoaderType") == ["4"]
            return {"data": [
                {"id": i, "classId": 6, "name": f"Spaceship {i}",
                 "summary": "Spaceship engine and launch",
                 "links": {"sourceUrl": f"https://github.com/example/spaceship-{i}"}}
                for i in range(1, 11)
            ], "pagination": {
                "index": 0, "pageSize": 50, "resultCount": 10, "totalCount": 10,
            }}
        raise AssertionError(url)

    monkeypatch.setattr(pre_design_grounded_rag, "_json", fake_json)
    records, receipt = pre_design_grounded_rag._search_curseforge(
        "spaceship", minecraft_version="1.21.1", loader="fabric"
    )
    assert len(records) == 10
    assert receipt["search_requests"] == 3
    assert receipt["source_requests"] == 6
    assert sum("/description" in url for url in calls) == 6
    assert sum(bool(row["body_retrieved"]) for row in records) == 6
    assert all(row["evidence_origin"].startswith("curseforge_mod_") for row in records)


def test_curseforge_unknown_mod_class_fails_closed(monkeypatch):
    monkeypatch.setattr(reuse_discovery, "_curseforge_api_key", lambda: "dummy")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    class Client:
        def get(self, url, **kwargs):
            assert "/categories" in url, "Must not browse unscoped Modpacks/Resourcepacks"
            return Response()

    monkeypatch.setattr(reuse_discovery, "_pooled_http_client", lambda: Client())
    with pytest.raises(RuntimeError, match="Mods class"):
        reuse_discovery._search_curseforge("trade", limit=8)
