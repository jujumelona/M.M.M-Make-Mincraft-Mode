from __future__ import annotations

from minecraft_mod_ai import reuse_discovery


def test_curseforge_lane_is_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)
    called = False

    def fail_pool():
        nonlocal called
        called = True
        raise AssertionError("CurseForge HTTP pool must not run without a key")

    monkeypatch.setattr(reuse_discovery, "_pooled_http_client", fail_pool)
    assert reuse_discovery._search_curseforge("trade system", limit=8) == []
    assert called is False


def test_curseforge_lane_uses_host_key_without_following_github_source(monkeypatch):
    monkeypatch.setenv("MMM_CURSEFORGE_API_KEY", "secret-test-key")
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)
    observed = {}

    class Response:
        def raise_for_status(self):
            return None

        def __init__(self, url):
            self.url = url

        def json(self):
            if "/categories" in self.url:
                return {"data": [{"id": 6, "gameId": 432, "name": "Mods", "slug": "mc-mods", "isClass": True}]}
            return {
                "data": [{"id": 99, "classId": 6, "name": "Trade Mod",
                          "links": {"sourceUrl": "https://github.com/example/trade-mod"}}],
                "pagination": {"index": 0, "pageSize": 50, "resultCount": 1, "totalCount": 1},
            }

    class FakeClient:
        def get(self, url, **kwargs):
            if "/mods/search" in url:
                observed["url"] = url
                observed["headers"] = dict(kwargs["headers"])
                observed["params"] = dict(kwargs["params"])
            return Response(url)

    monkeypatch.setattr(reuse_discovery, "_pooled_http_client", lambda: FakeClient())
    assert reuse_discovery._search_curseforge("trade system", limit=8) == []
    records = reuse_discovery._search_curseforge("trade system", limit=8, with_metadata=True)
    assert records[0][0] == ""
    assert records[0][2]["source_id"] == "curseforge:99"
    assert observed["url"] == "https://api.curseforge.com/v1/mods/search"
    assert observed["headers"]["x-api-key"] == "secret-test-key"
    assert "searchFilter" not in observed["params"]
    assert observed["params"]["classId"] == "6"
