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


def test_curseforge_lane_uses_host_key_and_resolves_github_source(monkeypatch):
    monkeypatch.setenv("MMM_CURSEFORGE_API_KEY", "secret-test-key")
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)
    observed = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"links": {"sourceUrl": "https://github.com/example/trade-mod"}}]}

    class FakeClient:
        def get(self, url, **kwargs):
            observed["url"] = url
            observed["headers"] = dict(kwargs["headers"])
            return Response()

    monkeypatch.setattr(reuse_discovery, "_pooled_http_client", lambda: FakeClient())
    assert reuse_discovery._search_curseforge("trade system", limit=8) == [("example/trade-mod", 1.0)]
    assert observed["url"] == "https://api.curseforge.com/v1/mods/search"
    assert observed["headers"]["x-api-key"] == "secret-test-key"
