"""CurseForge-only pre-design discovery contract."""
from minecraft_mod_ai.catalog_first_grounded_rag import _query_bundle

class Backend:
    def __init__(self):
        self.calls = []
    def _search_curseforge(self, query, **kwargs):
        self.calls.append(("curseforge", kwargs))
        return [{"source_id": "curseforge:3", "title": "Example Mod", "content": "description"}], {
            "provider": "curseforge", "status": "available", "result_count": 1,
        }
    def _versions(self, router):
        return ()
    def _search_authoritative_catalog(self, query, versions):
        return {"sources": [], "errors": []}
    def _existing_code_index(self):
        return None
    def _search_code_index(self, index, query):
        return {"hits": []}
    def _sha256_text(self, value):
        return "sha256:demo"
    def _error(self, provider, exc):
        return {"provider": provider, "status": "error", "error": str(exc)}

def test_only_curseforge_is_requested(monkeypatch):
    monkeypatch.setenv("MMM_CURSEFORGE_API_KEY", "test-key")
    backend = Backend()
    bundle = _query_bundle(backend, None, "space", ("curseforge", "modrinth", "github"))
    assert [provider for provider, _ in backend.calls] == ["curseforge"]
    assert list(bundle["external_rag"]["providers"]) == ["curseforge"]
    assert [r["source_id"] for r in bundle["external_rag"]["sources"]] == ["curseforge:3"]

def test_no_other_provider_used_without_curseforge_key(monkeypatch):
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)
    backend = Backend()
    bundle = _query_bundle(backend, None, "space", ("curseforge", "modrinth", "github"))
    assert backend.calls == []
    assert bundle["external_rag"]["providers"]["curseforge"]["status"] == "not_configured"
