"""Policy regression: external mod and donor provider interfaces remain absent."""
from minecraft_mod_ai.catalog_first_grounded_rag import _domain_specs, _query_bundle


class Backend:
    def __init__(self):
        self.requested = []

    def _versions(self, router):
        return (getattr(router, "minecraft_version", "26.2"),)

    def _search_authoritative_catalog(self, query, versions):
        self.requested.append(("api", query, versions))
        return {"sources": [], "errors": []}

    def _existing_code_index(self):
        return None

    def _search_code_index(self, index, query):
        self.requested.append(("project", query))
        return {"hits": []}

    def _sha256_text(self, query):
        return "sha256:query"


def test_target_api_and_local_project_not_external_catalogue():
    backend = Backend()
    row = _query_bundle(backend, object(), "spacecraft engine", ("curseforge", "github"))
    assert row["external_rag"]["providers"] == {}
    assert row["external_rag"]["sources"] == []
    assert [v[0] for v in backend.requested] == ["api", "project"]


def test_local_and_official_queries_preserve_intent():
    assert _domain_specs({"providers": ["modrinth"], "queries": ["energy control"]}) == [
        ("energy control", ("official_docs", "project_rag"))
    ]
