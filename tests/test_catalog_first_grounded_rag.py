"""Only selected-target official API and current-project RAG may be queried."""
from minecraft_mod_ai.catalog_first_grounded_rag import _query_bundle, _domain_specs


class Backend:
    def __init__(self):
        self.calls = []

    def _versions(self, router):
        self.calls.append(("versions", router))
        return ("26.2",)

    def _search_authoritative_catalog(self, query, versions):
        self.calls.append(("official", query, versions))
        return {"sources": [{"source_id": "official-api"}], "errors": []}

    def _existing_code_index(self):
        return "local-index"

    def _search_code_index(self, index, query):
        self.calls.append(("local", index, query))
        return {"hits": [{"symbol": "demo.Local"}]}

    def _sha256_text(self, value):
        return "sha256:query"


def test_external_mod_provider_requests_never_trigger_catalogue_search():
    backend = Backend()
    result = _query_bundle(backend, None, "spacecraft", ("curseforge", "modrinth", "github"))
    assert result["external_rag"]["sources"] == []
    assert result["external_rag"]["providers"] == {}
    assert result["project_rag"]["sources"][0]["source_id"] == "official-api"
    assert result["code_rag"]["hits"][0]["symbol"] == "demo.Local"
    assert not any(row[0] == "external" for row in backend.calls)


def test_domain_specs_filter_old_provider_names_and_empty_queries():
    domain = {
        "providers": ["curseforge", "modrinth", "official_docs", "project_rag"],
        "queries": ["", "spacecraft", "spacecraft", "  "],
    }
    assert _domain_specs(domain) == [
        ("spacecraft", ("official_docs", "project_rag"))
    ]
