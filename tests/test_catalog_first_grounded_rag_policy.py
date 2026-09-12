from __future__ import annotations

import hashlib

from minecraft_mod_ai.catalog_first_grounded_rag import _query_bundle


class _Backend:
    def __init__(self, *, catalog_records):
        self.catalog_records = list(catalog_records)
        self.linked_calls = 0
        self.broad_github_calls = 0

    def _search_curseforge(self, query):
        return list(self.catalog_records), {
            "provider": "curseforge",
            "status": "available",
            "result_count": len(self.catalog_records),
        }

    def _search_modrinth(self, query):
        return [], {"provider": "modrinth", "status": "available", "result_count": 0}

    def _linked_github_sources(self, records, *, disabled, disable):
        self.linked_calls += 1
        return [], {
            "provider": "github",
            "status": "skipped_no_linked_source",
            "result_count": 0,
            "search_requests": 0,
            "source_requests": 0,
        }

    def _search_github(self, query, *, disabled, disable):
        self.broad_github_calls += 1
        return [], {
            "provider": "github",
            "status": "available",
            "result_count": 0,
            "search_requests": 1,
            "source_requests": 0,
        }

    def _versions(self, router):
        return ()

    def _search_authoritative_catalog(self, query, versions):
        return {"schema_version": "test", "sources": [], "errors": []}

    def _existing_code_index(self):
        return None

    def _search_code_index(self, index, query):
        return {"schema_version": "test", "status": "not_indexed", "hits": []}

    @staticmethod
    def _sha256_text(value):
        return "sha256:" + hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def _error(provider, exc):
        return {"provider": provider, "status": "error", "error": str(exc)}


def _bundle(backend):
    return _query_bundle(
        backend,
        object(),
        "energy automation",
        ("curseforge", "modrinth"),
        github_disabled=lambda: False,
        disable_github=lambda: None,
    )


def test_catalog_candidate_without_link_uses_source_discovery_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CURSEFORGE_API_KEY", "configured-test-key")
    backend = _Backend(
        catalog_records=[
            {
                "source_id": "curseforge:1",
                "url": "https://www.curseforge.com/minecraft/mc-mods/example",
                "metadata": {"source_url": ""},
            }
        ]
    )

    row = _bundle(backend)

    assert backend.linked_calls == 1
    assert backend.broad_github_calls == 1
    assert row["external_rag"]["providers"]["github"]["policy"] == (
        "catalog_candidate_source_discovery_fallback"
    )


def test_empty_catalog_uses_broad_github_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CURSEFORGE_API_KEY", "configured-test-key")
    backend = _Backend(catalog_records=[])

    row = _bundle(backend)

    assert backend.linked_calls == 0
    assert backend.broad_github_calls == 1
    assert row["external_rag"]["providers"]["github"]["policy"] == "catalog_empty_fallback"
