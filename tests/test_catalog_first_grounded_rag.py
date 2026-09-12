from __future__ import annotations

from typing import Any

from minecraft_mod_ai.catalog_first_grounded_rag import _query_bundle


class _Backend:
    _MAX_QUERY_WORKERS = 2

    def __init__(self, *, catalog_records=None, linked_records=None, github_records=None):
        self.catalog_records = list(catalog_records or [])
        self.linked_records = list(linked_records or [])
        self.github_records = list(github_records or [])
        self.calls: list[str] = []

    def _search_curseforge(self, query):
        self.calls.append("curseforge")
        return list(self.catalog_records), {
            "provider": "curseforge",
            "status": "available",
            "result_count": len(self.catalog_records),
        }

    def _search_modrinth(self, query):
        self.calls.append("modrinth")
        return [], {"provider": "modrinth", "status": "available", "result_count": 0}

    def _linked_github_sources(self, records, *, disabled, disable):
        self.calls.append("github_linked")
        return list(self.linked_records), {
            "provider": "github",
            "status": "available" if self.linked_records else "skipped_no_linked_source",
            "result_count": len(self.linked_records),
            "search_requests": 0,
            "source_requests": len(records),
        }

    def _search_github(self, query, *, disabled, disable):
        self.calls.append("github_broad")
        return list(self.github_records), {
            "provider": "github",
            "status": "available",
            "result_count": len(self.github_records),
            "search_requests": 1,
            "source_requests": len(self.github_records),
        }

    @staticmethod
    def _error(provider, exc):
        return {"provider": provider, "status": "error", "error": str(exc)}

    @staticmethod
    def _versions(router):
        return ()

    @staticmethod
    def _search_authoritative_catalog(query, versions):
        return {"sources": [], "errors": []}

    @staticmethod
    def _existing_code_index():
        return None

    @staticmethod
    def _search_code_index(index, query):
        return {"status": "not_indexed", "hits": []}

    @staticmethod
    def _sha256_text(value):
        return "sha256:" + "a" * 64


def _catalog_record(source_url=""):
    return {
        "source_id": "curseforge:1",
        "url": "https://www.curseforge.com/minecraft/mc-mods/example",
        "title": "Example Mod",
        "content": "actual mod description",
        "metadata": {"source_url": source_url},
    }


def _github_record():
    return {
        "source_id": "github:owner/example",
        "url": "https://github.com/owner/example",
        "title": "example",
        "content": "repository readme",
    }


def _query(backend: _Backend) -> dict[str, Any]:
    return _query_bundle(
        backend,
        None,
        "spaceship progression implementation",
        ("curseforge", "modrinth", "github"),
        github_disabled=lambda: False,
        disable_github=lambda: None,
    )


def test_catalog_candidate_without_source_link_uses_broad_github_source_discovery(monkeypatch) -> None:
    monkeypatch.setenv("CURSEFORGE_API_KEY", "configured-test-key")
    backend = _Backend(catalog_records=[_catalog_record()])

    row = _query(backend)

    assert backend.calls[:2] == ["curseforge", "modrinth"] or set(backend.calls[:2]) == {"curseforge", "modrinth"}
    assert "github_linked" in backend.calls
    assert "github_broad" in backend.calls
    github = row["external_rag"]["providers"]["github"]
    assert github["status"] == "available"
    assert github["policy"] == "catalog_candidate_source_discovery_fallback"


def test_catalog_linked_source_uses_exact_github_repository_only(monkeypatch) -> None:
    monkeypatch.setenv("CURSEFORGE_API_KEY", "configured-test-key")
    backend = _Backend(
        catalog_records=[_catalog_record("https://github.com/owner/example")],
        linked_records=[_github_record()],
    )

    row = _query(backend)

    assert "github_linked" in backend.calls
    assert "github_broad" not in backend.calls
    assert any(
        source["source_id"] == "github:owner/example"
        for source in row["external_rag"]["sources"]
    )
    assert row["external_rag"]["providers"]["github"]["policy"] == "exact_catalog_link_only"


def test_broad_github_is_only_source_discovery_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CURSEFORGE_API_KEY", "configured-test-key")
    backend = _Backend(catalog_records=[], github_records=[_github_record()])

    row = _query(backend)

    assert "github_broad" in backend.calls
    assert row["external_rag"]["providers"]["github"]["policy"] == "catalog_empty_fallback"


def test_curseforge_key_enables_authenticated_catalog_role(monkeypatch) -> None:
    monkeypatch.setenv("CURSEFORGE_API_KEY", "configured-test-key")
    backend = _Backend(catalog_records=[_catalog_record()])

    _query(backend)

    assert "curseforge" in backend.calls


def test_missing_curseforge_key_keeps_modrinth_and_does_not_fake_configuration(monkeypatch) -> None:
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)
    backend = _Backend(catalog_records=[_catalog_record()])

    row = _query(backend)

    assert "curseforge" not in backend.calls
    assert "modrinth" in backend.calls
    assert row["external_rag"]["providers"]["curseforge"]["status"] == "not_configured"
