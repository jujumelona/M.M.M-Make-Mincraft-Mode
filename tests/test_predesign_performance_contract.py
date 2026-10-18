from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minecraft_mod_ai import catalog_first_grounded_rag as catalog_rag
from minecraft_mod_ai import pre_design_grounded_rag as rag


def test_deleted_github_source_transport_cannot_be_called():
    assert not hasattr(rag, "_linked_github_sources")


def test_source_worker_count_respects_operator_override(monkeypatch):
    monkeypatch.setenv("MMM_PREDESIGN_SOURCE_WORKERS", "3")
    assert rag._source_worker_count(1) == 1
    assert rag._source_worker_count(8) == 3


def test_identical_evidence_write_skips_second_fsync(monkeypatch, tmp_path):
    target = tmp_path / "evidence.json"
    rag._write(target, "same payload")

    def explode(_fd: int) -> None:
        raise AssertionError("identical content must not be rewritten or fsynced")

    monkeypatch.setattr(rag.os, "fsync", explode)
    rag._write(target, "same payload")
    assert target.read_text(encoding="utf-8") == "same payload"


def test_predesign_code_rag_uses_lexical_hot_path(monkeypatch):
    calls: list[dict[str, object]] = []

    @dataclass(frozen=True)
    class Receipt:
        route: str = "generic"

    class Result:
        hits = ()
        receipt = Receipt()

    class FakeIndex:
        def __init__(self, path: Path) -> None:
            self.path = path

        def search_with_receipt(self, query: str, **kwargs):
            calls.append({"query": query, **kwargs})
            return Result()

    monkeypatch.setattr(rag, "ProjectRAGIndex", FakeIndex)
    result = rag._search_code_index(Path("unused.db"), "alien combat behavior")

    assert result["status"] == "searched"
    assert calls == [
        {
            "query": "alien combat behavior",
            "limit": 3,
            "semantic": False,
            "rerank": False,
        }
    ]


def test_external_mod_requests_only_use_local_and_official_rag(monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "_versions", lambda _router: ())
    monkeypatch.setattr(
        rag, "_search_authoritative_catalog",
        lambda query, versions: {"sources": [{"query": query}], "errors": []},
    )
    monkeypatch.setattr(rag, "_existing_code_index", lambda: None)
    monkeypatch.setattr(
        rag, "_search_code_index",
        lambda index, query: {"status": "searched", "hits": []},
    )
    monkeypatch.setattr(rag, "_query_worker_count", lambda count: 1)
    brief = {
        "domains": [{
            "domain_id": "request",
            "providers": ["curseforge", "modrinth"],
            "queries": ["query one", "query two"],
        }]
    }
    bundle = catalog_rag.forced_rag_bundle(rag, object(), brief)
    rows = bundle["domains"][0]["queries"]
    assert len(rows) == 2
    assert [row["project_rag"]["sources"][0]["query"] for row in rows] == [
        "query one", "query two",
    ]
    assert all(row["external_rag"]["sources"] == [] for row in rows)
