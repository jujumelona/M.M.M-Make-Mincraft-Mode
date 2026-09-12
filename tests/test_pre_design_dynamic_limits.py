from __future__ import annotations

import json
import urllib.parse
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import catalog_first_grounded_rag as catalog_rag
from minecraft_mod_ai import pre_design_domain_research as domain_research
from minecraft_mod_ai import pre_design_grounded_rag as grounded_rag


def test_predesign_worker_fanout_is_not_hidden_capped(monkeypatch):
    for name in ("MMM_PREDESIGN_QUERY_WORKERS", "MMM_PREDESIGN_SOURCE_WORKERS"):
        monkeypatch.delenv(name, raising=False)

    assert grounded_rag._query_worker_count(20) == 20
    assert grounded_rag._source_worker_count(32) == 32

    monkeypatch.setenv("MMM_PREDESIGN_QUERY_WORKERS", "11")
    monkeypatch.setenv("MMM_PREDESIGN_SOURCE_WORKERS", "13")
    assert grounded_rag._query_worker_count(20) == 11
    assert grounded_rag._source_worker_count(32) == 13


@pytest.mark.parametrize(
    "name",
    (
        "MMM_PREDESIGN_QUERY_WORKERS",
        "MMM_PREDESIGN_SOURCE_WORKERS",
        "MMM_PREDESIGN_PROVIDER_RESULTS_PER_QUERY",
        "MMM_PREDESIGN_PROVIDER_SEARCH_PAGES",
    ),
)
def test_predesign_explicit_limits_must_be_positive(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError):
        if "QUERY_WORKERS" in name:
            grounded_rag._query_worker_count(2)
        elif "SOURCE_WORKERS" in name:
            grounded_rag._source_worker_count(2)
        elif "RESULTS" in name:
            grounded_rag._provider_result_limit()
        else:
            grounded_rag._provider_page_limit()


def test_provider_limits_are_optional_operator_policy(monkeypatch):
    monkeypatch.delenv("MMM_PREDESIGN_PROVIDER_RESULTS_PER_QUERY", raising=False)
    monkeypatch.delenv("MMM_PREDESIGN_PROVIDER_SEARCH_PAGES", raising=False)
    assert grounded_rag._provider_result_limit() is None
    assert grounded_rag._provider_page_limit() is None

    monkeypatch.setenv("MMM_PREDESIGN_PROVIDER_RESULTS_PER_QUERY", "37")
    monkeypatch.setenv("MMM_PREDESIGN_PROVIDER_SEARCH_PAGES", "9")
    assert grounded_rag._provider_result_limit() == 37
    assert grounded_rag._provider_page_limit() == 9


def test_modrinth_progress_can_cross_old_four_page_limit(monkeypatch):
    monkeypatch.delenv("MMM_PREDESIGN_PROVIDER_RESULTS_PER_QUERY", raising=False)
    monkeypatch.delenv("MMM_PREDESIGN_PROVIDER_SEARCH_PAGES", raising=False)
    terms = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    search_calls: list[int] = []

    def fake_json(url: str, headers=None):
        del headers
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path.endswith("/search"):
            offset = int(query.get("offset", ["0"])[0])
            search_calls.append(offset)
            term = terms[offset]
            return {
                "total_hits": len(terms),
                "offset": offset,
                "hits": [
                    {
                        "project_id": f"p{offset}",
                        "slug": f"p{offset}",
                        "title": term,
                        "description": term,
                        "versions": [],
                    }
                ],
            }
        if parsed.path.endswith("/projects"):
            ids = json.loads(query["ids"][0])
            offset = int(ids[0][1:])
            return [
                {
                    "id": ids[0],
                    "title": terms[offset],
                    "body": f"{terms[offset]} evidence",
                    "game_versions": [],
                    "loaders": [],
                }
            ]
        raise AssertionError(url)

    monkeypatch.setattr(grounded_rag, "_json", fake_json)
    records, receipt = grounded_rag._search_modrinth(" ".join(terms))

    assert len(records) == len(terms)
    assert receipt["search_requests"] == len(terms)
    assert len(search_calls) > 4


def test_grounded_evidence_preserves_more_than_four_cards_and_long_excerpt():
    long_body = "alpha " + ("x" * 1200)
    pages = []
    for index in range(6):
        body = long_body if index == 0 else f"alpha evidence {index}"
        pages.append(
            {
                "page_ref": f"page-{index}",
                "content": json.dumps(
                    {
                        "source_id": f"source-{index}",
                        "source_type": "modrinth_project_body",
                        "url": f"https://example.invalid/{index}",
                        "title": f"Alpha source {index}",
                        "content_sha256": f"sha256:{index}",
                        "content": body,
                    }
                ),
            }
        )

    rag = SimpleNamespace(_read_evidence_pages=lambda document: pages)
    cards = domain_research._grounded_evidence_cards(
        rag,
        {"page_count": len(pages)},
        {"objective": "alpha", "requirements": [], "queries": []},
    )

    assert len(cards) == 6
    assert max(len(card["exact_excerpt"]) for card in cards) > 800


def test_catalog_scheduler_uses_dynamic_query_authority():
    seen_task_counts: list[int] = []

    class Backend:
        @staticmethod
        def _query_worker_count(task_count):
            seen_task_counts.append(task_count)
            return task_count

        @staticmethod
        def _versions(router):
            del router
            return ()

        @staticmethod
        def _search_authoritative_catalog(query, versions):
            del query, versions
            return {"schema_version": "test", "sources": [], "errors": []}

        @staticmethod
        def _sha256_text(value):
            return f"sha256:{value}"

        @staticmethod
        def _sha256(value):
            del value
            return "sha256:bundle"

        @staticmethod
        def _error(provider, exc):
            return {"provider": provider, "error": str(exc)}

    queries = [f"query-{index}" for index in range(12)]
    bundle = catalog_rag.forced_rag_bundle(
        Backend(),
        object(),
        {
            "domains": [
                {
                    "domain_id": "request",
                    "providers": ["official_docs"],
                    "queries": queries,
                }
            ]
        },
    )

    assert seen_task_counts == [len(queries)]
    assert bundle["query_count"] == len(queries)
    assert bundle["unique_query_count"] == len(queries)
