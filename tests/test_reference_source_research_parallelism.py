from __future__ import annotations

import threading

from minecraft_mod_ai import reference_source_research as rsr


def test_parallel_reference_queries_are_bounded_and_keep_input_order(monkeypatch) -> None:
    queries = [f"reference-{index}" for index in range(6)]
    workers = 3
    barrier = threading.Barrier(workers, timeout=2.0)
    lock = threading.Lock()
    active = 0
    max_active = 0

    monkeypatch.setattr(rsr, "_MAX_QUERY_WORKERS", workers)

    def fake_retrieve(query: str) -> dict[str, object]:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait()
            return {"query": query}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(rsr, "_retrieve_query_row", fake_retrieve)

    result = rsr.retrieve_reference_grounded_evidence(queries)

    assert [row["query"] for row in result["queries"]] == queries
    assert max_active == workers


def test_query_row_skips_github_when_wikipedia_has_evidence(monkeypatch) -> None:
    wiki_record = {
        "source_id": "wikipedia:en:1",
        "content_sha256": "sha256:wikipedia",
    }

    monkeypatch.setattr(
        rsr,
        "_wikipedia_sources",
        lambda query: (
            [wiki_record],
            {"provider": "wikipedia", "status": "available", "result_count": 1},
        ),
    )

    def fail_if_called(query: str):
        raise AssertionError(f"GitHub fallback must not run for {query!r}")

    monkeypatch.setattr(rsr, "_github_reference_sources", fail_if_called)

    row = rsr._retrieve_query_row("known reference")

    assert row["evidence_records"] == [wiki_record]
    assert row["provider_receipts"]["github_reference"]["status"] == (
        "skipped_wikipedia_has_evidence"
    )


def test_query_row_uses_github_only_when_wikipedia_is_empty(monkeypatch) -> None:
    github_record = {
        "source_id": "github-reference:owner/repo",
        "content_sha256": "sha256:github",
    }
    github_queries: list[str] = []

    monkeypatch.setattr(
        rsr,
        "_wikipedia_sources",
        lambda query: (
            [],
            {"provider": "wikipedia", "status": "available", "result_count": 0},
        ),
    )

    def github_fallback(query: str):
        github_queries.append(query)
        return (
            [github_record],
            {
                "provider": "github_reference",
                "status": "available",
                "result_count": 1,
                "policy": "wikipedia_empty_fallback_only",
            },
        )

    monkeypatch.setattr(rsr, "_github_reference_sources", github_fallback)

    row = rsr._retrieve_query_row("missing reference")

    assert github_queries == ["missing reference"]
    assert row["evidence_records"] == [github_record]
    assert row["provider_receipts"]["github_reference"]["status"] == "available"
