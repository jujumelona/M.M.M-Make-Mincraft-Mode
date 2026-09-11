from __future__ import annotations

import threading

import minecraft_mod_ai.reference_source_research as reference_research


def _record(provider: str, query: str) -> dict[str, object]:
    return {
        "source_id": f"{provider}:{query}",
        "content_sha256": f"sha256:{provider}:{query}",
        "metadata": {"provider": provider, "query": query},
    }


def test_reference_query_rows_overlap_but_providers_do_not_fan_out(monkeypatch) -> None:
    barrier = threading.Barrier(2, timeout=2.0)
    lock = threading.Lock()
    active = 0
    max_active = 0
    github_calls: list[str] = []

    monkeypatch.setattr(reference_research, "_MAX_QUERY_WORKERS", 2)

    def wikipedia(query: str):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait()
            return [
                _record("wikipedia", query)
            ], {
                "provider": "wikipedia",
                "status": "available",
                "result_count": 1,
            }
        finally:
            with lock:
                active -= 1

    def github(query: str):
        github_calls.append(query)
        return [
            _record("github_reference", query)
        ], {
            "provider": "github_reference",
            "status": "available",
            "result_count": 1,
        }

    monkeypatch.setattr(reference_research, "_wikipedia_sources", wikipedia)
    monkeypatch.setattr(reference_research, "_github_reference_sources", github)

    payload = reference_research.retrieve_reference_grounded_evidence(["alpha", "beta"])

    assert [row["query"] for row in payload["queries"]] == ["alpha", "beta"]
    assert max_active == 2
    assert github_calls == []
    for query, row in zip(("alpha", "beta"), payload["queries"], strict=True):
        assert [record["source_id"] for record in row["evidence_records"]] == [
            f"wikipedia:{query}"
        ]
        assert row["provider_receipts"]["github_reference"]["status"] == (
            "skipped_wikipedia_has_evidence"
        )
        assert row["provider_receipts"]["wikidata"]["status"] == (
            "skipped_wikipedia_has_evidence"
        )
        assert row["content_record_count"] == 1
        assert row["retrieval_errors"] == []
        assert row["provider_policy"] == (
            "sequential_identity_first_then_github_empty_fallback"
        )


def test_reference_provider_failure_uses_sequential_github_fallback(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fail_wikipedia(query: str):
        calls.append(("wikipedia", query))
        raise RuntimeError(f"wiki failed for {query}")

    def github(query: str):
        calls.append(("github_reference", query))
        return [
            _record("github_reference", query)
        ], {
            "provider": "github_reference",
            "status": "available",
            "result_count": 1,
        }

    monkeypatch.setattr(reference_research, "_MAX_PROVIDER_ATTEMPTS", 1)
    monkeypatch.setattr(reference_research, "_wikipedia_sources", fail_wikipedia)
    monkeypatch.setattr(reference_research, "_github_reference_sources", github)

    payload = reference_research.retrieve_reference_grounded_evidence(["alpha"])
    row = payload["queries"][0]

    assert calls == [
        ("wikipedia", "alpha"),
        ("github_reference", "alpha"),
    ]
    assert row["provider_receipts"]["wikipedia"] == {
        "provider": "wikipedia",
        "status": "error",
        "result_count": 0,
        "attempts": 1,
    }
    assert row["provider_receipts"]["github_reference"]["status"] == "available"
    assert [record["source_id"] for record in row["evidence_records"]] == [
        "github_reference:alpha"
    ]
    assert row["retrieval_errors"] == [
        {"provider": "wikipedia", "error": "RuntimeError: wiki failed for alpha"}
    ]
    assert row["provider_policy"] == (
        "sequential_identity_first_then_github_empty_fallback"
    )
