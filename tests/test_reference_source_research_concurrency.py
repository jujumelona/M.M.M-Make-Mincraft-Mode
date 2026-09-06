from __future__ import annotations

from threading import Barrier

import minecraft_mod_ai.reference_source_research as reference_research


def _record(provider: str, query: str) -> dict[str, object]:
    return {
        "source_id": f"{provider}:{query}",
        "content_sha256": f"sha256:{provider}:{query}",
        "metadata": {"provider": provider, "query": query},
    }


def test_reference_queries_overlap_and_preserve_result_order(monkeypatch) -> None:
    barrier = Barrier(2)

    def wikipedia(query: str):
        barrier.wait(timeout=5)
        return [
            _record("wikipedia", query)
        ], {
            "provider": "wikipedia",
            "status": "available",
            "result_count": 1,
        }

    def forbidden_github(_query: str):
        raise AssertionError("GitHub fallback must not run when Wikipedia has evidence")

    monkeypatch.setattr(reference_research, "_wikipedia_sources", wikipedia)
    monkeypatch.setattr(reference_research, "_github_reference_sources", forbidden_github)

    payload = reference_research.retrieve_reference_grounded_evidence(["alpha", "beta"])

    assert [row["query"] for row in payload["queries"]] == ["alpha", "beta"]
    for query, row in zip(("alpha", "beta"), payload["queries"], strict=True):
        assert list(row["provider_receipts"]) == ["wikipedia", "github_reference"]
        assert row["provider_receipts"]["github_reference"]["status"] == (
            "skipped_wikipedia_has_evidence"
        )
        assert [record["source_id"] for record in row["evidence_records"]] == [
            f"wikipedia:{query}",
        ]
        assert row["content_record_count"] == 1
        assert row["retrieval_errors"] == []
        assert row["provider_policy"] == "wikipedia_then_github_empty_fallback"


def test_reference_provider_failure_is_isolated(monkeypatch) -> None:
    def fail_wikipedia(query: str):
        raise RuntimeError(f"wiki failed for {query}")

    def github(query: str):
        return [
            _record("github_reference", query)
        ], {
            "provider": "github_reference",
            "status": "available",
            "result_count": 1,
        }

    monkeypatch.setattr(reference_research, "_wikipedia_sources", fail_wikipedia)
    monkeypatch.setattr(reference_research, "_github_reference_sources", github)

    payload = reference_research.retrieve_reference_grounded_evidence(["alpha"])
    row = payload["queries"][0]

    assert row["provider_receipts"]["wikipedia"] == {
        "provider": "wikipedia",
        "status": "error",
        "result_count": 0,
    }
    assert row["provider_receipts"]["github_reference"]["status"] == "available"
    assert [record["source_id"] for record in row["evidence_records"]] == [
        "github_reference:alpha"
    ]
    assert row["retrieval_errors"] == [
        {"provider": "wikipedia", "error": "RuntimeError: wiki failed for alpha"}
    ]
