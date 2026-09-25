from __future__ import annotations

import threading

import minecraft_mod_ai.reference_source_research as reference_research


def _record(provider: str, query: str) -> dict[str, object]:
    return {
        "source_id": f"{provider}:{query}",
        "content_sha256": f"sha256:{provider}:{query}",
        "metadata": {"provider": provider, "query": query},
    }


def test_reference_providers_overlap_and_all_contribute(monkeypatch) -> None:
    barrier = threading.Barrier(3, timeout=2.0)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def provider(name: str):
        def run(queries, anchors):
            del anchors
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                barrier.wait()
                query = queries[0]
                return (
                    [_record(name, query)],
                    {"provider": name, "status": "available", "result_count": 1},
                )
            finally:
                with lock:
                    active -= 1
        return run

    monkeypatch.setattr(reference_research, "_wikipedia_sources", provider("wikipedia"))
    monkeypatch.setattr(reference_research, "_wikidata_sources", provider("wikidata"))
    monkeypatch.setattr(
        reference_research,
        "_github_reference_sources",
        provider("github_reference"),
    )

    payload = reference_research.retrieve_reference_grounded_evidence(["alpha", "beta"])

    assert max_active == 3
    assert payload["queries"][0]["authored_queries"] == ["alpha", "beta"]
    assert payload["queries"][0]["provider_policy"] == (
        "independent_parallel_after_identity_first_expansion"
    )
    assert {
        record["source_id"] for record in payload["queries"][0]["evidence_records"]
    } == {
        "wikipedia:alpha",
        "wikidata:alpha",
        "github_reference:alpha",
    }


def test_reference_provider_failure_does_not_suppress_other_providers(monkeypatch) -> None:
    def fail_wikipedia(queries, anchors):
        del queries, anchors
        raise RuntimeError("wiki failed")

    def wikidata(queries, anchors):
        del anchors
        return (
            [_record("wikidata", queries[0])],
            {"provider": "wikidata", "status": "available", "result_count": 1},
        )

    def github(queries, anchors):
        del anchors
        return (
            [_record("github_reference", queries[0])],
            {
                "provider": "github_reference",
                "status": "available",
                "result_count": 1,
            },
        )

    monkeypatch.setattr(reference_research, "_wikipedia_sources", fail_wikipedia)
    monkeypatch.setattr(reference_research, "_wikidata_sources", wikidata)
    monkeypatch.setattr(reference_research, "_github_reference_sources", github)

    payload = reference_research.retrieve_reference_grounded_evidence(["alpha"])
    row = payload["queries"][0]

    assert row["provider_receipts"]["wikipedia"]["status"] == "error"
    assert {
        record["source_id"] for record in row["evidence_records"]
    } == {
        "wikidata:alpha",
        "github_reference:alpha",
    }
    assert row["retrieval_errors"] == [
        {"provider": "wikipedia", "error": "RuntimeError: wiki failed"}
    ]
    assert row["provider_policy"] == (
        "independent_parallel_after_identity_first_expansion"
    )
