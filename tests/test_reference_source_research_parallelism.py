from __future__ import annotations

import threading

from minecraft_mod_ai import reference_source_research as rsr


def _provider_result(name: str, query: str):
    record = {
        "source_id": f"{name}:{query}",
        "content_sha256": f"sha256:{name}:{query}",
        "content": f"{name} evidence for {query}",
    }
    return (
        [record],
        {"provider": name, "status": "available", "result_count": 1},
    )


def test_reference_providers_are_bounded_parallel_and_result_order_is_canonical(
    monkeypatch,
) -> None:
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
                return _provider_result(name, queries[0])
            finally:
                with lock:
                    active -= 1

        return run

    monkeypatch.setattr(rsr, "_wikipedia_sources", provider("wikipedia"))
    monkeypatch.setattr(rsr, "_wikidata_sources", provider("wikidata"))
    monkeypatch.setattr(rsr, "_github_reference_sources", provider("github_reference"))

    result = rsr.retrieve_reference_grounded_evidence(
        ["Maple Story gameplay rules", "Maple Story documented systems behavior rules"]
    )

    assert max_active == 3
    assert result["retrieval_strategy"] == "identity_first_bounded_expansion"
    assert list(result["queries"][0]["provider_receipts"]) == [
        "wikipedia",
        "wikidata",
        "github_reference",
    ]
    assert result["queries"][0]["content_record_count"] == 3


def test_provider_retry_remains_bounded() -> None:
    attempts = 0

    def flaky_provider():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return _provider_result("fixture", "known")

    found, receipt, error = rsr._retrieve_provider("fixture", flaky_provider)

    assert attempts == 2
    assert found
    assert receipt["attempts"] == 2
    assert error is None
