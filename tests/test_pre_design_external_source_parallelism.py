from __future__ import annotations

import threading

from minecraft_mod_ai import pre_design_external_source_contract as external_sources


def _receipt(*, status: str = "available", reason: str = "search_exhausted_no_repository"):
    return {
        "records": [],
        "search_requests": 1,
        "source_requests": 0,
        "provider_status": status,
        "saturation_reason": reason,
        "errors": [],
    }


def test_selected_source_receipts_use_bounded_parallel_workers(monkeypatch):
    monkeypatch.setattr(external_sources, "_MAX_EXTERNAL_SOURCE_WORKERS", 2)
    barrier = threading.Barrier(2, timeout=2.0)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def retrieve(query: str):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait()
            return _receipt()
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(external_sources, "_retrieve_github_source_body", retrieve)

    receipts = external_sources._retrieve_selected_source_receipts(
        ["query one", "query two", "query three", "query four"]
    )

    assert max_active == 2
    assert list(receipts) == ["query one", "query two", "query three", "query four"]
    assert all(receipt["provider_status"] == "available" for receipt in receipts.values())


def test_rate_limit_stops_later_batches_and_preserves_serial_order_semantics(monkeypatch):
    monkeypatch.setattr(external_sources, "_MAX_EXTERNAL_SOURCE_WORKERS", 2)
    calls: list[str] = []
    barrier = threading.Barrier(2, timeout=2.0)

    def retrieve(query: str):
        calls.append(query)
        barrier.wait()
        if query == "query two":
            return _receipt(status="rate_limited", reason="provider_limited")
        return _receipt()

    monkeypatch.setattr(external_sources, "_retrieve_github_source_body", retrieve)

    receipts = external_sources._retrieve_selected_source_receipts(
        ["query one", "query two", "query three", "query four"]
    )

    assert len(calls) == 2
    assert set(calls) == {"query one", "query two"}
    assert receipts["query one"]["provider_status"] == "available"
    assert receipts["query two"]["provider_status"] == "rate_limited"
    assert receipts["query three"]["saturation_reason"] == "skipped_after_provider_rate_limit"
    assert receipts["query four"]["saturation_reason"] == "skipped_after_provider_rate_limit"
