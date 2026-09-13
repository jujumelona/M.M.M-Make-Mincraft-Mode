from __future__ import annotations

import threading

from minecraft_mod_ai import deadline_executor
from minecraft_mod_ai import reference_source_research as reference


def test_reference_identity_is_inferred_from_shared_query_prefix() -> None:
    anchors = reference._infer_reference_names(
        [
            "Maple Story gameplay rules and progression",
            "Maple Story documented systems behavior rules",
        ]
    )

    assert anchors == ["maple story"]


def test_reference_search_plan_is_identity_first_not_long_query_only() -> None:
    anchors, plan = reference._reference_search_plan(
        [
            "Maple Story gameplay rules and progression",
            "Maple Story documented systems behavior rules",
        ]
    )

    assert anchors == ["maple story"]
    assert plan[0] == "maple story"
    assert "maplestory" in plan[:3]
    assert "Maple Story gameplay rules and progression" in plan


def test_identity_match_tolerates_spacing_but_rejects_other_entity() -> None:
    assert reference._identity_matches("Maple Story", "MapleStory") is True
    assert reference._identity_matches("Maple Story", "DaVinci Resolve") is False


def test_wikipedia_direct_title_resolution_survives_empty_search(monkeypatch) -> None:
    body = (
        "MapleStory is an online role-playing game with character levels, quests, class "
        "advancement, progression systems, and other documented gameplay mechanics. "
        "This fixture is intentionally long enough to qualify as a retrieved source body."
    )

    def fake_json(url: str, *, headers=None):
        del headers
        if "prop=extracts%7Cinfo" in url:
            return {
                "query": {
                    "pages": {
                        "1": {
                            "pageid": 1,
                            "title": "MapleStory",
                            "extract": body,
                            "fullurl": "https://en.wikipedia.org/wiki/MapleStory",
                        }
                    }
                }
            }
        if "list=search" in url:
            return {"query": {"search": []}}
        raise AssertionError(url)

    monkeypatch.setattr(reference, "_json", fake_json)

    records, receipt = reference._wikipedia_sources(
        ["maple story", "Maple Story documented systems behavior rules"],
        ["maple story"],
    )

    assert records
    assert records[0]["title"] == "MapleStory"
    assert records[0]["content"] == body
    assert receipt["result_count"] >= 1
    assert receipt["source_requests"] >= 1


def test_provider_transport_retries_transient_failure() -> None:
    attempts = 0

    def flaky_provider():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return ([{"source_id": "fixture", "content_sha256": "sha256:x"}], {"provider": "fixture", "status": "available", "result_count": 1})

    found, receipt, error = reference._retrieve_provider("fixture", flaky_provider)

    assert attempts == 2
    assert found
    assert receipt["attempts"] == 2
    assert error is None


def test_combined_reference_retrieval_merges_independent_providers(monkeypatch) -> None:
    def provider(name: str):
        return (
            [
                {
                    "source_id": name,
                    "content_sha256": f"sha256:{name}",
                    "content": f"{name} evidence body",
                }
            ],
            {"provider": name, "status": "available", "result_count": 1},
        )

    monkeypatch.setattr(reference, "_wikipedia_sources", lambda queries, anchors: provider("wikipedia"))
    monkeypatch.setattr(reference, "_wikidata_sources", lambda queries, anchors: provider("wikidata"))
    monkeypatch.setattr(reference, "_github_reference_sources", lambda queries, anchors: provider("github_reference"))

    result = reference.retrieve_reference_grounded_evidence(
        [
            "MapleStory gameplay rules",
            "MapleStory documented systems behavior rules",
        ]
    )

    assert result["retrieval_strategy"] == "identity_first_bounded_expansion"
    assert result["inferred_reference_names"] == ["maplestory"]
    assert result["queries"][0]["content_record_count"] == 3
    assert set(result["queries"][0]["provider_receipts"]) == {
        "wikipedia",
        "wikidata",
        "github_reference",
    }


def test_reference_provider_deadline_returns_without_waiting_for_blocked_provider(monkeypatch) -> None:
    release = threading.Event()
    slow_finished = threading.Event()

    def provider(name: str):
        return (
            [
                {
                    "source_id": name,
                    "content_sha256": f"sha256:{name}",
                    "content": f"{name} evidence body",
                }
            ],
            {"provider": name, "status": "available", "result_count": 1},
        )

    def blocked_github(queries, anchors):
        del queries, anchors
        try:
            release.wait(timeout=1.0)
            return provider("github_reference")
        finally:
            slow_finished.set()

    monkeypatch.setattr(reference, "_wikipedia_sources", lambda queries, anchors: provider("wikipedia"))
    monkeypatch.setattr(reference, "_wikidata_sources", lambda queries, anchors: provider("wikidata"))
    monkeypatch.setattr(reference, "_github_reference_sources", blocked_github)
    monkeypatch.setattr(deadline_executor, "planning_work_unit_timeout_seconds", lambda: 0.02)
    monkeypatch.setattr(
        deadline_executor,
        "planning_stage_deadline",
        lambda *, work_units, workers, started_at=None: float(started_at) + 0.02,
    )

    try:
        result = reference.retrieve_reference_grounded_evidence(
            [
                "MapleStory gameplay rules",
                "MapleStory documented systems behavior rules",
            ]
        )
        assert slow_finished.is_set() is False
    finally:
        release.set()

    row = result["queries"][0]
    assert row["content_record_count"] == 2
    assert row["provider_receipts"]["wikipedia"]["status"] == "available"
    assert row["provider_receipts"]["wikidata"]["status"] == "available"
    assert row["provider_receipts"]["github_reference"]["status"] == "error"
    assert "ParallelExecutionTimeout" in row["retrieval_errors"][0]["error"]
