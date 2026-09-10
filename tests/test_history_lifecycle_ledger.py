from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.history_lifecycle_ledger import (
    ArtifactRef,
    HistoryKind,
    HistoryLifecycleLedger,
    HistoryRelation,
    HistoryStatus,
    build_history_record,
    history_record_from_verification,
)


def test_history_keeps_payloads_by_reference_and_projects_current_state(tmp_path) -> None:
    ledger = HistoryLifecycleLedger(tmp_path / ".mmm" / "history_lifecycle.jsonl")
    first = build_history_record(
        subject_id="plan:seasonal-mod",
        kind=HistoryKind.PLAN.value,
        status=HistoryStatus.ACTIVE.value,
        summary="Initial approved plan",
        artifact_refs=(ArtifactRef("proposal_store", "proposal:abc", "a" * 64),),
    )
    assert ledger.append(first) is True
    assert ledger.append(first) is False

    replacement = build_history_record(
        subject_id="plan:seasonal-mod",
        kind=HistoryKind.PLAN.value,
        status=HistoryStatus.ACTIVE.value,
        summary="Revised approved plan",
        artifact_refs=(ArtifactRef("proposal_store", "proposal:def", "b" * 64),),
        relations=(HistoryRelation("supersedes", first.record_id),),
    )
    ledger.append(replacement)

    assert [row.record_id for row in ledger.current(subject_id="plan:seasonal-mod")] == [
        replacement.record_id
    ]
    compact = ledger.compact_view(subject_id="plan:seasonal-mod")
    assert compact["record_count"] == 2
    assert compact["records"][0]["current"] is False
    assert compact["records"][1]["current"] is True
    assert "proposal body" not in json.dumps(compact)


def test_verification_record_reuses_existing_classifier_result(tmp_path) -> None:
    ledger = HistoryLifecycleLedger(tmp_path / "history.jsonl")
    plan = build_history_record(
        subject_id="plan:test",
        kind=HistoryKind.PLAN.value,
        status=HistoryStatus.ACTIVE.value,
        summary="Plan under verification",
    )
    ledger.append(plan)

    verification = {
        "level": "L3",
        "verified_failure": True,
        "verifier_chain": [
            {"kind": "build", "status": "PASS"},
            {"kind": "gametest", "status": "FAIL"},
        ],
    }
    record = history_record_from_verification(
        subject_id="plan:test",
        summary="GameTest found a behavioral failure",
        verification=verification,
        evidence_ref=ArtifactRef("trajectory_memory", "trajectory:receipt-1", "c" * 64),
        relations=(HistoryRelation("related_to", plan.record_id),),
    )
    ledger.append(record)

    assert record.verification_level == "L3"
    assert record.failure_taxonomy == ("gametest",)
    assert record.status == HistoryStatus.VERIFIED.value


def test_invalidation_is_explicit_and_removes_target_from_current_projection(tmp_path) -> None:
    ledger = HistoryLifecycleLedger(tmp_path / "history.jsonl")
    plan = build_history_record(
        subject_id="plan:test",
        kind=HistoryKind.PLAN.value,
        status=HistoryStatus.ACTIVE.value,
        summary="Plan under test",
    )
    ledger.append(plan)

    invalidation = build_history_record(
        subject_id="plan:test",
        kind=HistoryKind.FAILURE.value,
        status=HistoryStatus.INVALIDATED.value,
        summary="Acceptance evidence invalidated the plan",
        relations=(HistoryRelation("invalidates", plan.record_id),),
        failure_taxonomy=("acceptance",),
    )
    ledger.append(invalidation)
    assert ledger.current(subject_id="plan:test") == []


def test_unknown_relation_fails_closed(tmp_path) -> None:
    ledger = HistoryLifecycleLedger(tmp_path / "history.jsonl")
    record = build_history_record(
        subject_id="plan:test",
        kind=HistoryKind.DECISION.value,
        status=HistoryStatus.ACTIVE.value,
        summary="Decision referencing an absent record",
        relations=(HistoryRelation("derived_from", "hist:missing"),),
    )
    with pytest.raises(ValueError, match="HISTORY_UNKNOWN_RELATION"):
        ledger.append(record)


def test_rejected_history_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="HISTORY_REJECTION_REASON_REQUIRED"):
        build_history_record(
            subject_id="plan:test",
            kind=HistoryKind.DECISION.value,
            status=HistoryStatus.REJECTED.value,
            summary="Rejected alternative",
        )


def test_corrupt_or_tampered_history_fails_closed(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    ledger = HistoryLifecycleLedger(path)
    record = build_history_record(
        subject_id="plan:test",
        kind=HistoryKind.PLAN.value,
        status=HistoryStatus.ACTIVE.value,
        summary="Original",
    )
    ledger.append(record)
    payload = json.loads(path.read_text())
    payload["summary"] = "tampered"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="HISTORY_CORRUPT_LINE"):
        ledger.read_all()
