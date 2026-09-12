from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_state_adaptive_implementation import (
    _detail_matches_selection,
    _finish_requirement,
    _pending_work_items,
    _requirement_work_complete,
)


def _job(*, artifact_done: bool, criterion_done: bool) -> dict:
    return {
        "requirement": {"requirement_id": "req_001"},
        "requirement_ref": "req_001",
        "selected_sections": (),
        "evidence": [],
        "allowed": set(),
        "criteria": ("The block is observable in game.",),
        "fragments": {0: {}} if criterion_done else {},
        "artifact_kinds": ["block"],
        "artifact_steps": {"block": ["block/register_basic"]},
        "artifact_fragments": (
            {"block": {"block/register_basic": {"status": "PASS"}}}
            if artifact_done
            else {"block": {}}
        ),
    }


def test_artifact_work_never_suppresses_acceptance_work() -> None:
    job = _job(artifact_done=False, criterion_done=False)

    pending = list(_pending_work_items([job], {}))

    assert pending == [
        ("artifact", 0, "block", 0, "block/register_basic"),
        ("criterion", 0, "", 0, ""),
    ]


def test_requirement_requires_artifact_and_acceptance_completion() -> None:
    assert not _requirement_work_complete(
        _job(artifact_done=True, criterion_done=False)
    )
    assert not _requirement_work_complete(
        _job(artifact_done=False, criterion_done=True)
    )
    assert _requirement_work_complete(
        _job(artifact_done=True, criterion_done=True)
    )


def test_finish_requirement_rejects_missing_acceptance_fragments() -> None:
    job = _job(artifact_done=True, criterion_done=False)

    with pytest.raises(RuntimeError, match="DETAILED_PLAN_ACCEPTANCE_INCOMPLETE"):
        _finish_requirement(
            job,
            object(),
            working_state={},
            requirement_order=("req_001",),
            completed_details={},
            checkpoint=None,
        )


def test_legacy_placeholder_detail_is_not_restored_as_complete() -> None:
    detail = {
        "worksheet_contract": "authored_concern_records",
        "required_detail_sections": [],
        "engineering_worksheet": {},
    }

    assert not _detail_matches_selection(detail, ())

    detail["acceptance_criteria_complete"] = True
    assert _detail_matches_selection(detail, ())
