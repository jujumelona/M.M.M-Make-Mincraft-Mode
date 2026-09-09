from __future__ import annotations

from copy import deepcopy

import pytest

from minecraft_mod_ai import planning_state_adaptive_implementation as adaptive
from minecraft_mod_ai import planning_state_research as research


class _Router:
    pass


def _requirements(count: int = 1) -> list[dict[str, object]]:
    return [
        {
            "requirement_id": f"req_{index}",
            "statement": f"requirement {index}",
            "acceptance": [f"acceptance {index}"],
        }
        for index in range(1, count + 1)
    ]


def _base_state(*, blockers=None, decisions=None) -> dict[str, object]:
    return {
        "decisions": list(decisions or []),
        "coverage": [],
        "unresolved": [],
        "blockers": list(blockers or []),
        "state_sha256": "test",
    }


def _row(section: str) -> dict[str, object]:
    return {
        "specification": {"section": section},
        "constraint_evidence_refs": [],
    }


def _worksheet() -> dict[str, dict[str, object]]:
    return {section: _row(section) for section in adaptive.WORKSHEET_SECTIONS}


def _plan(requirement_ref: str, selected_sections, worksheet) -> dict[str, object]:
    return {
        "requirement_ref": requirement_ref,
        "required_detail_sections": list(selected_sections),
        "engineering_worksheet": deepcopy(worksheet),
    }


def _patch_compile_boundaries(monkeypatch, requirements):
    monkeypatch.setattr(adaptive, "validate_planning_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(adaptive, "_requirement_decisions", lambda _state: requirements)
    monkeypatch.setattr(
        adaptive,
        "_requirement_grounding",
        lambda _state, _requirement_ref: ([], set()),
    )
    monkeypatch.setattr(adaptive, "router_native_model_parallelism", lambda _router: 8)
    monkeypatch.setattr(
        adaptive,
        "_assemble_requirement_plan",
        lambda _requirement, requirement_ref, selected, worksheet, _allowed: _plan(
            requirement_ref, selected, worksheet
        ),
    )


def test_normal_path_attempts_each_unfinished_requirement_once_without_section_fanout(monkeypatch):
    requirements = _requirements(3)
    _patch_compile_boundaries(monkeypatch, requirements)
    calls: list[str] = []

    def whole_attempt(_router, *, requirement_ref, **_kwargs):
        calls.append(requirement_ref)
        return _worksheet(), ""

    monkeypatch.setattr(adaptive, "_attempt_whole_requirement", whole_attempt)
    monkeypatch.setattr(
        adaptive,
        "_compile_section_adaptive",
        lambda *args, **kwargs: pytest.fail("normal path must not create section fallback calls"),
    )

    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        _base_state(),
        required_sections_by_requirement={
            row["requirement_id"]: adaptive.WORKSHEET_SECTIONS for row in requirements
        },
    )

    assert sorted(calls) == ["req_1", "req_2", "req_3"]
    assert len(calls) == len(requirements)
    assert result["plan_ready"] is True


def test_partial_whole_result_keeps_valid_section_and_falls_back_only_for_missing_sections(monkeypatch):
    requirements = _requirements(1)
    _patch_compile_boundaries(monkeypatch, requirements)
    whole_calls = 0
    fallback_calls: list[str] = []

    def whole_attempt(*_args, **_kwargs):
        nonlocal whole_calls
        whole_calls += 1
        return {"behavior_contract": _row("behavior_contract")}, "invalid remainder"

    def fallback(_router, *, section, **_kwargs):
        fallback_calls.append(section)
        return _row(section)

    monkeypatch.setattr(adaptive, "_attempt_whole_requirement", whole_attempt)
    monkeypatch.setattr(adaptive, "_compile_section_adaptive", fallback)

    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        _base_state(),
        required_sections_by_requirement={"req_1": adaptive.WORKSHEET_SECTIONS},
    )

    assert whole_calls == 1
    assert "behavior_contract" not in fallback_calls
    assert set(fallback_calls) == set(adaptive.WORKSHEET_SECTIONS) - {"behavior_contract"}
    assert len(fallback_calls) == len(adaptive.WORKSHEET_SECTIONS) - 1
    assert result["plan_ready"] is True


def test_dependency_salvage_discards_sections_whose_prerequisite_failed(monkeypatch):
    decoded = {
        "behavior_contract": _row("behavior_contract"),
        "state_model": _row("state_model"),
        "algorithm": _row("algorithm"),
    }

    def validate(value, _allowed, section):
        if section == "state_model":
            raise ValueError("broken state model")
        return deepcopy(value)

    monkeypatch.setattr(adaptive, "validate_worksheet_section", validate)
    salvaged = adaptive._salvage_dependency_closed_sections(
        decoded,
        selected_sections=("behavior_contract", "state_model", "algorithm"),
        allowed=set(),
    )

    assert set(salvaged) == {"behavior_contract"}


def test_completed_requirement_is_not_regenerated_on_resume(monkeypatch):
    requirements = _requirements(1)
    _patch_compile_boundaries(monkeypatch, requirements)
    worksheet = _worksheet()
    existing = _plan("req_1", adaptive.WORKSHEET_SECTIONS, worksheet)
    existing.update(
        {
            "decision_type": "detailed_implementation_plan",
            "decision_id": "detail_001",
        }
    )

    monkeypatch.setattr(
        adaptive,
        "_attempt_whole_requirement",
        lambda *args, **kwargs: pytest.fail("completed requirement must not call the model"),
    )

    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        _base_state(decisions=[existing]),
        required_sections_by_requirement={"req_1": adaptive.WORKSHEET_SECTIONS},
    )

    assert result["plan_ready"] is True
    assert len(result["coverage"]) == 1


def test_minimal_fallback_failure_is_checkpointed_terminal_and_resume_makes_zero_calls(monkeypatch):
    requirements = _requirements(1)
    _patch_compile_boundaries(monkeypatch, requirements)
    whole_calls = 0
    fallback_calls = 0
    checkpoints: list[dict[str, object]] = []

    def whole_attempt(*_args, **_kwargs):
        nonlocal whole_calls
        whole_calls += 1
        return {}, "whole output invalid"

    def fallback(*_args, **_kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        raise ValueError("minimal contract made no valid progress")

    monkeypatch.setattr(adaptive, "_attempt_whole_requirement", whole_attempt)
    monkeypatch.setattr(adaptive, "_compile_section_adaptive", fallback)

    with pytest.raises(RuntimeError, match="DETAILED_PLAN_BLOCKED"):
        adaptive.compile_progress_monotone_detailed_plans(
            _Router(),
            "prompt",
            _base_state(),
            required_sections_by_requirement={"req_1": adaptive.WORKSHEET_SECTIONS},
            checkpoint=lambda state: checkpoints.append(deepcopy(state)),
        )

    assert whole_calls == 1
    assert fallback_calls == 1
    assert checkpoints
    terminal = checkpoints[-1]
    terminal_rows = [
        row
        for row in terminal["blockers"]
        if row.get("stage") == "detailed_planning" and row.get("terminal") is True
    ]
    assert len(terminal_rows) == 1

    calls_before_resume = (whole_calls, fallback_calls)
    with pytest.raises(RuntimeError, match="terminal detailed-planning blocker"):
        adaptive.compile_progress_monotone_detailed_plans(
            _Router(),
            "prompt",
            terminal,
            required_sections_by_requirement={"req_1": adaptive.WORKSHEET_SECTIONS},
        )
    assert (whole_calls, fallback_calls) == calls_before_resume


def test_blocked_research_stays_terminal_on_reentry(monkeypatch):
    monkeypatch.setattr(research, "validate_planning_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        research,
        "merge_repository_candidates",
        lambda existing, _new: list(existing),
    )
    monkeypatch.setattr(
        research,
        "forced_rag_bundle",
        lambda *args, **kwargs: pytest.fail("blocked research must not be retrieved again"),
    )
    state = {
        "repository_candidates": [],
        "unresolved": [],
        "research_queue": [
            {
                "research_id": "r_001",
                "resolves": ["u_001"],
                "objective": "already attempted",
                "information_needed": "already attempted",
                "source_kinds": ["project_rag"],
                "queries": ["already attempted"],
                "status": "blocked",
            }
        ],
        "evidence": [],
        "resolved": [],
        "blockers": [],
        "state_sha256": "test",
    }

    result = research.collect_planning_state_research(_Router(), "prompt", state)

    assert result["research_queue"][0]["status"] == "blocked"
