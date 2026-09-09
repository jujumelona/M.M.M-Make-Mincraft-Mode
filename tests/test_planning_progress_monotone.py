from __future__ import annotations

from copy import deepcopy

import pytest

from minecraft_mod_ai import planning_criterion_fragments as criterion_fragments
from minecraft_mod_ai import planning_state_adaptive_implementation as adaptive
from minecraft_mod_ai import planning_state_research as research


class _Router:
    pass


def _requirements(
    count: int = 1,
    *,
    acceptance_count: int = 1,
) -> list[dict[str, object]]:
    return [
        {
            "requirement_id": f"req_{index}",
            "statement": f"requirement {index}",
            "acceptance": [
                f"acceptance {index}.{criterion_index}"
                for criterion_index in range(1, acceptance_count + 1)
            ],
        }
        for index in range(1, count + 1)
    ]


def _base_state(*, blockers=None, decisions=None, detail_progress=None) -> dict[str, object]:
    return {
        "decisions": list(decisions or []),
        "coverage": [],
        "unresolved": [],
        "blockers": list(blockers or []),
        "detail_progress": list(detail_progress or []),
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


def _fragment(requirement_ref: str, criterion_index: int) -> dict[str, object]:
    return {
        "requirement_ref": requirement_ref,
        "criterion_index": criterion_index,
    }


def _real_fragment(*, evidence_ref: str | None = None) -> dict[str, object]:
    return {
        "section_updates": [
            {
                "section": section,
                "implementation": (
                    f"implement {section.replace('_', ' ')} for this acceptance criterion"
                ),
                "constraint": (
                    f"enforce the {section.replace('_', ' ')} boundary for this acceptance criterion"
                ),
                "evidence_refs": [evidence_ref] if evidence_ref else [],
            }
            for section in adaptive.WORKSHEET_SECTIONS
        ]
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
        "assemble_worksheet_from_fragments",
        lambda *args, **kwargs: _worksheet(),
    )
    monkeypatch.setattr(
        adaptive,
        "_assemble_requirement_plan",
        lambda _requirement, requirement_ref, selected, worksheet, _allowed: _plan(
            requirement_ref, selected, worksheet
        ),
    )


def test_normal_path_runs_once_per_unfinished_acceptance_criterion_without_section_fanout(monkeypatch):
    requirements = _requirements(3, acceptance_count=2)
    _patch_compile_boundaries(monkeypatch, requirements)
    calls: list[tuple[str, int, str]] = []

    def compile_criterion(
        _router,
        *,
        requirement_ref,
        criterion_index,
        criterion,
        selected_sections,
        **_kwargs,
    ):
        assert tuple(selected_sections) == tuple(adaptive.WORKSHEET_SECTIONS)
        calls.append((requirement_ref, criterion_index, criterion))
        return _fragment(requirement_ref, criterion_index)

    monkeypatch.setattr(adaptive, "_compile_criterion", compile_criterion)

    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        _base_state(),
        required_sections_by_requirement={
            row["requirement_id"]: adaptive.WORKSHEET_SECTIONS for row in requirements
        },
    )

    assert len(calls) == 6
    assert {(req, index) for req, index, _criterion in calls} == {
        (f"req_{req_index}", criterion_index)
        for req_index in range(1, 4)
        for criterion_index in range(2)
    }
    assert result["plan_ready"] is True


def test_checkpointed_criterion_progress_reuses_completed_work_and_runs_only_missing_criteria(monkeypatch):
    requirements = _requirements(1, acceptance_count=3)
    _patch_compile_boundaries(monkeypatch, requirements)
    calls: list[int] = []

    monkeypatch.setattr(
        adaptive,
        "load_requirement_progress",
        lambda *args, **kwargs: {0: _fragment("req_1", 0)},
    )

    def compile_criterion(_router, *, requirement_ref, criterion_index, **_kwargs):
        calls.append(criterion_index)
        return _fragment(requirement_ref, criterion_index)

    monkeypatch.setattr(adaptive, "_compile_criterion", compile_criterion)

    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        _base_state(),
        required_sections_by_requirement={"req_1": adaptive.WORKSHEET_SECTIONS},
    )

    assert sorted(calls) == [1, 2]
    assert result["plan_ready"] is True


def test_each_successful_criterion_is_checkpointed_before_requirement_assembly(monkeypatch):
    requirements = _requirements(1, acceptance_count=3)
    _patch_compile_boundaries(monkeypatch, requirements)
    monkeypatch.setattr(adaptive, "router_native_model_parallelism", lambda _router: 1)
    checkpoints: list[dict[str, object]] = []

    monkeypatch.setattr(
        adaptive,
        "_compile_criterion",
        lambda _router, *, requirement_ref, criterion_index, **_kwargs: _fragment(
            requirement_ref, criterion_index
        ),
    )

    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        _base_state(),
        required_sections_by_requirement={"req_1": adaptive.WORKSHEET_SECTIONS},
        checkpoint=lambda state: checkpoints.append(deepcopy(state)),
    )

    progress_counts = [
        len(state.get("detail_progress", []))
        for state in checkpoints
        if state.get("detail_progress")
    ]
    assert progress_counts == [1, 2, 3]
    assert result["detail_progress"] == []
    assert result["plan_ready"] is True


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
        "_compile_criterion",
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


def test_atomic_criterion_failure_is_checkpointed_terminal_and_resume_makes_zero_calls(monkeypatch):
    requirements = _requirements(1, acceptance_count=2)
    _patch_compile_boundaries(monkeypatch, requirements)
    calls = 0
    checkpoints: list[dict[str, object]] = []

    def fail_criterion(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("atomic criterion made no valid progress")

    monkeypatch.setattr(adaptive, "_compile_criterion", fail_criterion)
    monkeypatch.setattr(adaptive, "router_native_model_parallelism", lambda _router: 1)

    with pytest.raises(RuntimeError, match="DETAILED_PLAN_BLOCKED"):
        adaptive.compile_progress_monotone_detailed_plans(
            _Router(),
            "prompt",
            _base_state(),
            required_sections_by_requirement={"req_1": adaptive.WORKSHEET_SECTIONS},
            checkpoint=lambda state: checkpoints.append(deepcopy(state)),
        )

    assert calls == 1
    assert checkpoints
    terminal = checkpoints[-1]
    terminal_rows = [
        row
        for row in terminal["blockers"]
        if row.get("stage") == "detailed_planning" and row.get("terminal") is True
    ]
    assert len(terminal_rows) == 1
    assert terminal_rows[0]["section"] == "acceptance_criterion:1"

    calls_before_resume = calls
    with pytest.raises(RuntimeError, match="terminal detailed-planning blocker"):
        adaptive.compile_progress_monotone_detailed_plans(
            _Router(),
            "prompt",
            terminal,
            required_sections_by_requirement={"req_1": adaptive.WORKSHEET_SECTIONS},
        )
    assert calls == calls_before_resume


def test_full_section_criterion_schema_is_atomic_and_closed() -> None:
    schema = criterion_fragments.criterion_fragment_schema(adaptive.WORKSHEET_SECTIONS)

    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"section_updates"}
    assert schema["required"] == ["section_updates"]
    updates = schema["properties"]["section_updates"]
    assert updates["maxItems"] == len(adaptive.WORKSHEET_SECTIONS)
    assert updates["items"]["additionalProperties"] is False
    assert set(updates["items"]["properties"]["section"]["enum"]) == set(
        adaptive.WORKSHEET_SECTIONS
    )


def test_real_criterion_progress_round_trip_preserves_completed_fragment() -> None:
    requirement = _requirements(1)[0]
    criteria = criterion_fragments.requirement_acceptance_criteria(requirement)
    fragment = _real_fragment(evidence_ref="ev_001")
    stored = criterion_fragments.store_criterion_progress(
        _base_state(),
        requirement_ref="req_1",
        selected_sections=adaptive.WORKSHEET_SECTIONS,
        criterion_index=0,
        criterion=criteria[0],
        fragment=fragment,
    )

    restored = criterion_fragments.load_requirement_progress(
        stored,
        requirement_ref="req_1",
        selected_sections=adaptive.WORKSHEET_SECTIONS,
        criteria=criteria,
        allowed_refs={"ev_001"},
    )

    assert set(restored) == {0}
    assert restored[0] == criterion_fragments.validate_criterion_fragment(
        fragment,
        selected_sections=adaptive.WORKSHEET_SECTIONS,
        allowed_refs={"ev_001"},
    )


def test_real_criterion_fragments_assemble_canonical_all_section_worksheet() -> None:
    requirement = _requirements(1, acceptance_count=2)[0]
    criteria = criterion_fragments.requirement_acceptance_criteria(requirement)
    fragments = {
        index: _real_fragment(evidence_ref="ev_001")
        for index, _criterion in enumerate(criteria)
    }

    worksheet = criterion_fragments.assemble_worksheet_from_fragments(
        requirement,
        selected_sections=adaptive.WORKSHEET_SECTIONS,
        criteria=criteria,
        fragments=fragments,
        allowed_refs={"ev_001"},
    )

    assert tuple(worksheet) == tuple(adaptive.WORKSHEET_SECTIONS)
    for section in adaptive.WORKSHEET_SECTIONS:
        assert worksheet[section]["constraint_evidence_refs"] == ["ev_001"]
        assert isinstance(worksheet[section]["specification"], dict)
        assert worksheet[section]["specification"]["inapplicable_concerns"]


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
