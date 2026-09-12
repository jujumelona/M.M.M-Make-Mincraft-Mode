from __future__ import annotations

from copy import deepcopy
from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS

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
    return criterion_fragments.assemble_worksheet_from_fragments(
        {"statement": "Collect resources."},
        selected_sections=WORKSHEET_SECTIONS,
        criteria=("Collected items enter inventory.",),
        fragments={0: _real_fragment()},
        allowed_refs=set(),
    )[section]


def _worksheet() -> dict[str, dict[str, object]]:
    return {section: _row(section) for section in WORKSHEET_SECTIONS}


def _plan(requirement_ref: str, selected_sections, worksheet) -> dict[str, object]:
    return {
        "requirement_ref": requirement_ref,
        "required_detail_sections": list(selected_sections),
        "engineering_worksheet": deepcopy(worksheet),
        "worksheet_contract": "authored_concern_records",
        "acceptance_criteria_complete": True,
        "acceptance_criteria_count": 1,
    }


def _fragment(requirement_ref: str, criterion_index: int) -> dict[str, object]:
    return {
        "requirement_ref": requirement_ref,
        "criterion_index": criterion_index,
    }


def _real_fragment(*, evidence_ref=None):
    from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS

    rows = []
    for section in WORKSHEET_SECTIONS:
        specification = {
            concern: [
                {
                    field: f"{section}.{concern}.{field}"
                    for field in columns.split()
                }
            ]
            for concern, columns in DETAIL_RECORDS[section].items()
        }
        specification["inapplicable_concerns"] = []
        rows.append(
            {
                "section": section,
                "specification": specification,
                "constraint_evidence_refs": [evidence_ref] if evidence_ref else [],
            }
        )
    return {"section_updates": rows}


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
        assert tuple(selected_sections) == tuple(WORKSHEET_SECTIONS)
        calls.append((requirement_ref, criterion_index, criterion))
        return _fragment(requirement_ref, criterion_index)

    monkeypatch.setattr(adaptive, "_compile_criterion", compile_criterion)

    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        _base_state(),
        required_sections_by_requirement={
            row["requirement_id"]: WORKSHEET_SECTIONS for row in requirements
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
        required_sections_by_requirement={"req_1": WORKSHEET_SECTIONS},
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
        required_sections_by_requirement={"req_1": WORKSHEET_SECTIONS},
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
    existing = _plan("req_1", WORKSHEET_SECTIONS, worksheet)
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
        required_sections_by_requirement={"req_1": WORKSHEET_SECTIONS},
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
            required_sections_by_requirement={"req_1": WORKSHEET_SECTIONS},
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
            required_sections_by_requirement={"req_1": WORKSHEET_SECTIONS},
        )
    assert calls == calls_before_resume


def test_each_concern_schema_is_closed():
    from minecraft_mod_ai.task_template_catalog import load_template
    from minecraft_mod_ai.task_template_runner import record_response_schema

    for section in WORKSHEET_SECTIONS:
        for identifier in load_template(f"feature/{section}")["steps"]:
            schema = record_response_schema(load_template(identifier))
            assert schema["additionalProperties"] is False
            assert "section_updates" not in schema["properties"]


def test_real_criterion_progress_round_trip_preserves_completed_fragment() -> None:
    requirement = _requirements(1)[0]
    criteria = criterion_fragments.requirement_acceptance_criteria(requirement)
    fragment = _real_fragment(evidence_ref="ev_001")
    stored = criterion_fragments.store_criterion_progress(
        _base_state(),
        requirement_ref="req_1",
        selected_sections=WORKSHEET_SECTIONS,
        criterion_index=0,
        criterion=criteria[0],
        fragment=fragment,
    )

    restored = criterion_fragments.load_requirement_progress(
        stored,
        requirement_ref="req_1",
        selected_sections=WORKSHEET_SECTIONS,
        criteria=criteria,
        allowed_refs={"ev_001"},
    )

    assert set(restored) == {0}
    assert restored[0] == criterion_fragments.validate_criterion_fragment(
        fragment,
        selected_sections=WORKSHEET_SECTIONS,
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
        selected_sections=WORKSHEET_SECTIONS,
        criteria=criteria,
        fragments=fragments,
        allowed_refs={"ev_001"},
    )

    assert tuple(worksheet) == tuple(WORKSHEET_SECTIONS)
    for section in WORKSHEET_SECTIONS:
        assert worksheet[section]["constraint_evidence_refs"] == ["ev_001"]
        assert isinstance(worksheet[section]["specification"], dict)
        assert worksheet[section]["specification"]["inapplicable_concerns"] == []


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


def test_missing_requirement_sections_are_generated_and_checkpointed(monkeypatch):
    requirements = _requirements()
    _patch_compile_boundaries(monkeypatch, requirements)
    monkeypatch.setattr(
        adaptive,
        "assemble_worksheet_from_fragments",
        criterion_fragments.assemble_worksheet_from_fragments,
    )
    fragment = _real_fragment()
    fragment["section_updates"] = [
        row
        for row in fragment["section_updates"]
        if row["section"] not in {"reuse_assessment", "verification"}
    ]
    monkeypatch.setattr(adaptive, "_compile_criterion", lambda *a, **kw: deepcopy(fragment))
    calls = []
    evidence = [
        {"research_ref": "r_1", "claims": ["A donor exists but compatibility is unverified."]}
    ]
    monkeypatch.setattr(adaptive, "_requirement_grounding", lambda *a: (evidence, set()))

    def complete(_router, **kwargs):
        assert kwargs["evidence"] == evidence
        section = kwargs["target_section"]
        calls.append(section)
        return {
            "section_updates": [
                row
                for row in _real_fragment()["section_updates"]
                if row["section"] == section
            ]
        }

    monkeypatch.setattr(adaptive, "generate_targeted_section_fragment", complete)
    checkpoints = []
    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        _base_state(),
        required_sections_by_requirement={"req_1": WORKSHEET_SECTIONS},
        checkpoint=lambda state: checkpoints.append(deepcopy(state)),
    )
    assert calls == ["reuse_assessment", "verification"]
    assert result["plan_ready"] is True
    worksheet = result["decisions"][0]["engineering_worksheet"]
    assert worksheet["reuse_assessment"]["specification"]["verdicts"]
    assert worksheet["verification"]["specification"]["success_cases"]
    assert any(len(row.get("detail_progress", [])) == 1 for row in checkpoints)


def test_resume_rejects_legacy_plan_with_vacuous_required_section():
    worksheet = _worksheet()
    detail = _plan("req_1", WORKSHEET_SECTIONS, worksheet)
    assert adaptive._detail_matches_selection(detail, WORKSHEET_SECTIONS)
    spec = detail["engineering_worksheet"]["verification"]["specification"]
    for concern in list(spec):
        if concern != "inapplicable_concerns":
            spec[concern] = []
    assert not adaptive._detail_matches_selection(detail, WORKSHEET_SECTIONS)


def test_transport_interruption_preserves_individual_records_and_resumes(monkeypatch):
    requirements = _requirements()
    _patch_compile_boundaries(monkeypatch, requirements)
    monkeypatch.setattr(adaptive, "router_native_model_parallelism", lambda _: 1)
    checkpoints: list[dict[str, object]] = []
    authored: list[str] = []
    interrupted = False

    def compile_criterion(
        _router,
        *,
        requirement_ref,
        criterion_index,
        progress,
        record_checkpoint,
        **_kwargs,
    ):
        nonlocal interrupted
        del requirement_ref, criterion_index
        progress = dict(progress or {})
        for binding in ("concern:first", "concern:second", "concern:third"):
            if binding in progress:
                continue
            authored.append(binding)
            record_checkpoint(binding, {"status": "accepted", "binding": binding})
            if binding == "concern:second" and not interrupted:
                interrupted = True
                raise TimeoutError("interrupted within a concern")
        return _real_fragment()

    monkeypatch.setattr(adaptive, "_compile_criterion", compile_criterion)
    kwargs = dict(
        required_sections_by_requirement={"req_1": WORKSHEET_SECTIONS},
        checkpoint=lambda state: checkpoints.append(deepcopy(state)),
    )

    with pytest.raises(TimeoutError):
        adaptive.compile_progress_monotone_detailed_plans(
            _Router(), "request", _base_state(), **kwargs
        )

    saved = checkpoints[-1]
    assert not saved["blockers"]
    assert set(saved["template_progress"]) == {"concern:first", "concern:second"}
    assert authored == ["concern:first", "concern:second"]

    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(), "request", saved, **kwargs
    )
    assert result["plan_ready"]
    assert authored == ["concern:first", "concern:second", "concern:third"]
    assert result["template_progress"]["concern:first"]["status"] == "accepted"
    assert result["template_progress"]["concern:second"]["status"] == "accepted"
    assert result["template_progress"]["concern:third"]["status"] == "accepted"


def test_concurrent_record_checkpoints_do_not_overwrite_sibling_progress(monkeypatch):
    from threading import Barrier

    requirements = _requirements(2)
    _patch_compile_boundaries(monkeypatch, requirements)
    barrier = Barrier(2)
    checkpoints = []

    def compile_criterion(_router, *, requirement_ref, record_checkpoint, **kwargs):
        record_checkpoint(requirement_ref, [{"accepted": requirement_ref}])
        barrier.wait(timeout=5)
        return _fragment(requirement_ref, 0)

    monkeypatch.setattr(adaptive, "_compile_criterion", compile_criterion)
    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "request",
        _base_state(),
        required_sections_by_requirement={
            r["requirement_id"]: WORKSHEET_SECTIONS for r in requirements
        },
        checkpoint=lambda state: checkpoints.append(deepcopy(state)),
    )
    assert set(result["template_progress"]) == {"req_1", "req_2"}
    assert result["plan_ready"]
    sizes = [len(s.get("template_progress", {})) for s in checkpoints]
    assert sizes == sorted(sizes)