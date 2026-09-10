from __future__ import annotations

from typing import Any

import minecraft_mod_ai.planning_state_adaptive_implementation as adaptive
from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS


def authored(section, description):
    spec = {concern: [{field: description for field in columns.split()}]
            for concern, columns in DETAIL_RECORDS[section].items()}
    spec["inapplicable_concerns"] = []
    return {"section": section, "specification": spec, "constraint_evidence_refs": []}


def test_missing_worksheet_section_repairs_expose_only_each_missing_section(monkeypatch) -> None:
    """Reproduce the integration/persistence/reuse gap from the local-model planner crash."""

    repair_selections: list[tuple[str, ...]] = []
    implementations = {
        "integration": (
            "Route the accepted resource-farming result through the requirement-owned "
            "integration boundary before exposing the observable result."
        ),
        "persistence": (
            "Persist requirement-owned progression only after the accepted state transition commits."
        ),
        "reuse_assessment": (
            "Treat supplied repository candidates as reference-only unless the host evidence proves reuse."
        ),
    }

    def generate_targeted_fragment(
        _router: Any,
        *,
        requirement: dict[str, Any],
        criterion: str,
        selected_sections: tuple[str, ...],
        target_section: str,
        evidence: list[dict[str, Any]],
        allowed_refs: set[str],
        progress, checkpoint,
    ) -> dict[str, Any]:
        del requirement, criterion, evidence, allowed_refs
        assert tuple(selected_sections) == WORKSHEET_SECTIONS
        section = target_section
        repair_selections.append((section,))
        assert section in implementations
        return {
            "section_updates": [
                authored(section, implementations[section])
            ]
        }

    monkeypatch.setattr(adaptive, "generate_targeted_section_fragment", generate_targeted_fragment)
    monkeypatch.setattr(
        adaptive,
        "store_criterion_progress",
        lambda state, **_kwargs: dict(state),
    )
    monkeypatch.setattr(
        adaptive,
        "clear_requirement_progress",
        lambda state, _requirement_ref: dict(state),
    )
    monkeypatch.setattr(
        adaptive,
        "_checkpoint_state",
        lambda state, _checkpoint: dict(state),
    )
    monkeypatch.setattr(
        adaptive,
        "_assemble_requirement_plan",
        lambda _requirement, requirement_ref, selected, worksheet, _allowed: {
            "requirement_ref": requirement_ref,
            "required_detail_sections": list(selected),
            "engineering_worksheet": worksheet,
        },
    )
    monkeypatch.setattr(
        adaptive,
        "_merge_completed_details",
        lambda state, **_kwargs: dict(state),
    )

    job = {
        "requirement": {
            "requirement_id": "req_001",
            "statement": "Collected resources enter the player inventory.",
        },
        "requirement_ref": "req_001",
        "selected_sections": WORKSHEET_SECTIONS,
        "criteria": ("A successful collection adds the resource to inventory.",),
        "fragments": {
            0: {
                "section_updates": [
                    authored(section, f"Existing concrete contract for {section}.")
                    for section in WORKSHEET_SECTIONS
                    if section not in {"integration", "persistence", "reuse_assessment"}
                ]
            }
        },
        "evidence": [],
        "allowed": set(),
    }
    completed: dict[str, Any] = {}

    adaptive._finish_requirement(
        job,
        object(),
        working_state={},
        requirement_order=("req_001",),
        completed_details=completed,
        checkpoint=None,
    )

    assert repair_selections == [
        ("integration",),
        ("persistence",),
        ("reuse_assessment",),
    ]
    assert "req_001" in completed
    worksheet = completed["req_001"]["engineering_worksheet"]
    assert worksheet["integration"]["specification"]["responsibilities"]
    assert worksheet["persistence"]["specification"]["stored_state"]
    assert worksheet["reuse_assessment"]["specification"]["verdicts"]


def test_repairs_only_missing_criterion_section_and_preserves_existing_records(monkeypatch):
    from copy import deepcopy
    from minecraft_mod_ai.planning_criterion_fragments import store_criterion_progress
    criteria = ('First behavior succeeds.', 'Second behavior succeeds.')
    fragments = {index: {'section_updates': [authored(section, f'original {index} {section}')
                 for section in WORKSHEET_SECTIONS
                 if not (index == 1 and section == 'persistence')]}
                 for index in range(2)}
    first = deepcopy(fragments[0])
    calls, snapshots = [], []

    def generate(_router, **kwargs):
        calls.append((kwargs['criterion'], kwargs['target_section']))
        return {'section_updates': [authored(kwargs['target_section'], 'repaired second criterion')]}

    monkeypatch.setattr(adaptive, 'generate_targeted_section_fragment', generate)
    monkeypatch.setattr(adaptive, '_checkpoint_state', lambda state, cb: (cb(deepcopy(state)), state)[1])
    monkeypatch.setattr(adaptive, '_merge_completed_details', lambda state, **kw: state)
    monkeypatch.setattr(adaptive, '_assemble_requirement_plan', lambda *args: {})
    job = {'requirement': {'requirement_id': 'req'}, 'requirement_ref': 'req',
           'selected_sections': WORKSHEET_SECTIONS, 'criteria': criteria,
           'fragments': fragments, 'evidence': [], 'allowed': set()}
    state = {}
    for index in range(2):
        state = store_criterion_progress(state, requirement_ref='req',
            selected_sections=WORKSHEET_SECTIONS, criterion_index=index,
            criterion=criteria[index], fragment=fragments[index])
    adaptive._finish_requirement(job, None, working_state=state, requirement_order=('req',),
                                 completed_details={}, checkpoint=snapshots.append)
    assert calls == [(criteria[1], 'persistence')]
    assert job['fragments'][0] == first
    assert any(row['criterion_index'] == 1 and any(
        r['section'] == 'persistence' for r in row['fragment']['section_updates'])
        for snapshot in snapshots for row in snapshot.get('detail_progress', []))


def test_section_repair_resumes_an_interrupted_concern(monkeypatch):
    import json
    from copy import deepcopy
    import pytest
    from minecraft_mod_ai import task_template_runner as runner
    from minecraft_mod_ai.planning_criterion_fragments import store_criterion_progress, load_requirement_progress

    criteria = ('State survives restart.',)
    fragment = {'section_updates': [authored(section, f'original {section}')
                for section in WORKSHEET_SECTIONS if section != 'persistence']}
    job = {'requirement': {'requirement_id': 'req'}, 'requirement_ref': 'req',
           'selected_sections': WORKSHEET_SECTIONS, 'criteria': criteria,
           'fragments': {0: fragment}, 'evidence': [], 'allowed': set()}
    state = store_criterion_progress({}, requirement_ref='req', selected_sections=WORKSHEET_SECTIONS,
                                     criterion_index=0, criterion=criteria[0], fragment=fragment)
    snapshots, calls = [], []
    monkeypatch.setattr(adaptive, '_checkpoint_state', lambda value, cb: (cb(deepcopy(value)), value)[1])
    monkeypatch.setattr(adaptive, '_merge_completed_details', lambda value, **kw: value)
    monkeypatch.setattr(adaptive, '_assemble_requirement_plan', lambda *args: {})

    def generate(*args, response_schema, tool_name, **kwargs):
        context = json.loads(args[2][1]['content'])
        calls.append((tool_name, len(context['accepted_records'])))
        if len(calls) == 2:
            raise TimeoutError('repair interrupted')
        record = None if context['accepted_records'] else {
            field: f'authored {field}'
            for field in response_schema['properties']['record']['anyOf'][0]['required']}
        return {'status': 'record' if record is not None else 'done', 'record': record,
                'reason': '', 'evidence_refs': []}

    monkeypatch.setattr(runner, 'generate_fixed_template_value', generate)
    kwargs = dict(requirement_order=('req',), completed_details={}, checkpoint=snapshots.append)
    with pytest.raises(TimeoutError):
        adaptive._finish_requirement(job, None, working_state=state, **kwargs)
    resumed = deepcopy(snapshots[-1])
    assert len(resumed['template_progress']) == 1
    job['fragments'] = load_requirement_progress(resumed, requirement_ref='req',
        selected_sections=WORKSHEET_SECTIONS, criteria=criteria, allowed_refs=set())
    adaptive._finish_requirement(job, None, working_state=resumed, **kwargs)
    assert calls[2] == calls[1]
    assert calls.count(calls[0]) == 1
