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
