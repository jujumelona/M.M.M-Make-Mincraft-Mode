from __future__ import annotations

from typing import Any

import minecraft_mod_ai.planning_state_adaptive_implementation as adaptive


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
        evidence: list[dict[str, Any]],
        allowed_refs: set[str],
    ) -> dict[str, Any]:
        del requirement, criterion, evidence, allowed_refs
        selection = tuple(selected_sections)
        repair_selections.append(selection)
        assert len(selection) == 1
        section = selection[0]
        assert section in implementations
        return {
            "section_updates": [
                {
                    "section": section,
                    "implementation": implementations[section],
                    "constraint": f"Do not commit an invalid {section} result.",
                    "evidence_refs": [],
                }
            ]
        }

    monkeypatch.setattr(adaptive, "generate_criterion_fragment", generate_targeted_fragment)
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
        "selected_sections": (
            "behavior_contract",
            "integration",
            "persistence",
            "reuse_assessment",
        ),
        "criteria": ("A successful collection adds the resource to inventory.",),
        "fragments": {
            0: {
                "section_updates": [
                    {
                        "section": "behavior_contract",
                        "implementation": "A successful collection exposes the acquired resource.",
                        "constraint": "Rejected collection leaves inventory unchanged.",
                        "evidence_refs": [],
                    }
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
