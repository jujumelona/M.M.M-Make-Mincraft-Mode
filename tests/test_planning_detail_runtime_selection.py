from __future__ import annotations

import pytest

from minecraft_mod_ai import planning_state_implementation as planning_impl
from minecraft_mod_ai.planning_detail_template import (
    CORE_WORKSHEET_SECTIONS,
    WORKSHEET_SECTIONS,
)
from minecraft_mod_ai.planning_state_implementation import _host_section_selection


def _requirements() -> list[dict[str, str]]:
    return [
        {"requirement_id": "REQ-1"},
        {"requirement_id": "REQ-2"},
    ]


def test_host_selection_defaults_every_requirement_to_full_contract() -> None:
    selected = _host_section_selection(_requirements(), None)

    assert selected == {
        "REQ-1": WORKSHEET_SECTIONS,
        "REQ-2": WORKSHEET_SECTIONS,
    }


def test_host_selection_can_narrow_one_requirement_without_weakening_others() -> None:
    selected = _host_section_selection(
        _requirements(),
        {"REQ-1": tuple(reversed(CORE_WORKSHEET_SECTIONS))},
    )

    assert selected["REQ-1"] == CORE_WORKSHEET_SECTIONS
    assert selected["REQ-2"] == WORKSHEET_SECTIONS


def test_host_selection_rejects_unknown_requirement() -> None:
    with pytest.raises(ValueError, match="unknown requirement"):
        _host_section_selection(
            _requirements(),
            {"REQ-NOT-REAL": CORE_WORKSHEET_SECTIONS},
        )


def test_host_selection_cannot_omit_core_section() -> None:
    selection = tuple(key for key in WORKSHEET_SECTIONS if key != "verification")

    with pytest.raises(ValueError, match="core section.*cannot be omitted"):
        _host_section_selection(_requirements(), {"REQ-1": selection})


def test_host_selection_rejects_model_shaped_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="host selection must be a requirement mapping"):
        _host_section_selection(_requirements(), [CORE_WORKSHEET_SECTIONS])


def test_single_requirement_uses_one_requirement_batch_even_with_parallel_capacity(
    monkeypatch,
) -> None:
    requirement = {
        "requirement_id": "REQ-1",
        "statement": "Persist the approved gameplay state safely.",
        "acceptance": ["The gameplay state remains observable after reload."],
    }
    state = {
        "research_queue": [
            {
                "research_id": "research_001",
                "requirement_ref": "REQ-1",
                "status": "complete",
            }
        ],
        "evidence": [
            {
                "research_ref": "research_001",
                "sufficient": True,
                "claims": ["The approved behavior has grounded implementation evidence."],
                "evidence_refs": ["ev:1"],
            }
        ],
    }
    batch_calls: list[tuple[str, ...]] = []

    def fake_batch_plain_sections(
        _router,
        *,
        requirement,
        selected_sections,
        evidence,
    ):
        del requirement, evidence
        batch_calls.append(selected_sections)
        return {
            section: (
                f"{section} defines one concrete authoritative behavior with bounded failure "
                "handling and an observable verification outcome for the approved requirement."
            )
            for section in selected_sections
        }

    monkeypatch.setattr(
        planning_impl,
        "_batch_plain_sections",
        fake_batch_plain_sections,
    )
    plans = planning_impl._compile_requirement_plans_parallel(
        object(),
        state,
        [requirement],
        {"REQ-1": CORE_WORKSHEET_SECTIONS},
        workers=2,
    )

    assert len(plans) == 1
    assert batch_calls == [CORE_WORKSHEET_SECTIONS]
    assert tuple(plans[0]["engineering_worksheet"]) == CORE_WORKSHEET_SECTIONS
