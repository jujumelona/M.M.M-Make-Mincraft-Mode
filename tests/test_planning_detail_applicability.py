from __future__ import annotations

import pytest

import minecraft_mod_ai.planning_state_pipeline as planning_state_pipeline
from minecraft_mod_ai.planning_detail_applicability import (
    APPLICABILITY_FIELD,
    normalize_detail_section_applicability,
    required_detail_sections_for_requirement,
    required_sections_by_requirement,
)
from minecraft_mod_ai.planning_detail_template import (
    CORE_WORKSHEET_SECTIONS,
    WORKSHEET_SECTIONS,
)


def _requirement(
    requirement_id: str = "REQ-1",
    *,
    applicability: dict[str, str] | None = None,
    statement: str = "Do one bounded thing.",
) -> dict[str, object]:
    row: dict[str, object] = {
        "decision_type": "requirement",
        "requirement_id": requirement_id,
        "statement": statement,
    }
    if applicability is not None:
        row[APPLICABILITY_FIELD] = applicability
    return row


def _pipeline_state() -> dict[str, object]:
    return {
        "plan_ready": False,
        "decisions": [_requirement("REQ-1"), _requirement("REQ-2")],
        "unresolved": [],
        "research_queue": [],
        "blockers": [],
        "evidence": [],
    }


def _patch_pipeline_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planning_state_pipeline,
        "_transition",
        lambda _operation, callback: callback(),
    )
    monkeypatch.setattr(
        planning_state_pipeline,
        "validate_planning_state",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        planning_state_pipeline,
        "collect_planning_state_research",
        lambda _router, _prompt, value, **_kwargs: value,
    )


def test_missing_applicability_keeps_full_fail_safe_contract() -> None:
    assert required_detail_sections_for_requirement(_requirement()) == WORKSHEET_SECTIONS


def test_unknown_applicability_keeps_conditional_sections() -> None:
    selected = required_detail_sections_for_requirement(
        _requirement(
            applicability={
                "authority_and_network": "unknown",
                "persistence": "unknown",
                "resources_and_ui": "unknown",
            }
        )
    )

    assert selected == WORKSHEET_SECTIONS


def test_only_explicit_not_applicable_omits_conditional_sections() -> None:
    selected = required_detail_sections_for_requirement(
        _requirement(
            applicability={
                "authority_and_network": "not_applicable",
                "persistence": "not_applicable",
                "resources_and_ui": "not_applicable",
            }
        )
    )

    assert selected == CORE_WORKSHEET_SECTIONS


def test_required_and_unknown_conditionals_are_retained() -> None:
    selected = required_detail_sections_for_requirement(
        _requirement(
            applicability={
                "authority_and_network": "required",
                "persistence": "not_applicable",
                "resources_and_ui": "unknown",
            }
        )
    )

    assert "authority_and_network" in selected
    assert "resources_and_ui" in selected
    assert "persistence" not in selected
    assert all(section in selected for section in CORE_WORKSHEET_SECTIONS)


def test_prompt_wording_never_omits_a_section() -> None:
    requirement = _requirement(
        statement=(
            "This is local-only, has no networking, never persists, and needs no UI or resources."
        )
    )

    assert required_detail_sections_for_requirement(requirement) == WORKSHEET_SECTIONS


def test_state_projection_uses_each_requirement_owned_applicability() -> None:
    state = {
        "decisions": [
            _requirement(
                "REQ-1",
                applicability={
                    "authority_and_network": "not_applicable",
                    "persistence": "not_applicable",
                    "resources_and_ui": "not_applicable",
                },
            ),
            _requirement("REQ-2"),
        ]
    }

    assert required_sections_by_requirement(state) == {
        "REQ-1": CORE_WORKSHEET_SECTIONS,
        "REQ-2": WORKSHEET_SECTIONS,
    }


def test_pipeline_passes_requirement_owned_selection_to_detailed_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _pipeline_state()
    state["decisions"] = [
        _requirement(
            "REQ-1",
            applicability={
                "authority_and_network": "not_applicable",
                "persistence": "not_applicable",
                "resources_and_ui": "not_applicable",
            },
        ),
        _requirement("REQ-2"),
    ]
    captured: dict[str, object] = {}

    _patch_pipeline_shell(monkeypatch)
    monkeypatch.setattr(
        planning_state_pipeline,
        "ensure_host_detail_section_applicability",
        lambda value: value,
    )

    def _compile(
        _router: object,
        _prompt: str,
        value: dict[str, object],
        *,
        required_sections_by_requirement: dict[str, tuple[str, ...]],
    ) -> dict[str, object]:
        captured["selection"] = required_sections_by_requirement
        result = dict(value)
        result["plan_ready"] = True
        return result

    monkeypatch.setattr(
        planning_state_pipeline,
        "compile_detailed_implementation_plans",
        _compile,
    )

    result = planning_state_pipeline.prepare_planning_state(
        object(),
        "bounded prompt",
        existing_state=state,
    )

    assert captured["selection"] == {
        "REQ-1": CORE_WORKSHEET_SECTIONS,
        "REQ-2": WORKSHEET_SECTIONS,
    }
    assert result["plan_ready"] is True


def test_host_resolver_receives_only_requirement_ids_and_drives_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _pipeline_state()
    captured: dict[str, object] = {}

    _patch_pipeline_shell(monkeypatch)

    def _apply(
        value: dict[str, object],
        applicability_by_requirement: dict[str, dict[str, str]],
    ) -> dict[str, object]:
        result = dict(value)
        decisions = []
        for raw in value["decisions"]:
            item = dict(raw)
            requirement_id = str(item["requirement_id"])
            if requirement_id in applicability_by_requirement:
                item[APPLICABILITY_FIELD] = applicability_by_requirement[requirement_id]
            decisions.append(item)
        result["decisions"] = decisions
        return result

    monkeypatch.setattr(
        planning_state_pipeline,
        "apply_host_detail_section_applicability",
        _apply,
    )

    def _compile(
        _router: object,
        _prompt: str,
        value: dict[str, object],
        *,
        required_sections_by_requirement: dict[str, tuple[str, ...]],
    ) -> dict[str, object]:
        captured["selection"] = required_sections_by_requirement
        result = dict(value)
        result["plan_ready"] = True
        return result

    monkeypatch.setattr(
        planning_state_pipeline,
        "compile_detailed_implementation_plans",
        _compile,
    )

    def _resolver(requirement_ids: tuple[str, ...]) -> dict[str, dict[str, str]]:
        captured["resolver_input"] = requirement_ids
        return {
            "REQ-1": {
                "authority_and_network": "not_applicable",
                "persistence": "not_applicable",
                "resources_and_ui": "not_applicable",
            }
        }

    result = planning_state_pipeline.prepare_planning_state(
        object(),
        "prompt text must not be resolver input",
        existing_state=state,
        detail_section_applicability_resolver=_resolver,
    )

    assert captured["resolver_input"] == ("REQ-1", "REQ-2")
    assert captured["selection"] == {
        "REQ-1": CORE_WORKSHEET_SECTIONS,
        "REQ-2": WORKSHEET_SECTIONS,
    }
    assert result["plan_ready"] is True


def test_missing_status_keys_fail_closed_to_unknown() -> None:
    normalized = normalize_detail_section_applicability(
        {"persistence": "not_applicable"}
    )

    assert normalized == {
        "authority_and_network": "unknown",
        "persistence": "not_applicable",
        "resources_and_ui": "unknown",
    }


@pytest.mark.parametrize(
    "value, message",
    [
        (["persistence"], "section mapping"),
        ({"made_up": "not_applicable"}, "unknown conditional section"),
        ({"persistence": "optional"}, "invalid status"),
    ],
)
def test_invalid_host_applicability_is_rejected(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_detail_section_applicability(value)
