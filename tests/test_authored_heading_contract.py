from __future__ import annotations

import re

import pytest

from minecraft_mod_ai.complete_planner import (
    CompleteGameDesignPlanner,
    _design_writing_template,
)
from minecraft_mod_ai.authored_document_contract import normalize_authored_document
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.implementation_ir import (
    ImplementationGraphError,
    decompose_authored_units,
)


def _legacy_planner_design() -> str:
    return (
        "# StarForge Odyssey: 게임 설계 문서\n"
        "## 개요\n"
        "우주선 제작과 거래를 설명한다.\n"
        "# behavior_contract\n"
        "플레이어는 자원을 거래해 부품을 구매한다.\n"
        "# state_model\n"
        "크레딧과 우주선 상태를 저장한다.\n"
        "# algorithm\n"
        "가격과 업그레이드 계산을 정의한다.\n"
        "# integration\n"
        "게임 초기화 경로에 시스템을 연결한다.\n"
        "# authority_and_network\n"
        "서버가 거래와 상태 변경을 승인한다.\n"
        "# persistence\n"
        "플레이어 진행 상태를 저장하고 복원한다.\n"
        "# resources_and_ui\n"
        "거래 UI와 필요한 리소스를 제공한다.\n"
        "# failure_and_limits\n"
        "잔액 부족과 잘못된 요청을 거부한다.\n"
        "# reuse_assessment\n"
        "재사용 가능성을 기록한다.\n"
        "# verification\n"
        "거래 성공과 실패 경로를 검증한다.\n"
    )


def test_design_writing_template_emits_canonical_sections_at_h2() -> None:
    template = _design_writing_template(
        {
            "behavior_contract": {"actors": ("actor",)},
            "state_model": {"variables": ("name", "owner")},
        }
    )

    assert template.startswith("## behavior_contract\n")
    assert "\n## state_model\n" in template
    assert re.search(r"(?m)^# (?:behavior_contract|state_model)$", template) is None


def test_planner_remains_free_markdown_and_does_not_compile_structured_state() -> None:
    class Router:
        def __init__(self) -> None:
            self.calls = []

        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, messages, kwargs))
            return _legacy_planner_design()

    router = Router()
    plan = CompleteGameDesignPlanner(router).plan("space mod")

    assert plan.text == _legacy_planner_design()
    assert plan.structured_sections == {}
    assert len(router.calls) == 1
    role, _messages, kwargs = router.calls[0]
    assert role == "planner"
    assert kwargs["response_format"] == "text"
    assert kwargs["response_schema"] is None
    assert kwargs["enable_tools"] is False

def test_legacy_planner_heading_layout_is_migrated_without_mutating_requirements() -> None:
    units = decompose_authored_units(_legacy_planner_design())

    assert [unit["unit_id"] for unit in units] == [
        "state_model",
        "behavior_contract",
        "algorithm",
        "authority_and_network",
        "persistence",
        "resources_and_ui",
        "failure_and_limits",
        "integration",
    ]
    behavior = next(unit for unit in units if unit["unit_id"] == "behavior_contract")
    assert "# behavior_contract" in behavior["requirements"].values()


def test_depth_compatibility_is_limited_to_exact_legacy_planner_shape() -> None:
    malformed = (
        "## behavior_contract\nA\n"
        "# state_model\nB\n"
        "# algorithm\nC\n"
    )

    with pytest.raises(
        ImplementationGraphError,
        match="IMPLEMENTATION_IR_AUTHORED_SECTION_DEPTH",
    ):
        decompose_authored_units(malformed)

def test_fresh_canonical_h2_layout_is_accepted_without_legacy_migration() -> None:
    sections = (
        "behavior_contract",
        "state_model",
        "algorithm",
        "integration",
        "authority_and_network",
        "persistence",
        "resources_and_ui",
        "failure_and_limits",
    )
    text = "\n".join(f"## {section}\n{section} details." for section in sections)

    units = decompose_authored_units(text)

    assert {unit["unit_id"] for unit in units} == set(sections)



def test_structured_renderer_omits_inapplicable_execution_sections() -> None:
    from minecraft_mod_ai.authored_structured_design import render_structured_sections

    state_spec = {
        **{name: [] for name in DETAIL_RECORDS["state_model"]},
        "variables": [{
            "name": "credits",
            "owner": "Player",
            "type": "Int",
            "unit": "credits",
            "default": "0",
            "domain": "non-negative",
        }],
        "inapplicable_concerns": [
            {"concern": name, "reason": "not required"}
            for name in DETAIL_RECORDS["state_model"]
            if name != "variables"
        ],
    }
    text = render_structured_sections({
        "state_model": {
            "specification": state_spec,
            "constraint_evidence_refs": [],
        }
    })

    assert text.startswith("## state_model\n")
    assert "- variables:" in text
    assert "## authority_and_network" not in text
    assert "## persistence" not in text


def test_canonicalizer_repairs_duplicate_order_and_unknown_peer_headings() -> None:
    malformed = (
        "## state_model\nState A.\n"
        "## behavior_contract\nBehavior.\n"
        "## behavior_contract\nBehavior B.\n"
        "## custom_notes\nKeep this note.\n"
        "## algorithm\nAlgorithm.\n"
        "## integration\nIntegration.\n"
        "## resources_and_ui\nResources.\n"
        "## failure_and_limits\nFailures.\n"
    )

    normalized, report = normalize_authored_document(malformed)

    assert report is not None
    assert report["merged_duplicate_sections"] == ["behavior_contract"]
    assert set(report["missing_execution_sections"]) == {
        "authority_and_network", "persistence"
    }
    assert normalized.count("## behavior_contract\n") == 1
    assert "## authority_and_network\n" not in normalized
    assert "## persistence\n" not in normalized
    assert "### custom_notes\nKeep this note." in normalized
    decompose_authored_units(normalized)


def test_canonicalizer_preserves_already_valid_document_bytes() -> None:
    text = _legacy_planner_design()

    normalized, report = normalize_authored_document(text)

    assert normalized == text
    assert report is None


def test_canonicalizer_promotes_nested_execution_contracts_without_losing_semantics() -> None:
    malformed = (
        "## behavior_contract\nTrade ore.\n"
        "## state_model\nCredits and ship state.\n"
        "## algorithm\nCalculate prices.\n"
        "## integration\n"
        "- authority_and_network:\n"
        "  - Server authoritatively validates trade requests.\n"
        "  - Client to server packets carry requested trade actions.\n"
        "- integration_status:\n"
        "  - Register runtime hooks.\n"
        "## resources_and_ui\nTrade screen.\n"
        "## failure_and_limits\nReject bad requests.\n"
        "## verification\n"
        "- persistence_cases:\n"
        "  - Logout and login restores credits and ship state from saved data.\n"
        "- runtime_cases:\n"
        "  - Exercise normal gameplay.\n"
    )

    normalized, report = normalize_authored_document(malformed)

    assert report is not None
    assert report["promoted_nested_sections"] == [
        "authority_and_network",
        "persistence",
    ]
    assert "authority_and_network" not in report["missing_execution_sections"]
    assert "persistence" not in report["missing_execution_sections"]
    authority_start = normalized.index("## authority_and_network")
    persistence_start = normalized.index("## persistence")
    resources_start = normalized.index("## resources_and_ui")
    assert "Server authoritatively validates trade requests." in normalized[
        authority_start:persistence_start
    ]
    assert "Logout and login restores credits and ship state from saved data." in normalized[
        persistence_start:resources_start
    ]
    decompose_authored_units(normalized)


def test_valid_nested_canonical_heading_does_not_rewrite_document() -> None:
    text = (
        "## behavior_contract\nTrade ore.\n"
        "### persistence\nThis behavior note mentions persistence but stays nested.\n"
        "## state_model\nCredits.\n"
        "## algorithm\nCalculate prices.\n"
        "## integration\nWire systems.\n"
        "## authority_and_network\nServer validates trades.\n"
        "## persistence\nPersist credits.\n"
        "## resources_and_ui\nTrade screen.\n"
        "## failure_and_limits\nReject bad requests.\n"
    )

    normalized, report = normalize_authored_document(text)

    assert normalized == text
    assert report is None


def test_empty_authored_document_is_rejected_instead_of_becoming_placeholders() -> None:
    from minecraft_mod_ai.authored_document_contract import (
        AuthoredDocumentContractError,
    )

    with pytest.raises(AuthoredDocumentContractError, match="AUTHORED_DOCUMENT_EMPTY"):
        normalize_authored_document("  \n\t")
