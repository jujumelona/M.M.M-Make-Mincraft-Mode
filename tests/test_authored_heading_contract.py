from __future__ import annotations

import re

import pytest

from minecraft_mod_ai.complete_planner import (
    CompleteGameDesignPlanner,
    _design_writing_template,
)
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


def test_planner_prompt_requires_canonical_sections_instead_of_allowing_omission() -> None:
    class Router:
        messages = ()

        def generate_text(self, _role, messages, **_kwargs):
            self.messages = messages
            return "## behavior_contract\nplaceholder\n"

    router = Router()
    CompleteGameDesignPlanner(router).plan("space mod")
    system_prompt = router.messages[0]["content"]

    assert "Use every canonical template section heading exactly once" in system_prompt
    assert "level 2 (`##`)" in system_prompt
    assert "leaving irrelevant parts aside" not in system_prompt


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
