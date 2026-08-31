from __future__ import annotations

import pytest

from minecraft_mod_ai.agentic_research_game_design import generate_sectioned_game_design
from minecraft_mod_ai.spec import SpecValidationError


class _Router:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, **kwargs})
        return self.outputs.pop(0)


class _GameDesignModule:
    @staticmethod
    def _validate_design(value):
        assert isinstance(value["title"], str)
        assert isinstance(value["core_loop"], list)
        assert isinstance(value["progression"], list)
        assert isinstance(value["combat"], dict)
        assert isinstance(value["mod_context"], dict)
        assert isinstance(value["modules"], list)
        assert isinstance(value["assets"], list)
        assert isinstance(value["acceptance_tests"], list)


def _outputs() -> list[str]:
    return [
        """## title
Orbital Frontier
## pitch
행성을 탐사하고 우주 기지를 확장한다.
## core_loop
- 탐사
- 자원 회수
- 기지 확장
""",
        """## progression
- 궤도 진입
- 달 기지
- 심우주 탐사
## combat
### hazards
- 방사선 폭풍
### enemies
- 적대 드론
## mod_context
### persistence
- 행성 진행도를 저장한다
""",
        """## modules
- none
## assets
- orbital_console | gui | 궤도 항법 콘솔
""",
        """## acceptance_tests
- 플레이어가 탐사와 귀환 루프를 완료할 수 있다
## art_direction
### palette
- 차가운 금속과 강한 경고 조명
""",
    ]


def test_game_design_drafting_is_text_not_json_schema():
    router = _Router(_outputs())
    design = generate_sectioned_game_design(
        _GameDesignModule,
        router,
        "우주 탐사 모드를 만들어줘",
        research={},
    )

    assert design["title"] == "Orbital Frontier"
    assert design["progression"] == ["궤도 진입", "달 기지", "심우주 탐사"]
    assert design["assets"][0]["id"] == "orbital_console"
    assert len(router.calls) == 4
    for call in router.calls:
        assert call["response_format"] == "text"
        assert call["response_schema"] is None
        system = call["messages"][0]["content"]
        assert "not JSON" in system


def test_missing_heading_fails_once_without_model_repair_loop():
    router = _Router([
        """## title
Orbital Frontier
## pitch
행성을 탐사한다.
"""
    ])

    with pytest.raises(SpecValidationError, match="core_loop"):
        generate_sectioned_game_design(
            _GameDesignModule,
            router,
            "우주 탐사 모드를 만들어줘",
            research={},
        )

    assert len(router.calls) == 1