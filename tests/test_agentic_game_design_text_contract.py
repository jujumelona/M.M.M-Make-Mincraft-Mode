from __future__ import annotations

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.agentic_research_game_design import generate_sectioned_game_design


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
    router = _Router(
        [
            design._section_field_body(raw, field, fields)
            for raw, (_, fields, _) in zip(
                _outputs(), design._SECTION_SPECS, strict=True
            )
            for field in fields
        ]
    )
    result = generate_sectioned_game_design(
        _GameDesignModule,
        router,
        "우주 탐사 모드를 만들어줘",
        research={},
    )

    assert result["title"] == "Orbital Frontier"
    assert result["progression"] == ["궤도 진입", "달 기지", "심우주 탐사"]
    assert result["assets"][0]["id"] == "orbital_console"
    assert len(router.calls) == 10
    for call in router.calls:
        assert call["response_format"] == "text"
        assert call["response_schema"] is None
        system = call["messages"][0]["content"]
        assert "No JSON" in system


def test_missing_heading_repairs_only_missing_field():
    router = _Router(
        ["Orbital Frontier", "행성을 탐사한다.", "", "- 탐사하고 귀환한다"]
    )
    section = design._generate_section(
        router,
        prompt="우주 탐사 모드를 만들어줘",
        section_id="identity_and_loop",
        fields=("title", "pitch", "core_loop"),
        research={},
        media_paths=(),
        trace_metadata=None,
    )
    assert section["title"] == "Orbital Frontier"
    assert section["core_loop"] == ["탐사하고 귀환한다"]
    assert len(router.calls) == 4
    assert "content is missing" in router.calls[-1]["messages"][-1]["content"]
    assert [
        call["messages"][1]["content"].split("FIELD\n")[1].split("\n")[0]
        for call in router.calls
    ] == ["title", "pitch", "core_loop", "core_loop"]
