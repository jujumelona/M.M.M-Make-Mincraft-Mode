from __future__ import annotations

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.spec import SpecValidationError


def test_labeled_module_records_accept_nested_obligations() -> None:
    body = """
### crystal_portal
- status: custom
- reason: 수정 수집과 포탈 진행을 구현한다.
- requirement_refs: req_collect, req_portal
- implementation_obligations:
  - 수정 조각 수집 상태를 저장한다.
  - 해금 조건을 만족하면 포탈을 활성화한다.

### travel_gate
- status: custom
- reason: 포탈을 통과한 플레이어를 새 지역으로 이동시킨다.
- requirement_refs:
  - req_travel
- implementation_obligations:
  - 서버 권한으로 목적지를 검증한다.
  - 플레이어 이동 결과를 검증한다.
"""

    rows = design._module_rows(body)

    assert [row["plugin_id"] for row in rows] == ["crystal_portal", "travel_gate"]
    assert rows[0]["requirement_refs"] == ["req_collect", "req_portal"]
    assert rows[0]["implementation_obligations"] == [
        "수정 조각 수집 상태를 저장한다.",
        "해금 조건을 만족하면 포탈을 활성화한다.",
    ]
    assert rows[1]["requirement_refs"] == ["req_travel"]
    assert len(rows[1]["implementation_obligations"]) == 2


def test_pipe_module_rows_tolerate_table_syntax_and_pipes_inside_reason() -> None:
    body = """
| plugin_id | status | reason | requirement_refs | implementation_obligations |
| --- | --- | --- | --- | --- |
| crystal_portal | custom | 수정 수집 | 포탈 해금 | req_collect, req_portal | 상태 저장; 포탈 활성화 |
"""

    rows = design._module_rows(body)

    assert rows == [
        {
            "plugin_id": "crystal_portal",
            "status": "custom",
            "reason": "수정 수집 | 포탈 해금",
            "requirement_refs": ["req_collect", "req_portal"],
            "implementation_obligations": ["상태 저장", "포탈 활성화"],
        }
    ]


def test_asset_records_accept_labeled_and_pipe_forms_with_pipe_in_brief() -> None:
    labeled = """
### portal_core
- kind: block
- brief: 포탈 중심부를 표현하는 블록 자산
"""
    pipe = "portal_frame | block | 외곽 프레임 | 발광 테두리 포함"

    assert design._asset_rows(labeled) == [
        {"id": "portal_core", "kind": "block", "brief": "포탈 중심부를 표현하는 블록 자산"}
    ]
    assert design._asset_rows(pipe) == [
        {"id": "portal_frame", "kind": "block", "brief": "외곽 프레임 | 발광 테두리 포함"}
    ]


def test_module_parser_remains_fail_closed_for_missing_semantic_obligations() -> None:
    body = """
### crystal_portal
- status: custom
- reason: 포탈을 구현한다.
- requirement_refs: req_portal
"""

    with pytest.raises(SpecValidationError, match="implementation_obligations"):
        design._module_rows(body)


def test_full_markdown_section_accepts_resilient_module_and_asset_records() -> None:
    raw = """
## modules
### crystal_portal
- status: custom
- reason: 수정 수집과 포탈 진행을 구현한다.
- requirement_refs: req_collect, req_portal
- implementation_obligations:
  - 수정 상태를 저장한다.
  - 포탈 해금을 처리한다.

## assets
### portal_core
- kind: block
- brief: 포탈 중심부 자산
"""

    section = design._parse_markdown_section(raw, ("modules", "assets"))

    assert section["modules"][0]["plugin_id"] == "crystal_portal"
    assert section["modules"][0]["requirement_refs"] == ["req_collect", "req_portal"]
    assert section["assets"] == [
        {"id": "portal_core", "kind": "block", "brief": "포탈 중심부 자산"}
    ]
