"""Guard the Colab live runtime input path used by certified releases."""
from __future__ import annotations

import ast
import json
from pathlib import Path


_NOTEBOOK = Path(__file__).resolve().parents[1] / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"


def _cell(title: str) -> str:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source", ()))
        for cell in notebook.get("cells", ())
        if cell.get("cell_type") == "code"
        and "".join(cell.get("source", ())).startswith(title)
    ]
    assert len(matches) == 1, (title, len(matches))
    return matches[0]


def test_colab_runtime_configuration_is_editable_and_does_not_claim_a_receipt() -> None:
    settings = _cell("# @title 1. 실행 모드 및 설정")
    for field in (
        "RUN_RUNTIME",
        "RUN_CLIENT",
        "RUN_MINEFLAYER",
        "RUN_VISUAL_REVIEW",
        "SERVER_LAUNCHER",
        "PLAYTEST_ACTIONS_FILE",
    ):
        assert f"{field} =" in settings
        assert f"{field} =" in settings and "#@param" in next(
            line for line in settings.splitlines() if line.startswith(f"{field} =")
        )
    assert "RUN_RUNTIME = RUN_CLIENT = RUN_MINEFLAYER = RUN_VISUAL_REVIEW = False" not in settings


def test_colab_runtime_playtest_actions_flow_to_real_build_options() -> None:
    cell = _cell("# @title 6. 제작 또는 Audit")
    ast.parse(cell)
    for evidence in (
        "RUNTIME_SERVER_LAUNCHER_MISSING",
        "RUNTIME_SERVER_LAUNCHER_NOT_FOUND",
        "RUNTIME_PLAYTEST_ACTIONS_MISSING",
        "RUNTIME_PLAYTEST_ACTIONS_NOT_FOUND",
        "RUNTIME_PLAYTEST_ACTIONS_INVALID",
        "RUNTIME_PLAYTEST_ASSERTIONS_MISSING",
        "playtest_actions=_playtest_actions",
        "BUILD_RESULT = session.build(reply, run_name=RUN_NAME, options=options)",
    ):
        assert evidence in cell
    assert cell.index("RUNTIME_PLAYTEST_ACTIONS_MISSING") < cell.index("session.build(")


def test_colab_settings_cell_compiles_as_python() -> None:
    ast.parse(_cell("# @title 1. 실행 모드 및 설정"))
