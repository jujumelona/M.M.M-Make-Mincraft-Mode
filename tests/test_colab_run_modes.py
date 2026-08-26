from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai.colab_run_modes import (
    DEBUG_AUDIT_RELATIVE_PATH,
    DEBUG_MODE,
    EXISTING_MOD_MODE,
    EXISTING_PLAN_MODE,
    FULL_MODE,
    PLAN_MODE,
    RUN_MODES,
    debug_audit_path,
    resolve_plan_path,
    run_plan_dialog,
    should_build,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb",)
LEGACY_NOTEBOOK = ROOT / "Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb"


class _Proposal:
    def __init__(self, value: str) -> None:
        self.value = value

    def to_dict(self):
        return {"value": self.value, "items": list(range(40))}


class _Reply:
    def __init__(self, value: str) -> None:
        self.message = f"plan:{value}"
        self.complete_proposal = _Proposal(value)


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.saved: list[Path] = []

    def plan(self, prompt: str):
        self.calls.append(("plan", prompt))
        return _Reply(prompt)

    def revise(self, message: str):
        self.calls.append(("revise", message))
        return _Reply(message)

    def load_plan(self, path: Path):
        self.calls.append(("load", str(path)))
        return _Reply("loaded")

    def save_plan(self, path: Path):
        target = Path(path)
        self.saved.append(target)
        return target


def _cell_source(path: Path, cell_id: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell.get("source") or [])
    raise AssertionError(f"missing cell {cell_id!r} in {path.name}")


def _must_not_prompt(_: str) -> str:
    raise AssertionError("automatic Colab flow must not request manual approval input")


def test_run_modes_are_exact_and_full_mode_builds_by_default() -> None:
    assert PLAN_MODE == "Plan"
    assert FULL_MODE == "Full"
    assert EXISTING_MOD_MODE == "Revise"
    assert EXISTING_PLAN_MODE == "Execute"
    assert DEBUG_MODE == "Debug"
    assert RUN_MODES == (
        PLAN_MODE,
        FULL_MODE,
        EXISTING_MOD_MODE,
        EXISTING_PLAN_MODE,
        DEBUG_MODE,
    )
    assert should_build(PLAN_MODE) is False
    assert should_build(FULL_MODE) is True
    assert should_build(EXISTING_MOD_MODE) is True
    assert should_build(EXISTING_PLAN_MODE) is True
    assert should_build(DEBUG_MODE) is False


def test_canonical_notebook_dropdown_defaults_to_full_mode_and_has_debug_mode() -> None:
    expected = 'RUN_MODE = "Full" #@param ["Full", "Plan", "Revise", "Execute", "Debug"]'
    legacy_labels = (
        "플랜모드",
        "풀모드",
        "이미 만들어진 모드 수정보안모드",
        "이미 있는 플랜을 만드는모드",
    )
    assert not LEGACY_NOTEBOOK.exists()
    for notebook in NOTEBOOKS:
        assert notebook.is_file()
        source = _cell_source(notebook, "configuration")
        assert expected in source
        assert "PATCH_EXISTING" not in source
        assert all(label not in source for label in legacy_labels)
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        assert not any(cell.get("id") == "revise" for cell in payload["cells"])


def test_debug_audit_entrypoint_is_canonical() -> None:
    assert DEBUG_AUDIT_RELATIVE_PATH == "tools/full_project_audit.py"
    assert debug_audit_path("/repo") == Path("/repo/tools/full_project_audit.py")


def test_new_plan_auto_saves_without_user_confirmation(tmp_path: Path) -> None:
    session = _Session()
    output: list[str] = []

    result = run_plan_dialog(
        session=session,
        run_mode=FULL_MODE,
        prompt="새 모드",
        plan_path=tmp_path / "proposal.json",
        input_fn=_must_not_prompt,
        print_fn=lambda *values, **_: output.append(" ".join(map(str, values))),
    )

    assert result.approved is True
    assert session.calls == [("plan", "새 모드")]
    assert session.saved == [tmp_path / "proposal.json"]
    rendered = "\n".join(output)
    assert rendered.count("현재 플랜") == 1
    assert "사용자 승인 대기 없이 제작 단계로 진행" in rendered
    assert '"items"' in rendered
    assert "39" in rendered


def test_plan_mode_auto_saves_but_does_not_change_build_policy(tmp_path: Path) -> None:
    session = _Session()
    result = run_plan_dialog(
        session=session,
        run_mode=PLAN_MODE,
        prompt="플랜만",
        plan_path=tmp_path / "proposal.json",
        input_fn=_must_not_prompt,
        print_fn=lambda *_, **__: None,
    )
    assert result.approved is True
    assert session.calls == [("plan", "플랜만")]
    assert session.saved == [tmp_path / "proposal.json"]
    assert should_build(PLAN_MODE) is False


def test_existing_plan_mode_loads_and_builds_without_manual_approval(tmp_path: Path) -> None:
    session = _Session()
    plan_path = tmp_path / "proposal.json"
    plan_path.write_text("{}", encoding="utf-8")

    result = run_plan_dialog(
        session=session,
        run_mode=EXISTING_PLAN_MODE,
        prompt="",
        plan_path=plan_path,
        input_fn=_must_not_prompt,
        print_fn=lambda *_, **__: None,
    )

    assert result.approved is True
    assert session.calls == [("load", str(plan_path))]
    assert session.saved == []


def test_existing_plan_configured_path_must_exist(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        resolve_plan_path(
            run_mode=EXISTING_PLAN_MODE,
            output_root=tmp_path,
            configured_path=str(missing),
        )
