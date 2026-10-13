from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_mod_ai import CompleteExecutionOptions, CompleteModAISession
from minecraft_mod_ai.colab_run_modes import FULL_MODE, run_plan_dialog, should_build
from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator


class _CapturingOrchestrator(CompleteProductionOrchestrator):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        proposal,
        *,
        approval_hash: str,
        run_name: str,
        options: CompleteExecutionOptions,
        existing_input,
    ):
        self.calls.append(
            {
                "proposal": proposal,
                "approval_hash": approval_hash,
                "run_name": run_name,
                "options": options,
                "existing_input": existing_input,
            }
        )
        return SimpleNamespace(status="CAPTURED_DEBUG_DISPATCH")


def _debug_session(tmp_path: Path) -> tuple[CompleteModAISession, _CapturingOrchestrator]:
    session = CompleteModAISession.__new__(CompleteModAISession)
    session.output_root = tmp_path
    session.workspace_root = tmp_path.resolve()
    session.complete_proposal = None
    session.brief = ""
    session.existing_input = None
    # The host-only fixture bypasses __init__; provide an inert router so
    # production target/preflight inspection can exercise its real contract.
    # It must never construct or call a model during deterministic Debug Mode.
    session.router = SimpleNamespace()
    orchestrator = _CapturingOrchestrator()
    session.orchestrator = orchestrator

    def fail_if_planner_runs(*_args, **_kwargs):
        raise AssertionError("Colab Debug Mode must not call the planner")

    session.plan = fail_if_planner_runs
    return session, orchestrator


def test_colab_debug_fixture_loads_validated_plan_then_uses_normal_build_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, orchestrator = _debug_session(tmp_path)
    prepared = []
    toolchain_checks = []

    def prepare(owner, approved, *, run_root, adapter, error_type):
        # This dispatch fixture does not download an external scaffold. Keep
        # the actual proposal writer and assert preparation happens first.
        root = run_root / "project"
        owner._write_base_proposal(root, approved.base_proposal)
        assert not owner.calls
        prepared.append(root)
        return root

    monkeypatch.setattr("minecraft_mod_ai.platform_live_execution_contract.prepare_official_fabric_project",
                        prepare)
    monkeypatch.setattr("minecraft_mod_ai.complete_preflight_contract.validate_platform_toolchain_preflight",
                        lambda platform, options: toolchain_checks.append((platform, options)))
    plan_path = tmp_path / "proposal.json"
    printed: list[str] = []

    dialog = run_plan_dialog(
        session=session,
        run_mode=FULL_MODE,
        prompt="ignored because Debug Mode owns the deterministic fixture",
        plan_path=plan_path,
        debug_mode=True,
        debug_strategy="host_smoke",
        minecraft_version="26.2",
        loader="Auto",
        print_fn=lambda *parts, **_kwargs: printed.append(" ".join(map(str, parts))),
    )

    assert dialog.approved is True
    assert dialog.plan_path == plan_path
    assert plan_path.is_file()
    assert should_build(FULL_MODE) is True

    proposal = dialog.reply.complete_proposal
    assert proposal.requested_prompt == (
        "M.M.M Debug Mode fixture: add one deterministic host-generated debug token item and "
        "run the normal implementation/verification pipeline."
    )
    # Debug Mode is host-owned and therefore does not manufacture a production
    # module/evidence task just to drive the deterministic fixture through the build.
    assert proposal.modules == ()
    fixture = proposal.game_design["fixture"]
    assert fixture["artifact_id"] == "debug_token"
    assert fixture["kind"] == "item"
    assert fixture["deterministic"] is True
    assert fixture["source_contract"]["path"].endswith("/DebugToken.java")
    assert any("host_smoke" in line for line in printed)

    options = CompleteExecutionOptions(
        run_blockbench=False,
        run_runtime=False,
        run_client=False,
        run_mineflayer=False,
        run_visual_review=False,
        eula_accepted=False,
        server_launcher=None,
        screenshot_paths=(),
        resume=True,
    )
    result = session.build(
        dialog.reply,
        run_name="complete-colab-run",
        options=options,
    )

    assert result.status == "CAPTURED_DEBUG_DISPATCH"
    assert prepared == [orchestrator._prepared_fabric_project]
    assert (prepared[0] / ".minecraft_ai/base-proposal.json").is_file()
    assert len(toolchain_checks) == 1 and toolchain_checks[0][1] is options
    assert len(orchestrator.calls) == 1
    call = orchestrator.calls[0]
    assert call["proposal"] is proposal
    assert call["approval_hash"] == proposal.calculate_hash()
    assert call["run_name"] == "complete-colab-run"
    assert call["options"] is options
    assert call["existing_input"] is None


def test_colab_notebook_wires_debug_dialog_reply_into_normal_session_build() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook.get("cells", ())
        if cell.get("cell_type") == "code"
    )

    assert "dialog = run_plan_dialog(" in code
    assert "debug_mode=DEBUG_MODE" in code
    assert "reply = dialog.reply" in code
    assert "if DEBUG_MODE and RUN_MODE != \"Full\":" in code
    assert "BUILD_RESULT = session.build(reply, run_name=RUN_NAME, options=options)" in code


def test_colab_debug_defaults_to_live_model_path() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = {
        cell["id"]: "".join(cell.get("source", ()))
        for cell in notebook.get("cells", ())
    }
    assert 'DEBUG_STRATEGY = "Model path (slow)"' in cells["configuration"]
    assert '"Model path (slow)": "model_path"' in cells["plan"]
    assert 'prompt=PROMPT' in cells["plan"]
    assert "reply = dialog.reply" in cells["plan"]
    assert "BUILD_RESULT = session.build(reply, run_name=RUN_NAME, options=options)" in cells["build"]


def test_colab_build_checks_setup_and_rembg_version_before_generation() -> None:
    """A rerun of only cell 6 cannot bypass the installed runtime contract."""
    notebook_path = Path(__file__).resolve().parents[1] / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(next(
        cell["source"] for cell in notebook["cells"] if cell.get("id") == "build"
    ))
    compile(source, "<colab:build>", "exec")
    dispatch = source.index("session.build(")
    assert source.index("module.assert_setup_state(") < dispatch
    assert source.index('_distribution_version("rembg")') < dispatch
    assert source.index('_rembg_version != "2.0.67"') < dispatch
    assert source.index("BUILD_DEPENDENCY_PREFLIGHT:") < dispatch
    # rembg transitively imports onnxruntime; leave it in the isolated worker.
    assert "from rembg import" not in source
