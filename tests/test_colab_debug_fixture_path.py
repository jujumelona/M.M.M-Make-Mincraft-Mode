from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from minecraft_mod_ai import CompleteExecutionOptions, CompleteModAISession
from minecraft_mod_ai.colab_run_modes import FULL_MODE, run_plan_dialog, should_build


class _CapturingOrchestrator:
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
    orchestrator = _CapturingOrchestrator()
    session.orchestrator = orchestrator

    def fail_if_planner_runs(*_args, **_kwargs):
        raise AssertionError("Colab Debug Mode must not call the planner")

    session.plan = fail_if_planner_runs
    return session, orchestrator


def test_colab_debug_fixture_loads_validated_plan_then_uses_normal_build_dispatch(
    tmp_path: Path,
) -> None:
    session, orchestrator = _debug_session(tmp_path)
    plan_path = tmp_path / "proposal.json"
    printed: list[str] = []

    dialog = run_plan_dialog(
        session=session,
        run_mode=FULL_MODE,
        prompt="ignored because Debug Mode owns the deterministic fixture",
        plan_path=plan_path,
        debug_mode=True,
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
        "M.M.M Debug Mode fixture: add one deterministic debug token item and "
        "run the normal implementation/verification pipeline."
    )
    assert [module.module_id for module in proposal.modules] == ["debug_token"]
    task = proposal.modules[0].config["evidence_task"]
    assert task["task_id"] == "debug_token"
    assert task["execution_role"] == "production"
    assert task["required_gates"] == ["target_compile"]
    assert any("planner 호출 없이" in line for line in printed)

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
