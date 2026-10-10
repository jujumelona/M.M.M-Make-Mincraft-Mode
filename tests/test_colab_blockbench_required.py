"""The public Colab session must schedule required Blockbench work, not drop it."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import api
from minecraft_mod_ai.colab_run_modes import write_debug_example_plan
from minecraft_mod_ai.complete_orchestrator import CompleteExecutionOptions
from minecraft_mod_ai.complete_spec import CompleteProposal, ProductionModule


@pytest.mark.parametrize(
    ("kind", "source_only", "must_enable"),
    [
        ("entity", False, True),
        ("boss", False, True),
        ("npc", False, True),
        ("item", False, False),
        ("entity", True, False),
    ],
)
def test_session_autoselects_mandatory_blockbench_review_for_entity_plan(
    monkeypatch, tmp_path: Path, kind: str, source_only: bool, must_enable: bool
) -> None:
    path = write_debug_example_plan(
        tmp_path / "plan.json", minecraft_version="1.21.8", loader="fabric"
    )
    proposal = CompleteProposal.from_dict(json.loads(path.read_text(encoding="utf-8")))
    proposal = replace(
        proposal,
        modules=(ProductionModule(module_id="test_module", kind=kind),),
    )
    session = api.CompleteModAISession.__new__(api.CompleteModAISession)
    session.output_root = tmp_path
    session.workspace_root = tmp_path.resolve()
    session.existing_input = None
    calls: list[CompleteExecutionOptions] = []
    session.orchestrator = SimpleNamespace(
        execute=lambda _proposal, *, approval_hash, run_name, options, existing_input:
            calls.append(options) or SimpleNamespace(status="CAPTURED")
    )
    monkeypatch.setattr(api, "_validate_internal_engine_preflight", lambda: None)
    monkeypatch.setattr(api, "_bound_platform_for_preflight", lambda _s, _p: None)
    options = CompleteExecutionOptions(
        source_only=source_only,
        run_blockbench=False,
        run_runtime=False,
        run_client=False,
        run_mineflayer=False,
        run_visual_review=False,
    )
    result = session.build(proposal, options=options)
    assert result.status == "CAPTURED"
    assert len(calls) == 1
    assert calls[0].run_blockbench is must_enable
    assert options.run_blockbench is False  # immutable caller input


def test_colab_notebook_schedules_blockbench_by_default() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "M.M.M_Make_Mincraft_Mode_Colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    assert "RUN_BLOCKBENCH = True" in source
    assert "run_blockbench=RUN_BLOCKBENCH" in source
