from __future__ import annotations

import json
from pathlib import Path

from tools import model_realistic_replay_matrix

ROOT = Path(__file__).resolve().parents[1]
REPLAY_DIR = ROOT / "tests" / "model_realistic_replay"
REPLAY_WORKFLOW = ROOT / ".github" / "workflows" / "model-realistic-replay.yml"
MAIN_WORKFLOW = ROOT / ".github" / "workflows" / "main-ci.yml"


def test_replay_discovery_covers_every_behavior_file_exactly_once() -> None:
    expected = sorted(
        path.relative_to(ROOT).as_posix()
        for path in REPLAY_DIR.glob("test_*.py")
    )
    discovered = model_realistic_replay_matrix.discover()
    assert discovered == expected
    assert discovered
    assert len(discovered) == len(set(discovered))


def test_replay_matrix_has_one_behavior_file_per_cell(capsys) -> None:
    assert model_realistic_replay_matrix.main() == 0
    payload = json.loads(capsys.readouterr().out)
    cells = payload["include"]
    assert len(cells) == len(model_realistic_replay_matrix.discover())
    assert all(set(cell) == {"id", "test"} for cell in cells)
    assert [cell["test"] for cell in cells] == model_realistic_replay_matrix.discover()


def test_replay_workflow_is_dynamic_non_fail_fast_and_reusable() -> None:
    workflow = REPLAY_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_call:" in workflow
    assert "python -m tools.model_realistic_replay_matrix" in workflow
    assert "fail-fast: false" in workflow
    assert '${{ matrix.test }}' in workflow
    assert "Model-realistic replay gate" in workflow


def test_final_ci_gate_requires_replay_workflow() -> None:
    workflow = MAIN_WORKFLOW.read_text(encoding="utf-8")
    assert "model-realistic-replay:" in workflow
    assert "uses: ./.github/workflows/model-realistic-replay.yml" in workflow
    assert "needs: [audit, tests, python313, model-realistic-replay]" in workflow
    assert 'MODEL_REPLAY: ${{ needs.model-realistic-replay.result }}' in workflow
    assert 'test "$MODEL_REPLAY" = success' in workflow
