from __future__ import annotations

import pytest

from minecraft_mod_ai.stage_template_pipeline import (
    KNOWN_STAGES,
    run_stage_pipeline,
    validate_stage_workflows,
)


def test_stage_workflows_are_all_valid():
    validate_stage_workflows()
    assert KNOWN_STAGES == ("code", "asset", "integration", "validation")


@pytest.mark.parametrize("stage,expected_steps", [
    ("code", 9),
    ("asset", 6),
    ("integration", 6),
    ("validation", 10),
])
def test_stage_pipeline_execution_and_checkpoint(stage, expected_steps):
    context = {"feature_id": "test_feature", "platform": "fabric"}
    saved_progress = {}

    def checkpoint(binding, receipt):
        saved_progress[binding] = receipt

    result = run_stage_pipeline(stage, context, checkpoint=checkpoint)
    assert result["stage"] == stage
    assert len(result["receipts"]) == expected_steps
    assert len(saved_progress) == expected_steps
    assert all(r["status"] == "PASS" for r in result["receipts"])
    assert all(r["proof"]["passed"] is True for r in result["receipts"])

    # Replay with saved progress
    replayed = run_stage_pipeline(stage, context, progress=saved_progress)
    assert replayed["receipts"] == result["receipts"]
