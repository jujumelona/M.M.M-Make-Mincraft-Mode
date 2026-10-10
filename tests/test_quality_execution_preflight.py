"""Preflight must reveal unsatisfiable release evidence before Qwen/FLUX."""
from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.complete_orchestrator import (
    CompleteExecutionOptions,
    _quality_execution_preflight_gaps,
)


def _proposal(dimensions, design=None):
    return SimpleNamespace(
        game_design={
            **(design or {}),
            "_production_contract": {
                "quality_dimension_catalog": [
                    {"dimension_id": key} for key in dimensions
                ],
            },
        }
    )


def test_debug_build_reports_all_unavailable_release_evidence() -> None:
    proposal = _proposal(("correctness", "build", "research", "runtime", "visual_3d", "accessibility"))
    options = CompleteExecutionOptions(
        run_runtime=False, run_mineflayer=False, run_visual_review=False,
        server_launcher=None, screenshot_paths=(), playtest_actions=(),
    )
    gaps = _quality_execution_preflight_gaps(proposal, options)
    assert set(gaps) == {"research", "runtime", "visual_3d"}
    assert "run_runtime" in gaps["runtime"]
    assert "run_mineflayer" in gaps["runtime"]
    assert "run_visual_review" in gaps["visual_3d"]
    assert "technology radar" in gaps["research"]
    assert "accessibility" not in gaps  # Pending validators are not preflight-impossible.


def test_quality_preflight_cannot_misreport_clean_build_or_gametest() -> None:
    proposal = _proposal(("correctness", "build"))
    assert _quality_execution_preflight_gaps(
        proposal,
        CompleteExecutionOptions(run_runtime=False, run_visual_review=False),
    ) == {}


def test_quality_preflight_not_invoked_for_source_only_or_legacy() -> None:
    proposal = _proposal(("research", "runtime"))
    assert _quality_execution_preflight_gaps(
        proposal, CompleteExecutionOptions(source_only=True)
    ) == {}
    assert _quality_execution_preflight_gaps(
        SimpleNamespace(game_design={"mode": "debug_fixture"}),
        CompleteExecutionOptions(),
    ) == {}
