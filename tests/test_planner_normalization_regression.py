from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import minecraft_mod_ai.planning_state_pipeline as pipeline


PROMPT = "Build the requested behavior."


def _direct_transition(_operation, callback, *, input_state=None):
    return callback()


def test_requirement_resolution_skips_research_when_no_gap_exists(monkeypatch):
    state = {
        "plan_ready": False,
        "decisions": [],
        "research_queue": [],
        "unresolved": [],
        "blockers": [],
        "evidence": [],
    }

    monkeypatch.setattr(pipeline, "_transition", _direct_transition)

    def forbidden_research(*_args, **_kwargs):
        raise AssertionError("research must not run without an explicit knowledge gap")

    monkeypatch.setattr(
        pipeline,
        "collect_planning_state_research_convergent",
        forbidden_research,
    )

    def compile_requirements(_router, _prompt, current):
        value = deepcopy(current)
        value["decisions"].append(
            {
                "decision_id": "d_001",
                "decision_type": "requirement",
                "requirement_id": "req_001",
            }
        )
        return value

    monkeypatch.setattr(
        pipeline,
        "compile_researched_requirements_convergent",
        compile_requirements,
    )

    result, waiting = pipeline._resolve_requirements_or_wait(
        None,
        PROMPT,
        deepcopy(state),
        trace_metadata=None,
        checkpoint=None,
    )

    assert waiting is False
    assert pipeline._requirements_exist(result)


def test_research_stage_predicate_is_gap_driven():
    assert pipeline._research_stage_needed(
        {"research_queue": [], "unresolved": []}
    ) is False
    assert pipeline._research_stage_needed(
        {
            "research_queue": [{"status": "pending"}],
            "unresolved": [],
        }
    ) is True
    assert pipeline._research_stage_needed(
        {
            "research_queue": [],
            "unresolved": [
                {
                    "status": "open",
                    "resolution_route": "default_policy",
                }
            ],
        }
    ) is True
    assert pipeline._research_stage_needed(
        {
            "research_queue": [{"status": "blocked"}],
            "unresolved": [
                {
                    "status": "resolved",
                    "resolution_route": "external_research",
                }
            ],
        }
    ) is False


def test_one_shot_normalization_scaffolding_is_not_committed():
    repository_root = Path(__file__).resolve().parents[1]
    assert not (
        repository_root / "tools" / "apply_cumulative_architecture_normalization.py"
    ).exists()
    assert not (
        repository_root
        / ".github"
        / "workflows"
        / "cumulative-architecture-normalization.yml"
    ).exists()
    assert not (
        repository_root
        / ".github"
        / "workflows"
        / "runner-efficiency-normalization.yml"
    ).exists()
