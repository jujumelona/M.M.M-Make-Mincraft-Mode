from __future__ import annotations

from copy import deepcopy

import minecraft_mod_ai.planning_state_pipeline as pipeline


PROMPT = "Build the requested behavior with the default scope policy."


def test_prompt_convergence_runs_with_empty_research_queue_before_requirement_selection(monkeypatch):
    initial_state = {
        "plan_ready": False,
        "decisions": [],
        "research_queue": [],
        "unresolved": [
            {
                "unresolved_id": "u_001",
                "reason": "scope",
                "resolution_route": "default_policy",
                "status": "open",
                "blocks": ["requirement_selection"],
            }
        ],
        "blockers": [],
        "evidence": [],
    }
    events: list[tuple[str, int, int] | str] = []

    def direct_transition(_operation, callback, *, input_state=None):
        return callback()

    def collect(_router, _prompt, current, *, trace_metadata=None):
        requirement_count = sum(
            1
            for row in current["decisions"]
            if row.get("decision_type") == "requirement"
        )
        events.append(("collect", requirement_count, len(current["research_queue"])))
        value = deepcopy(current)
        for row in value["unresolved"]:
            if row.get("resolution_route") == "default_policy":
                row["status"] = "resolved"
        return value

    def compile_requirements(_router, _prompt, current):
        events.append("compile_requirements")
        assert current["research_queue"] == []
        assert current["unresolved"][0]["status"] == "resolved"
        value = deepcopy(current)
        value["decisions"].append(
            {
                "decision_id": "d_001",
                "decision_type": "requirement",
                "requirement_id": "req_001",
            }
        )
        return value

    def finish_plan(_router, _prompt, current, **_kwargs):
        value = deepcopy(current)
        value["plan_ready"] = True
        return value

    monkeypatch.setattr(pipeline, "_transition", direct_transition)
    monkeypatch.setattr(pipeline, "emit_root_cause", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "repository_revision", lambda: {})
    monkeypatch.setattr(
        pipeline,
        "build_initial_planning_state",
        lambda _router, _prompt: deepcopy(initial_state),
    )
    monkeypatch.setattr(
        pipeline,
        "collect_planning_state_research_convergent",
        collect,
    )
    monkeypatch.setattr(
        pipeline,
        "compile_researched_requirements_convergent",
        compile_requirements,
    )
    monkeypatch.setattr(
        pipeline,
        "ensure_host_detail_section_applicability",
        lambda current: current,
    )
    monkeypatch.setattr(
        pipeline,
        "required_sections_by_requirement",
        lambda _current: {},
    )
    monkeypatch.setattr(
        pipeline,
        "compile_progress_monotone_detailed_plans",
        finish_plan,
    )
    monkeypatch.setattr(
        pipeline,
        "validate_planning_state",
        lambda _state, *, prompt: None,
    )
    monkeypatch.setattr(
        pipeline,
        "emit_planning_goal_satisfied",
        lambda _state: None,
    )

    result = pipeline.prepare_planning_state(None, PROMPT)

    assert events[:2] == [("collect", 0, 0), "compile_requirements"]
    assert events[-1] == ("collect", 1, 0)
    assert result["plan_ready"] is True
