from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import agentic_optimization_contract as agentic
from minecraft_mod_ai.agentic_search_efficiency_contract import install


def _risky_request() -> dict[str, object]:
    return {
        "current_target_deliverables": ["a", "b", "c", "d"],
        "scope": "custom_java networking integration persistence",
    }


def test_auto_planner_search_preserves_risk_width_when_slots_exist(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    assert agentic._planner_candidate_count(_risky_request(), "production page") == 3


def test_auto_planner_search_does_not_duplicate_serial_decode(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert agentic._planner_candidate_count(_risky_request(), "production page") == 1


def test_auto_planner_search_caps_breadth_to_native_slots(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    assert agentic._planner_candidate_count(_risky_request(), "production page") == 2


def test_explicit_agentic_search_on_keeps_requested_width(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert agentic._planner_candidate_count({}, "planner") == 3


def test_auto_repair_search_escalates_only_after_same_failure_repeats(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "2")

    engine = SimpleNamespace()
    engine._signature = lambda evidence: "same-signature"
    evidence = {
        "diagnostics": {
            "diagnostics": [
                {"path": "A.java", "message": "error one"},
                {"path": "B.java", "message": "error two"},
            ]
        },
        "build": {"status": "FAIL", "error": "x" * 200},
    }

    first = agentic._repair_candidate_count(engine, evidence, ())
    second = agentic._repair_candidate_count(engine, evidence, ())
    assert first == 1
    assert second == 2
