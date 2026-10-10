"""Regression tests for executable gameplay entrypoint recovery."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.gameplay_entrypoint_repair import (
    repair_missing_gameplay_entrypoint,
)
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS


def _section(name, updates=None):
    spec = {concern: [] for concern in DETAIL_RECORDS[name]}
    spec["inapplicable_concerns"] = []
    spec.update(updates or {})
    return {"specification": spec, "constraint_evidence_refs": []}


def _sections(trigger="player_join"):
    return {
        "integration": _section("integration", {
            "entry_points": [
                {"boundary": "initialization", "trigger": trigger, "owner": "host"}
            ],
        }),
        "algorithm": _section("algorithm", {
            "steps": [{"operation": "upgrade", "input": "credits",
                       "output": "upgrade", "next_step": "finish"}],
            "atomic_mutations": [{"mutations": "spend credits; grant part", "commit": "purchase succeeds", "rollback": "restore previous state"}],
        }),
    }


def test_repair_adds_explicit_gameplay_command_without_losing_join(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    captured = []

    def fake_model(_router, _role, messages, *, response_schema, **_kwargs):
        captured.append((messages, response_schema))
        assert response_schema["properties"]["trigger"]["pattern"].startswith("^command:")
        return {
            "boundary": "shipyard upgrade invocation",
            "trigger": "command:shipyard_upgrade",
            "owner": "authoritative server",
        }

    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value", fake_model,
    )
    original = _sections()
    repaired = repair_missing_gameplay_entrypoint(object(), "buy ship upgrades", original)
    entry_points = repaired["integration"]["specification"]["entry_points"]
    assert [row["trigger"] for row in entry_points] == [
        "player_join", "command:shipyard_upgrade",
    ]
    assert repaired["algorithm"] == original["algorithm"]
    assert original["integration"]["specification"]["entry_points"] == [
        {"boundary": "initialization", "trigger": "player_join", "owner": "host"}
    ]
    assert len(captured) == 1


def test_existing_gameplay_command_needs_no_repair(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    def no_model(*args, **kwargs):
        raise AssertionError("unnecessary second model call")

    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value", no_model,
    )
    input_sections = _sections("command:launch_ship")
    assert repair_missing_gameplay_entrypoint(
        object(), "launch ship", input_sections,
    ) == input_sections


def test_no_atomic_mutations_needs_no_repair(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    def no_model(*args, **kwargs):
        raise AssertionError("no atomic mutations")

    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value", no_model,
    )
    sections = _sections()
    sections["algorithm"]["specification"]["atomic_mutations"] = []
    assert repair_missing_gameplay_entrypoint(object(), "view items", sections) == sections


def test_repair_does_not_drop_existing_hooks_when_full():
    sections = _sections()
    sections["integration"]["specification"]["entry_points"] = [
        {"boundary": f"startup_{i}", "trigger": "player_join", "owner": "host"}
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="GAMEPLAY_ENTRYPOINT_CAPACITY_EXHAUSTED"):
        repair_missing_gameplay_entrypoint(object(), "buy upgrades", sections)


def test_atomic_mutation_must_not_execute_on_player_join():
    from minecraft_mod_ai.typed_plan_authoring import author_semantic_game_dispatch

    with pytest.raises(ValueError, match="TYPED_GAMEPLAY_MUTATION_EVENT_UNBOUND"):
        author_semantic_game_dispatch(
            object(),
            "credits purchase",
            {},
            {},
            coverage_refs=("algorithm.atomic_mutations",),
            bound_events=("player_join",),
        )


def test_algorithm_steps_are_not_collapsed_into_one_model_action(monkeypatch):
    import json
    from minecraft_mod_ai import typed_plan_authoring, fixed_template_generation

    sections = _sections("command:launch_ship")
    steps = [
        {"operation": f"step_{i}", "input": "state",
         "output": "state", "next_step": "next"}
        for i in range(4)
    ]
    sections["algorithm"]["specification"]["steps"] = steps
    from minecraft_mod_ai.typed_plan_authoring import author_semantic_game_dispatch

    monkeypatch.setattr(
        typed_plan_authoring, "_extract_state_variable_types",
        lambda _sections: {"counter": "int"},
    )
    observed = []

    def model(_router, _role, messages, **_kwargs):
        payload = json.loads(messages[1]["content"])
        observed.append(payload["focus_record"]["operation"])
        return {
            "trigger_event": "command", "action_kind": "increment_state",
            "state_key": "counter", "value": 1,
        }

    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value", model,
    )
    body = author_semantic_game_dispatch(
        object(), "four sequential gameplay steps", sections, {},
        coverage_refs=("algorithm.steps",),
        bound_events=("command",),
    )
    assert observed == ["step_0", "step_1", "step_2", "step_3"]
    assert sum(row["op"] == "if" for row in body) == 4
