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

    def fake_model(_router, _role, messages, *, response_schema, **kwargs):
        from minecraft_mod_ai.model_output_atomicity_contract import (
            structured_output_token_ceiling,
        )
        # Real production budget proof must execute here. The previous tests
        # never covered the unbounded inherited boundary/owner fields.
        assert all(
            response_schema["properties"][field].get("maxLength")
            for field in ("boundary", "owner", "trigger")
        )
        assert kwargs["output_token_ceiling"] == structured_output_token_ceiling(
            response_schema,
        )
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


def test_full_page_with_duplicate_join_hooks_preserves_all_obligations(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    sections = _sections()
    original = [
        {"boundary": f"startup_{i}", "trigger": "player_join", "owner": f"owner_{i}"}
        for i in range(4)
    ]
    sections["integration"]["specification"]["entry_points"] = original
    calls = []

    def model(*_args, **_kwargs):
        calls.append(1)
        return {
            "boundary": "purchase invocation",
            "trigger": "command:purchase",
            "owner": "server",
        }

    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value", model,
    )
    repaired = repair_missing_gameplay_entrypoint(
        object(), "buy upgrades", sections,
    )
    rows = repaired["integration"]["specification"]["entry_points"]
    assert len(rows) == 5
    assert rows[:4] == original
    assert rows[4]["trigger"] == "command:purchase"
    assert sections["integration"]["specification"]["entry_points"] == original
    assert calls == [1]


def test_full_page_with_distinct_hooks_preserves_all_in_fifth_slot(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    sections = _sections()
    original = [
        {"boundary": str(i), "trigger": trigger, "owner": "host"}
        for i, trigger in enumerate((
            "player_join", "server_started", "server_stopping", "server_tick",
        ))
    ]
    sections["integration"]["specification"]["entry_points"] = original
    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value",
        lambda *_args, **_kwargs: {
            "boundary": "buy an upgrade", "trigger": "command:buy_upgrade",
            "owner": "server",
        },
    )
    repaired = repair_missing_gameplay_entrypoint(
        object(), "buy upgrades", sections,
    )
    rows = repaired["integration"]["specification"]["entry_points"]
    assert rows[:4] == original
    assert len(rows) == 5
    assert rows[4]["trigger"] == "command:buy_upgrade"


def test_overfilled_legacy_page_coalesces_only_duplicates(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    sections = _sections()
    original = [
        {"boundary": f"join_{i}", "trigger": "player_join", "owner": f"init_{i}"}
        for i in range(5)
    ]
    sections["integration"]["specification"]["entry_points"] = original
    # This helper consumes a legacy record list before validating the result.
    # Five rows are canonical; one duplicate can be coalesced before appending.
    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value",
        lambda *_args, **_kwargs: {
            "boundary": "player purchase", "trigger": "command:buy",
            "owner": "server",
        },
    )
    output = repair_missing_gameplay_entrypoint(object(), "buy", sections)
    rows = output["integration"]["specification"]["entry_points"]
    assert len(rows) == 2
    assert all(f"join_{i}" in rows[0]["boundary"] for i in range(5))
    assert all(f"init_{i}" in rows[0]["owner"] for i in range(5))
    assert rows[1]["trigger"] == "command:buy"


def test_server_tick_does_not_substitute_for_a_player_action(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    sections = _sections("server_tick")
    monkeypatch.setattr(
        fixed_template_generation,
        "generate_fixed_template_value",
        lambda *_args, **_kwargs: {
            "boundary": "launch invocation",
            "trigger": "command:launch",
            "owner": "server",
        },
    )
    repaired = repair_missing_gameplay_entrypoint(object(), "launch", sections)
    assert [x["trigger"] for x in repaired["integration"]["specification"]["entry_points"]] == [
        "server_tick", "command:launch",
    ]


def test_atomic_mutation_must_not_execute_on_player_join(monkeypatch):
    from minecraft_mod_ai import typed_plan_authoring

    monkeypatch.setattr(
        typed_plan_authoring, "_extract_state_variable_types",
        lambda _sections: {"credits": "int"},
    )
    with pytest.raises(ValueError, match="TYPED_GAMEPLAY_MUTATION_EVENT_UNBOUND"):
        typed_plan_authoring.author_semantic_game_dispatch(
            object(),
            "credits purchase",
            _sections(),
            {},
            coverage_refs=("algorithm.atomic_mutations",),
            bound_events=("player_join",),
        )


def test_algorithm_steps_are_not_collapsed_into_one_model_action(monkeypatch):
    import json
    from minecraft_mod_ai import typed_plan_authoring

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
        typed_plan_authoring, "generate_fixed_template_value", model,
    )
    body = author_semantic_game_dispatch(
        object(), "four sequential gameplay steps", sections, {},
        coverage_refs=("algorithm.steps",),
        bound_events=("command",),
    )
    assert observed == ["step_0", "step_1", "step_2", "step_3"]
    assert sum(row["op"] == "if" for row in body) == 4


def test_repair_executes_actual_planner_decoder_contract_without_model(monkeypatch):
    """Exercise the real sampler projection and budget proof without an LLM."""
    import json

    class FakeRouter:
        def __init__(self):
            self.calls = []

        def generate_text(self, role, messages, **kwargs):
            assert role == "planner"
            assert kwargs["response_format"] == "json"
            assert kwargs["response_schema"] is not None
            assert kwargs["output_token_ceiling"] <= 4096
            self.calls.append(kwargs)
            return json.dumps({
                "boundary": "server-authoritative purchase",
                "trigger": "command:purchase_upgrade",
                "owner": "server",
            })

    router = FakeRouter()
    sections = _sections()
    rows = repair_missing_gameplay_entrypoint(
        router, "buy an upgrade with credits", sections,
    )["integration"]["specification"]["entry_points"]
    assert len(router.calls) == 1
    assert len(rows) == 2
    assert rows[1]["trigger"] == "command:purchase_upgrade"
