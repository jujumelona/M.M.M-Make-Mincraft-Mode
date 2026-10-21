"""Resolve state planning gaps before the canonical Java state owner is built."""
import json

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.construction_behavior_state import reconcile_behavior_state
from minecraft_mod_ai.construction_planner import author_behavior_requirements
from minecraft_mod_ai.production_state_compiler import (
    compile_production_state_section,
    render_production_state_java,
)


def variable(name="credits", kind="long", default="0"):
    return {"name": name, "owner": "server", "type": kind,
            "unit": "units", "default": default, "domain": "nonnegative"}


def plan(reads=(), writes=(), variables=None):
    behavior = {"behavior_id": "b1", "requirement_id": "r1", "intent": "Change charge",
                "source_quote": "Change charge", "trigger": "player interaction",
                "effects": "Change charge", "state_reads": list(reads), "state_writes": list(writes)}
    return AuthoredPlan(
        requested_prompt="Change charge", text="Original saved design",
        structured_sections={"state_model": {"specification": {"variables": variables or [variable()]}}},
        request_scope={"requirements": [{"requirement_id": "r1", "statement": "Change charge"}],
                       "behavior_requirements": [behavior]},
    )


class BindingRouter:
    def __init__(self, decisions=()):
        self.decisions = iter(decisions)
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append((data_name, parameters, messages))
        value = next(self.decisions)
        assert set(value) == set(parameters["properties"])
        return value


def new_state_decisions():
    return [{"binding": "new_scalar_state"}, {"owner": "server"}, {"type": "int"},
            {"unit": "charge"}, {"default": "0"}, {"domain": "nonnegative charge"}]


def test_missing_state_is_declared_before_production_and_original_is_preserved():
    original = plan(reads=["charge"], writes=["charge"])
    before = original.to_dict()
    router = BindingRouter(new_state_decisions())
    result = reconcile_behavior_state(router, original)
    state = compile_production_state_section(result)
    assert state["specification"]["variables"] == [variable(), variable("charge", "int") | {
        "unit": "charge", "domain": "nonnegative charge"}]
    java = render_production_state_java(state, package_name="example")
    assert "int readState_charge(" in java
    assert "writeState_charge(int value" in java
    assert original.to_dict() == before
    assert result.text == original.text
    assert len(router.calls) == 6


def test_world_observation_does_not_become_fake_persistent_scalar():
    original = plan(reads=["build_area_contents"], writes=["build_area_contents"])
    router = BindingRouter([{"binding": "world_state"}, {"observation": "actual blocks in the event world"}])
    result = reconcile_behavior_state(router, original)
    behavior = result.request_scope["behavior_requirements"][0]
    assert behavior["state_reads"] == behavior["state_writes"] == []
    assert behavior["runtime_state_reads"] == behavior["runtime_state_writes"] == ["build_area_contents"]
    assert result.structured_sections == original.structured_sections
    restored = author_behavior_requirements(result.request_scope, None)
    assert restored[0].runtime_state_writes == ("build_area_contents",)


def test_identifier_equivalent_alias_is_bound_by_host_without_model():
    router = BindingRouter()
    result = reconcile_behavior_state(router, plan(reads=["Credits"], writes=["Credits"]))
    assert result.request_scope["behavior_requirements"][0]["state_reads"] == ["credits"]
    assert router.calls == []
    binding = result.request_scope["behavior_state_bindings"]["Credits"]
    assert binding["state_name"] == "credits"
    assert binding["semantic_status"] == "host_verified_identifier_match"


def test_unrelated_reference_cannot_select_existing_canonical_state():
    router = BindingRouter([{"binding": "canonical_state"}])
    with pytest.raises(Exception):
        reconcile_behavior_state(router, plan(reads=["wallet"]))
    assert router.calls[0][1]["properties"]["binding"]["enum"] == [
        "new_scalar_state", "world_state",
    ]


def test_replayed_unproven_alias_is_rejected():
    from minecraft_mod_ai.construction_behavior_state import _validate_replayed_binding
    with pytest.raises(ValueError, match="CANONICAL_MEANING_UNPROVEN"):
        _validate_replayed_binding("Wallet", {
            "original_name": "Wallet", "binding": "canonical_state", "state_name": "credits",
        }, [variable()])


def test_serialized_plan_cannot_bypass_alias_validation_on_fast_return():
    original = plan(reads=["credits"], writes=["credits"])
    original.request_scope["behavior_state_bindings"] = {
        "Wallet": {
            "original_name": "Wallet", "binding": "canonical_state",
            "state_name": "credits",
        },
    }
    with pytest.raises(ValueError, match="CANONICAL_MEANING_UNPROVEN"):
        reconcile_behavior_state(BindingRouter(), original)


def test_declared_state_keeps_exact_type_default_without_inference():
    original = plan(reads=["credits"], writes=["credits"], variables=[variable(default="12")])
    assert reconcile_behavior_state(BindingRouter(), original).to_dict() == original.to_dict()


def test_reconciled_round_trip_does_not_infer_or_declare_state_twice():
    result = reconcile_behavior_state(BindingRouter(new_state_decisions()), plan(writes=["charge"]))
    restored = AuthoredPlan.from_dict(json.loads(json.dumps(result.to_dict())))
    assert reconcile_behavior_state(BindingRouter(), restored).to_dict() == restored.to_dict()


def test_binding_checkpoint_restores_completed_choices(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_CONSTRUCTION_BEHAVIOR_CACHE_ROOT", str(tmp_path))
    original = plan(writes=["charge"])
    first = reconcile_behavior_state(BindingRouter(new_state_decisions()), original)
    second = reconcile_behavior_state(BindingRouter(), original)
    assert first.to_dict() == second.to_dict()


def test_new_state_cannot_encode_an_aggregate_in_a_scalar_default():
    decisions = new_state_decisions()
    decisions[2] = {"type": "string"}
    decisions[4] = {"default": '{"hidden":"collection"}'}
    with pytest.raises(Exception, match="default|pattern"):
        reconcile_behavior_state(BindingRouter(decisions), plan(writes=["charge"]))
