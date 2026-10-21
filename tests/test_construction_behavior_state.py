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
        reconcile_behavior_state(router, plan(reads=["wallet"], writes=["wallet"]))
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



def test_reject_control_plane_label_as_new_scalar_owner():
    router = BindingRouter([
        {"binding": "new_scalar_state"},
        {"owner": "new_scalar_state"},
    ])
    with pytest.raises(Exception):
        reconcile_behavior_state(router, plan(writes=["charge"]))
    assert len(router.calls) == 2
    choices = router.calls[1][1]["properties"]["owner"]["enum"]
    assert "new_scalar_state" not in choices
    assert "world_state" not in choices
    assert {"server", "player", "world", "entity"}.issubset(choices)


def test_scalar_owner_uses_constrained_schema_and_keeps_reference():
    router = BindingRouter([
        {"binding": "new_scalar_state"},
        {"owner": "player"},
        {"type": "int"}, {"unit": "charge"}, {"default": "0"},
        {"domain": "charge"},
    ])
    result = reconcile_behavior_state(router, plan(writes=["charge"]))
    variable_row = result.structured_sections["state_model"]["specification"]["variables"][-1]
    assert variable_row["owner"] == "player"
    assert variable_row["name"] == "charge"
    assert len(router.calls) == 6
    assert result.request_scope["behavior_requirements"][0]["state_writes"] == ["charge"]


def test_missing_read_only_state_cannot_be_invented_as_stored_value():
    router = BindingRouter()
    result = reconcile_behavior_state(router, plan(reads=["items in nearby containers"]))
    assert router.calls == []
    assert result.structured_sections == plan(reads=["items in nearby containers"]).structured_sections
    behavior = result.request_scope["behavior_requirements"][0]
    assert behavior["state_reads"] == []
    assert behavior["runtime_state_reads"] == ["items in nearby containers"]
    record = result.request_scope["behavior_state_bindings"]["items in nearby containers"]
    assert record["binding"] == "world_state"
    assert record["semantic_status"] == "host_classified_unbound_read"


def test_write_only_state_still_declares_real_scalar():
    router = BindingRouter(new_state_decisions())
    result = reconcile_behavior_state(router, plan(writes=["charge"]))
    assert result.request_scope["behavior_requirements"][0]["state_writes"] == ["charge"]
    assert result.structured_sections["state_model"]["specification"]["variables"][-1]["name"] == "charge"


def test_saved_control_plane_owner_cannot_be_reused():
    from minecraft_mod_ai.construction_behavior_state import _validate_replayed_binding
    with pytest.raises(ValueError, match="BEHAVIOR_STATE_OWNER_CONTROL_LABEL"):
        _validate_replayed_binding("charge", {
            "original_name": "charge", "binding": "new_scalar_state",
            "state_name": "charge",
            "variable": {"name": "charge", "owner": "world_state"},
        }, [variable()])


def test_final_state_validation_uses_inner_specification_not_wrapper():
    from minecraft_mod_ai.authored_structured_design import normalize_structured_sections
    from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
    specification = {concern: [] for concern in DETAIL_RECORDS["state_model"]}
    specification["variables"] = [
        {**variable(name="capacity", kind="int"), "default": "not_a_number"}
    ]
    specification["inapplicable_concerns"] = []
    section = {"state_model": {
        "specification": specification, "constraint_evidence_refs": [],
    }}
    with pytest.raises(ValueError, match="STRUCTURED_STATE_VARIABLE_DEFAULT"):
        normalize_structured_sections(section)
