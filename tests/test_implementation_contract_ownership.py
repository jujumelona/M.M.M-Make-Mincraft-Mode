"""Regression shapes from the AstraCraft Colab trace, without a live model."""
from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator
from test_implementation_ir import TARGET, Decisions, node

from minecraft_mod_ai import implementation_ir as ir
from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _implementation_authored_plan


def test_logged_korean_design_title_projects_exact_final_document():
    prefix = "Thinking Process:\n\n1. **Analyze the Request:**\n   Draft a space mod.\n\n"
    design = (
        "# 아스트라크래프트: 우주 프론티어 (AstraCraft: Space Frontier) 모드 설계서\n\n"
        "## # behavior_contract\n우주선을 조립하고 발사한다.\n"
        "## # state_model\n서버가 연료 상태를 소유한다.\n"
    )
    plan = AuthoredPlan(requested_prompt="우주 모드", text=prefix + design)
    projected, provenance = _implementation_authored_plan(plan)
    assert projected.text == design
    assert provenance["source_plan"] == plan.to_dict()
    assert provenance["stripped_prefix_bytes"] == len(prefix.encode())
    assert all("Thinking Process" not in v for v in ir.source_requirements(projected.text).values())
    units = ir.decompose_authored_units(projected.text)
    assert len(units) == 2
    assert units[0]["title"] == "behavior_contract"
    assert next(iter(units[0]["requirements"].values())) == design.splitlines()[0]


def test_activation_api_is_host_derived_before_coder_contract_freezes():
    raw = node("VerificationTests", activation=True, api=[
        "public final void onInitialize()",
        "public boolean testFuelExhaustionLogic(int fuelAmount, int consumptionRate)",
    ])
    errors = list(Draft202012Validator(ir.PAGE_SCHEMA).iter_errors({"nodes": [raw]}))
    assert not errors, [e.message for e in errors]
    admitted = ir.validate_node(raw, package="example", mod_id="test", refs=set(raw["requirements"]))
    assert admitted["public_api"] == [*raw["public_api"], "public static void initialize()"]
    assert raw["public_api"] == admitted["public_api"][:-1]


def test_integration_only_node_does_not_need_an_invented_domain_api():
    raw = {**node("RegisterEvents", activation=True), "public_api": []}
    Draft202012Validator(ir.PAGE_SCHEMA).validate({"nodes": [raw]})
    admitted = ir.validate_node(raw, package="example", mod_id="test", refs=set(raw["requirements"]))
    assert admitted["public_api"] == ["public static void initialize()"]


def test_activation_cannot_be_disabled_by_a_later_contribution():
    text = "# Register\nRegister events.\n# State\nOwn fuel."
    owner = node("ShipState", refs=["R1", "R2"], activation=True)
    router = NativeSchemaDecisions([
        {"nodes": [owner]},
        {"nodes": [{"symbol": "ShipState", "requirements": ["R3", "R4"], "activation": False}]},
    ])
    graph = ir.compile_graph(router, text=text, package="example", mod_id="test", target=TARGET)
    assert graph["nodes"][0]["activation"] is True
    assert graph["nodes"][0]["public_api"].count("public static void initialize()") == 1


class NativeSchemaDecisions(Decisions):
    """Exercise the actual schema passed to native decoding on every response."""

    def generate_tool_decision(self, role, messages, **kwargs):
        result = super().generate_tool_decision(role, messages, **kwargs)
        Draft202012Validator(kwargs["parameters"]).validate(result)
        return result


@pytest.mark.parametrize("full_restatement", [False, True])
def test_existing_owner_accepts_incremental_work_and_activation_promotion(full_restatement):
    text = "# State\nOwn fuel.\n# Integration\nRegister server events."
    owner = node("ShipState", refs=["R1", "R2"], api=["public int fuel()"])
    contribution = {"symbol": "ShipState", "requirements": ["R3", "R4"], "activation": True}
    if full_restatement:
        contribution = {**copy.deepcopy(owner), **contribution, "obligations": []}
    router = NativeSchemaDecisions([{"nodes": [owner]}, {"nodes": [contribution]}])
    graph = ir.compile_graph(router, text=text, package="example", mod_id="test", target=TARGET)
    assert len(router.calls) == 2
    result, = graph["nodes"]
    assert result["requirements"] == ["R1", "R2", "R3", "R4"]
    assert result["obligations"] == owner["obligations"]
    assert result["activation"] is True
    assert result["public_api"] == ["public int fuel()", "public static void initialize()"]


def test_new_owner_still_requires_complete_contract():
    raw = {"symbol": "Unknown", "requirements": ["R1"]}
    schema = ir._page_schema({"requirements": {"R1": "fuel"}, "accepted_nodes": [node()]})
    assert list(Draft202012Validator(schema).iter_errors({"nodes": [raw]}))


@pytest.mark.parametrize("api", ["public void initialize()", "public static int initialize()"])
def test_conflicting_activation_signature_is_explicitly_rejected(api):
    raw = node(activation=True, api=[api])
    with pytest.raises(ir.ImplementationGraphError, match="ACTIVATION_API_CONFLICT"):
        ir.validate_node(raw, package="example", mod_id="test", refs=set(raw["requirements"]))


def test_incremental_request_carries_existing_obligations_and_estimate():
    raw = node()
    view = ir._model_node_view(raw)
    assert view["obligations"] == raw["obligations"]
    assert view["estimated_tokens"] == raw["estimated_tokens"]
    assert json.loads(json.dumps(view)) == view


def test_promotion_checks_the_merged_api_including_old_instance_initialize():
    old = node(api=["public void initialize()"], refs=["R1"])
    new = node(api=["public int fuel()"], refs=["R2"], activation=True)
    old = ir.validate_node(old, package="example", mod_id="test", refs={"R1", "R2"})
    new = ir.validate_node(new, package="example", mod_id="test", refs={"R1", "R2"})
    with pytest.raises(ir.ImplementationGraphError, match="ACTIVATION_API_CONFLICT"):
        ir._merge_accepted_owner(old, new)


def test_incremental_member_body_uses_the_same_canonicalization_as_new_nodes():
    text = "# State\nOwn fuel.\n# API\nRead fuel."
    owner = node("ShipState", refs=["R1", "R2"])
    router = Decisions([
        {"nodes": [owner]},
        {"nodes": [{"symbol": "ShipState", "requirements": ["R3", "R4"],
                    "public_api": ["public int fuel() { return 1; }"]}]},
    ])
    graph = ir.compile_graph(router, text=text, package="example", mod_id="test", target=TARGET)
    assert "public int fuel()" in graph["nodes"][0]["public_api"]
    assert len(router.calls) == 2


@pytest.mark.parametrize("symbol", [["ShipState"], {"name": "ShipState"}])
def test_malformed_symbol_remains_a_repairable_schema_failure(symbol):
    raw = node()
    raw["symbol"] = symbol
    router = Decisions([{"nodes": [raw]}, {"nodes": [node()]}])
    graph = ir.compile_graph(router, text="\n".join(["Own fuel."] * 6),
                             package="example", mod_id="test", target=TARGET)
    assert graph["nodes"][0]["symbol"] == "PlayerCredits"
    assert len(router.calls) == 2
