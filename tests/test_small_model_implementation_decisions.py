"""Host-owned implementation lowering; production planning must not depend on a small model."""
from __future__ import annotations

import json

from minecraft_mod_ai import implementation_ir as ir
from minecraft_mod_ai.implementation_decisions import work_packet
from minecraft_mod_ai.model_router import ModelRouter


class NoPlanningModelRouter(ModelRouter):
    def __init__(self):
        self.calls = []

    def _generate_tool_decision_impl(self, role, messages, **kwargs):
        self.calls.append((role, kwargs.get("tool_name")))
        raise AssertionError("implementation lowering must not call the text model")


def compile_with(router, *, text="# Trading\n- Buy an item with credits.", **kwargs):
    return ir.compile_graph(
        router,
        text=text,
        package="example",
        mod_id="test",
        target={"loader": "fabric"},
        **kwargs,
    )


def test_production_compiler_is_host_owned_and_model_free():
    router = NoPlanningModelRouter()
    graph = compile_with(router)
    assert router.calls == []
    node, = graph["nodes"]
    assert node["symbol"] == "AuthoredUnit0"
    assert node["kind"] == "java"
    assert node["activation"] is True
    assert node["depends_on"] == []
    assert node["public_api"] == ["public static void initialize()"]
    assert node["requirements"] == ["R1", "R2"]
    obligation = json.loads(node["obligations"][0])
    assert obligation["source_requirements"] == {
        "R1": "# Trading",
        "R2": "- Buy an item with credits.",
    }


def test_peer_packets_merge_into_one_host_owner_without_model_semantics():
    router = NoPlanningModelRouter()
    graph = compile_with(
        router,
        text="# Trading\n- Own credits.\n- Persist credits.\n- Reject negative credits.",
    )
    assert router.calls == []
    node, = graph["nodes"]
    assert node["symbol"] == "AuthoredUnit0"
    assert node["requirements"] == ["R1", "R2", "R3", "R4"]
    assert len(node["obligations"]) == 3
    captured = set()
    for raw in node["obligations"]:
        captured.update(json.loads(raw)["source_requirements"])
    assert captured == {"R1", "R2", "R3", "R4"}


def test_host_refinement_splits_budget_without_model_call():
    router = NoPlanningModelRouter()
    requirements = {
        "R1": "# Trading",
        "R2": "- Own credits.",
        "R3": "- Persist credits.",
    }
    original = ir.validate_node(
        {
            "symbol": "AuthoredUnit0",
            "kind": "java",
            "resource_path": "",
            "responsibility": "Implement the approved authored unit: Trading",
            "requirements": list(requirements),
            "obligations": [json.dumps({"source_requirements": requirements})],
            "public_api": [],
            "depends_on": [],
            "activation": True,
            "estimated_tokens": 4000,
        },
        package="example",
        mod_id="test",
        refs=set(requirements),
    )
    refined = ir.refine_node(
        router,
        original,
        nodes=[original],
        package="example",
        mod_id="test",
        requirements=requirements,
        reason="OUTPUT_BUDGET_EXHAUSTED",
        budget=2000,
    )
    assert router.calls == []
    assert [node["symbol"] for node in refined] == [
        "AuthoredUnit0Part1",
        "AuthoredUnit0",
    ]
    helper, facade = refined
    assert helper["public_api"] == ["public static void run()"]
    assert helper["activation"] is False
    assert facade["public_api"] == ["public static void initialize()"]
    assert facade["depends_on"] == [helper["symbol"]]
    assert helper["estimated_tokens"] < original["estimated_tokens"]
    assert facade["estimated_tokens"] < original["estimated_tokens"]
    assert set(helper["requirements"]) | set(facade["requirements"]) == set(requirements)


def test_packet_keeps_all_subordinate_state_fields_and_failure_conditions():
    context = {
        "R1": "# State",
        "R2": "- Balance:",
        "R3": "  - Player owns credits.",
        "R4": "  - Reject negative credits.",
        "R5": "- Market owns stock.",
    }
    packet = work_packet(context, context)
    assert list(packet["requirements"]) == ["R1", "R2", "R3", "R4"]
    second = work_packet({"R5": context["R5"]}, context)
    assert list(second["requirements"]) == ["R5"]
    assert second["context"]["R2"] == "- Balance:"
    assert second["context"]["R4"] == "  - Reject negative credits."


def test_single_requirement_refinement_is_finite_by_strictly_decreasing_cost():
    router = NoPlanningModelRouter()
    requirements = {"R1": "One indivisible authored requirement."}
    original = ir.validate_node(
        {
            "symbol": "AuthoredUnit0",
            "kind": "java",
            "resource_path": "",
            "responsibility": "Implement one requirement",
            "requirements": ["R1"],
            "obligations": [json.dumps({"source_requirements": requirements})],
            "public_api": [],
            "depends_on": [],
            "activation": True,
            "estimated_tokens": 8,
        },
        package="example",
        mod_id="test",
        refs={"R1"},
    )
    refined = ir.refine_node(
        router,
        original,
        nodes=[original],
        package="example",
        mod_id="test",
        requirements=requirements,
        reason="OUTPUT_BUDGET_EXHAUSTED",
        budget=4,
    )
    assert router.calls == []
    assert max(node["estimated_tokens"] for node in refined) == 4
