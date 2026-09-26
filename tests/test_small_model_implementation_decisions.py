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


def test_canonical_design_schema_skips_context_sections_and_uses_named_owners():
    router = NoPlanningModelRouter()
    text = (
        "# Stellar Odyssey Mod Design Document\n"
        "## 개요\nOverview only.\n"
        "# behavior_contract\n- Buy when funds are sufficient.\n"
        "# state_model\n- Balance is stored as an integer.\n"
        "# algorithm\n- Subtract price from balance.\n"
        "# integration\n- Wire systems into the host lifecycle.\n"
        "# authority_and_network\n- Server owns transactions.\n"
        "# persistence\n- Persist balance.\n"
        "# resources_and_ui\n- Show balance in UI.\n"
        "# failure_and_limits\n- Reject negative balance.\n"
        "# reuse_assessment\nNo donor is required.\n"
        "# verification\nCompile and test.\n"
    )
    graph = compile_with(router, text=text, authored_schema=True)
    symbols = [node["symbol"] for node in graph["nodes"]]
    assert "AuthoredUnit0" not in symbols
    assert "AuthoredStateModel" in symbols
    assert "AuthoredBehaviorContract" in symbols
    assert "AuthoredIntegration" in symbols
    assert all("Overview only." not in obligation for node in graph["nodes"] for obligation in node["obligations"])
    by_symbol = {node["symbol"]: node for node in graph["nodes"]}
    assert by_symbol["AuthoredBehaviorContract"]["depends_on"] == ["AuthoredStateModel"]
    assert "AuthoredStateModel" in by_symbol["AuthoredIntegration"]["depends_on"]
    assert router.calls == []


def test_host_section_nodes_expand_to_fixed_concern_obligations_without_model_planning():
    from minecraft_mod_ai.authored_execution_schema import concern_names

    router = NoPlanningModelRouter()
    text = (
        "# state_model\nState.\n"
        "# behavior_contract\nBehavior.\n"
        "# algorithm\nAlgorithm.\n"
        "# authority_and_network\nAuthority.\n"
        "# persistence\nPersistence.\n"
        "# resources_and_ui\nResources.\n"
        "# failure_and_limits\nFailures.\n"
        "# integration\nIntegration.\n"
    )
    graph = compile_with(router, text=text, authored_schema=True)
    by_symbol = {node["symbol"]: node for node in graph["nodes"]}
    state = by_symbol["AuthoredStateModel"]
    assert len(state["obligations"]) == len(concern_names("state_model"))
    joined = " ".join(state["obligations"])
    assert all(name in joined for name in concern_names("state_model"))
    network = by_symbol["AuthoredAuthorityNetwork"]
    assert len(network["obligations"]) == len(concern_names("authority_and_network"))
    assert router.calls == []


def test_ir_leaf_carries_same_fixed_concern_sequence_into_coder_contract():
    from minecraft_mod_ai.authored_execution_schema import concern_names
    from minecraft_mod_ai.implementation_graph_execution import _leaf_module

    router = NoPlanningModelRouter()
    text = (
        "# state_model\nState.\n"
        "# behavior_contract\nBehavior.\n"
        "# algorithm\nAlgorithm.\n"
        "# authority_and_network\nAuthority.\n"
        "# persistence\nPersistence.\n"
        "# resources_and_ui\nResources.\n"
        "# failure_and_limits\nFailures.\n"
        "# integration\nIntegration.\n"
    )
    graph = compile_with(router, text=text, authored_schema=True)
    node = next(item for item in graph["nodes"] if item["symbol"] == "AuthoredStateModel")
    request = {
        "target": {"minecraft_version": "1.21.1", "loader": "fabric"},
        "package": "example",
        "mod_id": "test",
    }
    leaf = _leaf_module(node, graph, request)
    concerns = leaf.config["implementation_atomic_concerns"]
    assert [item["concern"] for item in concerns] == list(concern_names("state_model"))
    assert leaf.config["implementation_section"] == "state_model"
    assert len(leaf.config["evidence_task"]["implementation_obligations"]) == len(concerns)


def test_localized_canonical_headings_keep_nested_concerns_and_drop_preamble():
    router = NoPlanningModelRouter()
    text = (
        "Thinking Process:\nThis must never become production work.\n"
        "# 우주 모드 게임 디자인 문서\n"
        "## 1. 개요 (Overview)\nOverview only.\n"
        "## 2. 행동 계약 (behavior_contract)\n### actors\nPlayer and server.\n"
        "## 3. 상태 모델 (state_model)\n### variables\nCredits and ship state.\n"
        "## 4. 알고리즘 (algorithm)\n### steps\nValidate, mutate, persist.\n"
        "## 5. 통합 (integration)\n### entry_points\nHost initialize hook.\n"
        "## 6. 권한 및 네트워크 (authority_and_network)\n### packets\nServer authoritative packet.\n"
        "## 7. 지속성 (persistence)\n### stored_state\nPersist credits.\n"
        "## 8. 자원 및 UI (resources_and_ui)\n### displayed_state\nShow credits.\n"
        "## 9. 실패 및 제한 (failure_and_limits)\n### invalid_inputs\nReject negative amounts.\n"
        "## 10. 재사용 평가 (reuse_assessment)\nReview only.\n"
        "## 11. 검증 (verification)\nTests only.\n"
        "## 12. 결론\nSummary only.\n"
    )
    graph = compile_with(router, text=text, authored_schema=True)
    symbols = [node["symbol"] for node in graph["nodes"]]
    assert symbols == [
        "AuthoredStateModel", "AuthoredBehaviorContract", "AuthoredAlgorithm",
        "AuthoredAuthorityNetwork", "AuthoredPersistence", "AuthoredResourcesUi",
        "AuthoredFailureLimits", "AuthoredIntegration",
    ]
    joined = " ".join(
        obligation for node in graph["nodes"] for obligation in node["obligations"]
    )
    assert "Thinking Process" not in joined
    assert "This must never become production work" not in joined
    assert "actors" in joined
    assert "variables" in joined
    assert router.calls == []


@__import__("pytest").mark.parametrize(
    "text,code",
    [
        (
            "# behavior_contract\nB\n# state_model\nS\n# algorithm\nA\n"
            "# integration\nI\n# authority_and_network\nN\n# persistence\nP\n"
            "# resources_and_ui\nR\n",
            "IMPLEMENTATION_IR_AUTHORED_SECTION_MISSING",
        ),
        (
            "# behavior_contract\nB\n# state_model\nS\n# state_model\nS2\n"
            "# algorithm\nA\n# integration\nI\n# authority_and_network\nN\n"
            "# persistence\nP\n# resources_and_ui\nR\n# failure_and_limits\nF\n",
            "IMPLEMENTATION_IR_AUTHORED_SECTION_DUPLICATE",
        ),
        (
            "# state_model\nS\n# behavior_contract\nB\n# algorithm\nA\n"
            "# integration\nI\n# authority_and_network\nN\n# persistence\nP\n"
            "# resources_and_ui\nR\n# failure_and_limits\nF\n",
            "IMPLEMENTATION_IR_AUTHORED_SECTION_ORDER",
        ),
    ],
)
def test_authored_schema_has_no_generic_fallback(text, code):
    router = NoPlanningModelRouter()
    with __import__("pytest").raises(ir.ImplementationGraphError, match=code):
        compile_with(router, text=text, authored_schema=True)
