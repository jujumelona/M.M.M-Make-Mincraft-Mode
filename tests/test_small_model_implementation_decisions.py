"""Host-owned implementation lowering; production planning must not depend on a small model."""
from __future__ import annotations

import json
import pytest

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
    authored = bool(kwargs.pop("authored_schema", False))
    compiler = ir.compile_authored_graph if authored else ir.compile_graph
    return compiler(
        router,
        text=text,
        package="example",
        mod_id="test",
        target={"loader": "fabric"},
        **kwargs,
    )


STRICT_DESIGN = (
    "## 1. 행동 계약 (behavior_contract)\nBehavior.\n"
    "## 2. 상태 모델 (state_model)\nState.\n"
    "## 3. 알고리즘 (algorithm)\nAlgorithm.\n"
    "## 4. 통합 (integration)\nIntegration.\n"
    "## 5. 권한 및 네트워크 (authority_and_network)\nAuthority.\n"
    "## 6. 지속성 (persistence)\nPersistence.\n"
    "## 7. 자원 및 UI (resources_and_ui)\nResources.\n"
    "## 8. 실패 및 제한 (failure_and_limits)\nFailures.\n"
)


def test_production_compiler_is_host_owned_and_model_free():
    router = NoPlanningModelRouter()
    graph = compile_with(router, text=STRICT_DESIGN, authored_schema=True)
    assert router.calls == []
    symbols = {node["symbol"] for node in graph["nodes"]}
    assert symbols == {
        "AuthoredStateModel", "AuthoredBehaviorContract", "AuthoredAlgorithm",
        "AuthoredAuthorityNetwork", "AuthoredPersistence", "AuthoredResourcesUi",
        "AuthoredFailureLimits", "AuthoredIntegration",
    }
    assert all(not symbol.startswith("AuthoredGeneric") for symbol in symbols)


def test_only_integration_owns_authored_lifecycle_activation():
    from minecraft_mod_ai.implementation_lifecycle import ACTIVATION_API

    router = NoPlanningModelRouter()
    graph = compile_with(router, text=STRICT_DESIGN, authored_schema=True)
    by_symbol = {node["symbol"]: node for node in graph["nodes"]}

    active = [node["symbol"] for node in graph["nodes"] if node["activation"]]
    assert active == ["AuthoredIntegration"]
    assert ACTIVATION_API in by_symbol["AuthoredIntegration"]["public_api"]

    for symbol, node in by_symbol.items():
        if symbol == "AuthoredIntegration":
            continue
        assert node["activation"] is False
        assert ACTIVATION_API not in node["public_api"]

    # Internal authored units are valid without a fake lifecycle API. This is the
    # contract that prevents failure_and_limits/persistence/etc. from being told
    # to implement initialize() and then being rejected for doing exactly that.
    assert by_symbol["AuthoredFailureLimits"]["public_api"] == []
    assert by_symbol["AuthoredPersistence"]["public_api"] == []


def test_empty_public_api_exception_is_scoped_to_canonical_authored_internal_nodes():
    router = NoPlanningModelRouter()
    graph = compile_with(router, text=STRICT_DESIGN, authored_schema=True)
    failure = next(
        node for node in graph["nodes"]
        if node["symbol"] == "AuthoredFailureLimits"
    )
    assert failure["activation"] is False
    assert failure["public_api"] == []

    generic = {
        "symbol": "GenericInternal",
        "kind": "java",
        "resource_path": "",
        "responsibility": "generic internal",
        "requirements": ["R1"],
        "obligations": ["implement generic internal"],
        "public_api": [],
        "depends_on": [],
        "activation": False,
        "estimated_tokens": 128,
    }
    import pytest

    with pytest.raises(ir.ImplementationGraphError, match="IMPLEMENTATION_IR_INVALID_NODE"):
        ir.validate_node(generic, package="example", mod_id="test", refs={"R1"})


def test_peer_behavior_packets_extend_one_named_owner_without_model_semantics():
    router = NoPlanningModelRouter()
    text = STRICT_DESIGN.replace(
        "Behavior.\n",
        "- Own credits.\n- Persist purchase result.\n- Reject negative credits.\n",
    )
    graph = compile_with(router, text=text, authored_schema=True)
    assert router.calls == []
    behavior = next(
        node for node in graph["nodes"]
        if node["symbol"] == "AuthoredBehaviorContract"
    )
    joined = " ".join(behavior["obligations"])
    assert "Own credits." in joined
    assert "Persist purchase result." in joined
    assert "Reject negative credits." in joined

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
        "## 1. 개요 (overview)\nOverview only.\n"
        "## 2. 행동 계약 (behavior_contract)\n- Buy when funds are sufficient.\n"
        "## 3. 상태 모델 (state_model)\n- Balance is stored as an integer.\n"
        "## 4. 알고리즘 (algorithm)\n- Subtract price from balance.\n"
        "## 5. 통합 (integration)\n- Wire systems into the host lifecycle.\n"
        "## 6. 권한 및 네트워크 (authority_and_network)\n- Server owns transactions.\n"
        "## 7. 지속성 (persistence)\n- Persist balance.\n"
        "## 8. 자원 및 UI (resources_and_ui)\n- Show balance in UI.\n"
        "## 9. 실패 및 제한 (failure_and_limits)\n- Reject negative balance.\n"
        "## 10. 재사용 평가 (reuse_assessment)\nNo donor is required.\n"
        "## 11. 검증 (verification)\nCompile and test.\n"
    )
    graph = compile_with(router, text=text, authored_schema=True)
    symbols = {node["symbol"] for node in graph["nodes"]}
    assert "AuthoredGeneric0" not in symbols
    assert "AuthoredStateModel" in symbols
    assert "AuthoredBehaviorContract" in symbols
    assert "AuthoredIntegration" in symbols
    assert all(
        "Overview only." not in obligation
        for node in graph["nodes"] for obligation in node["obligations"]
    )
    by_symbol = {node["symbol"]: node for node in graph["nodes"]}
    assert by_symbol["AuthoredBehaviorContract"]["depends_on"] == ["AuthoredStateModel"]
    assert "AuthoredStateModel" in by_symbol["AuthoredIntegration"]["depends_on"]
    assert router.calls == []

def test_unanchored_section_prose_is_owned_once_without_synthetic_sibling_work():
    from minecraft_mod_ai.authored_execution_schema import concern_names

    router = NoPlanningModelRouter()
    graph = compile_with(router, text=STRICT_DESIGN, authored_schema=True)
    by_symbol = {node["symbol"]: node for node in graph["nodes"]}

    state = by_symbol["AuthoredStateModel"]
    state_primary = concern_names("state_model")[0]
    assert len(state["obligations"]) == 1
    assert state_primary in state["obligations"][0]

    network = by_symbol["AuthoredAuthorityNetwork"]
    network_primary = concern_names("authority_and_network")[0]
    assert len(network["obligations"]) == 1
    assert network_primary in network["obligations"][0]
    assert router.calls == []


def test_ir_leaf_rejects_unstructured_state_prose_instead_of_inventing_work():
    from minecraft_mod_ai.implementation_graph_execution import (
        ImplementationGraphError,
        _leaf_module,
    )

    router = NoPlanningModelRouter()
    graph = compile_with(router, text=STRICT_DESIGN, authored_schema=True)
    node = next(item for item in graph["nodes"] if item["symbol"] == "AuthoredStateModel")
    request = {
        "target": {"minecraft_version": "1.21.1", "loader": "fabric"},
        "package": "example",
        "mod_id": "test",
        "structured_sections": {},
        "structured_sections_sha256": (
            "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        ),
    }
    with pytest.raises(
        ImplementationGraphError,
        match="IMPLEMENTATION_IR_STRUCTURED_STATE_REQUIRED",
    ):
        _leaf_module(node, graph, request)


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
    assert set(symbols) == {
        "AuthoredStateModel", "AuthoredBehaviorContract", "AuthoredAlgorithm",
        "AuthoredAuthorityNetwork", "AuthoredPersistence", "AuthoredResourcesUi",
        "AuthoredFailureLimits", "AuthoredIntegration",
    }
    assert all(not symbol.startswith("AuthoredGeneric") for symbol in symbols)
    joined = " ".join(
        obligation for node in graph["nodes"] for obligation in node["obligations"]
    )
    assert "Thinking Process" not in joined
    assert "This must never become production work" not in joined
    assert "actors" in joined
    assert "variables" in joined
    assert router.calls == []


def test_authored_schema_omits_absent_optional_role_without_synthetic_work():
    router = NoPlanningModelRouter()
    text = (
        "# behavior_contract\nB\n# state_model\nS\n# algorithm\nA\n"
        "# integration\nI\n# authority_and_network\nN\n# persistence\nP\n"
        "# resources_and_ui\nR\n"
    )

    graph = compile_with(router, text=text, authored_schema=True)

    symbols = {node["symbol"] for node in graph["nodes"]}
    assert "AuthoredFailureLimits" not in symbols
    assert symbols == {
        "AuthoredStateModel",
        "AuthoredBehaviorContract",
        "AuthoredAlgorithm",
        "AuthoredAuthorityNetwork",
        "AuthoredPersistence",
        "AuthoredResourcesUi",
        "AuthoredIntegration",
    }
    assert router.calls == []


@__import__("pytest").mark.parametrize(
    "text,code",
    [
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
