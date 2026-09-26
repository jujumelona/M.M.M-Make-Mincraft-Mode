"""Production native decision frontend, including malformed small-model responses."""
from __future__ import annotations

import copy
import json

import pytest

from minecraft_mod_ai import implementation_ir as ir
from minecraft_mod_ai.implementation_decisions import work_packet
from minecraft_mod_ai.model_adapters.base import NativeToolDecisionRejected
from minecraft_mod_ai.model_router import ModelRouter


class NativeRouter(ModelRouter):
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def _generate_tool_decision_impl(self, role, messages, **kwargs):
        self.calls.append((kwargs["tool_name"], kwargs["parameters"], json.loads(messages[-1]["content"])))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return copy.deepcopy(answer)


BEHAVIOR = {"state_transition": "Wallet owns credits; a purchase deducts the balance once.",
            "success_condition": "A purchase decreases balance by its price.",
            "failure_condition": "Insufficient balance leaves the balance unchanged."}
INTERFACE = {"public_api": ["public static int balance()"],
             "activation": False, "estimated_tokens": 600}
DEPENDENCIES = {"depends_on": []}


def compile_with(router, **kwargs):
    return ir.compile_graph(router, text="# Trading\n- Buy an item with credits.",
                            package="example", mod_id="test", target={"loader": "fabric"}, **kwargs)


def test_production_compiler_uses_small_native_decisions_not_graph_serialization():
    router = NativeRouter([{"symbol": "Wallet", "kind": "java"}, BEHAVIOR, DEPENDENCIES, INTERFACE])
    graph = compile_with(router)
    assert [call[0] for call in router.calls] == [
        "select_implementation_owner", "define_implementation_behavior",
        "select_implementation_dependencies", "define_implementation_interface"]
    for _, schema, _ in router.calls:
        assert len(schema["properties"]) <= 4
        assert not {"nodes", "requirements", "resource_path"}.intersection(schema["properties"])
    assert graph["nodes"][0]["requirements"] == ["R1", "R2"]
    assert json.loads(graph["nodes"][0]["obligations"][0]) == BEHAVIOR


def test_native_missing_field_repairs_only_that_field_and_preserves_other_answers():
    rejection = NativeToolDecisionRejected("define_implementation_behavior", [{
        "original_tool": "define_implementation_behavior", "failure_code": "TOOL_SCHEMA_INVALID",
        "raw_arguments": json.dumps({k: v for k, v in BEHAVIOR.items() if k != "failure_condition"}),
    }])
    router = NativeRouter([{"symbol": "Wallet", "kind": "java"}, rejection,
                           {"failure_condition": BEHAVIOR["failure_condition"]}, DEPENDENCIES, INTERFACE])
    graph = compile_with(router)
    assert router.calls[2][1]["required"] == ["failure_condition"]
    assert set(router.calls[2][2]["accepted_fields"]) == {"state_transition", "success_condition"}
    assert graph["nodes"][0]["requirements"] == ["R1", "R2"]


def test_completed_native_stages_survive_transport_failure_and_resume():
    saved = []
    router = NativeRouter([{"symbol": "Wallet", "kind": "java"}, BEHAVIOR, DEPENDENCIES, OSError("offline")])
    with pytest.raises(OSError, match="offline"):
        compile_with(router, checkpoint=lambda state: saved.append(copy.deepcopy(state)))
    resumed = NativeRouter([INTERFACE])
    graph = compile_with(resumed, resume=saved[-1])
    assert len(resumed.calls) == 1
    assert resumed.calls[0][0] == "define_implementation_interface"
    assert graph["nodes"][0]["symbol"] == "Wallet"


@pytest.mark.parametrize("bad", ["class PlayerActor", "enum State", "static final String[] TRIGGER_EVENTS ="])
def test_logged_pseudo_java_is_repaired_before_any_owner_is_frozen(bad):
    router = NativeRouter([{"symbol": "Wallet", "kind": "java"}, BEHAVIOR, DEPENDENCIES,
                           {**INTERFACE, "public_api": [bad]}, {"public_api": INTERFACE["public_api"]}])
    graph = compile_with(router)
    assert router.calls[-1][1]["required"] == ["public_api"]
    assert graph["nodes"][0]["public_api"] == INTERFACE["public_api"]


def test_malformed_graph_strings_cannot_look_like_progress_or_reset_on_resume():
    saved = []
    router = NativeRouter([{"nodes": '[{"symbol":"Wallet"}]]'}, {"nodes": '[{"symbol":"Wallet"}]}' }])
    with pytest.raises(ir.ImplementationGraphError, match="IMPLEMENTATION_DECISION_NO_PROGRESS"):
        compile_with(router, checkpoint=lambda state: saved.append(copy.deepcopy(state)))
    with pytest.raises(ir.ImplementationGraphError, match="IMPLEMENTATION_DECISION_NO_PROGRESS"):
        compile_with(router, resume=saved[-1])
    assert len(router.calls) == 2


def test_packet_keeps_all_subordinate_state_fields_and_failure_conditions():
    context = {"R1": "# State", "R2": "- Balance:", "R3": "  - Player owns credits.",
               "R4": "  - Reject negative credits.", "R5": "- Market owns stock."}
    packet = work_packet(context, context)
    assert list(packet["requirements"]) == ["R1", "R2", "R3", "R4"]
    second = work_packet({"R5": context["R5"]}, context)
    assert list(second["requirements"]) == ["R5"]
    assert second["context"]["R2"] == "- Balance:"
    assert second["context"]["R4"] == "  - Reject negative credits."


def test_dependency_contract_is_explicit_in_coder_handoff():
    from minecraft_mod_ai.implementation_graph_execution import _leaf_module
    dependency = {"symbol": "Wallet", "path": "src/main/java/example/Wallet.java",
                  "public_api": INTERFACE["public_api"], "responsibility": "Own credits"}
    node = {"symbol": "Trade", "path": "src/main/java/example/Trade.java", "requirements": ["R1"],
            "public_api": ["public static void buy()"], "depends_on": ["Wallet"], "activation": False,
            "responsibility": "Trade once", "obligations": [json.dumps(BEHAVIOR)]}
    module = _leaf_module(node, {"nodes": [dependency, node], "requirements": {"R1": "Trade once"}},
                          {"target": {"minecraft_version": "26.2", "loader": "fabric"}})
    rendered = json.dumps(module.config)
    assert module.config["evidence_task"]["depends_on"] == ["ir_wallet"]
    assert module.config["evidence_task"]["consumes"] == ["Wallet"]
    assert BEHAVIOR["failure_condition"] in rendered
    assert "public static int balance()" in rendered


def test_internal_schema_rejects_logged_type_declarations_too():
    from test_implementation_ir import node
    raw = node(api=["class PlayerActor"])
    with pytest.raises(ir.ImplementationGraphError):
        ir.validate_node(raw, package="example", mod_id="test", refs=set(raw["requirements"]))


def test_wrapper_corruption_ranks_worse_than_missing_node_fields():
    field_errors = {"diagnostics": [{"code": "IMPLEMENTATION_IR_SCHEMA_INVALID", "field": f"nodes.{i}.requirements"} for i in range(8)]}
    shape_error = {"diagnostics": [{"code": "IMPLEMENTATION_IR_SCHEMA_INVALID", "field": "nodes"}]}
    assert ir._repair_measure(shape_error) > ir._repair_measure(field_errors)


def test_existing_owner_receives_only_new_packet_and_retains_failure_contracts():
    router = NativeRouter([
        {"symbol": "Wallet", "kind": "java"}, BEHAVIOR, DEPENDENCIES, INTERFACE,
        {"symbol": "Wallet", "kind": "java"}, {**BEHAVIOR, "state_transition": "Persist Wallet credits."},
        DEPENDENCIES, {**INTERFACE, "public_api": ["public static void save()"]},
    ])
    graph = ir.compile_graph(router, text="# Trading\n- Own credits.\n- Persist credits.",
                            package="example", mod_id="test", target={"loader": "fabric"})
    owner, = graph["nodes"]
    assert owner["requirements"] == ["R1", "R2", "R3"]
    assert owner["public_api"] == ["public static int balance()", "public static void save()"]
    assert len(owner["obligations"]) == 2
    assert list(router.calls[4][2]["work_packet"]["requirements"]) == ["R3"]
    assert router.calls[5][2]["owner"]["public_api"] == INTERFACE["public_api"]


def test_missing_dependency_is_forced_by_host_and_receives_consumers():
    router = NativeRouter([
        {"symbol": "Trade", "kind": "java"}, BEHAVIOR, {"depends_on": ["Wallet"]}, INTERFACE,
        BEHAVIOR, DEPENDENCIES, INTERFACE,
    ])
    graph = compile_with(router)
    assert [n["symbol"] for n in graph["nodes"]] == ["Wallet", "Trade"]
    assert router.calls[4][0] == "define_implementation_behavior"
    assert router.calls[4][2]["owner"]["symbol"] == "Wallet"
    assert router.calls[4][2]["consumers"][0]["symbol"] == "Trade"
    assert router.calls[5][1]["properties"]["depends_on"]["items"]["enum"] == ["Trade"]


def test_known_dependency_api_arrives_before_consumers_interface_decision():
    router = NativeRouter([
        {"symbol": "Wallet", "kind": "java"}, BEHAVIOR, DEPENDENCIES, INTERFACE,
        {"symbol": "Trade", "kind": "java"}, BEHAVIOR, {"depends_on": ["Wallet"]},
        {**INTERFACE, "public_api": ["public static void buy()"]},
    ])
    ir.compile_graph(router, text="# Trading\n- Own credits.\n- Trade using credits.",
                     package="example", mod_id="test", target={"loader": "fabric"})
    assert router.calls[-1][2]["dependencies"][0]["public_api"] == INTERFACE["public_api"]


def test_native_decomposition_keeps_facade_and_pins_helper_dependencies():
    from test_implementation_ir import node
    original = ir.validate_node(node(cost=4000), package="example", mod_id="test",
                                refs={f"R{i}" for i in range(1, 7)})
    router = NativeRouter([
        {"symbol": "PlayerCreditsPartStore"},
        {**BEHAVIOR, "facade_work": "Delegate balance storage to PlayerCreditsPartStore."},
        {"public_api": ["public static int balance()"], "depends_on": [],
         "estimated_tokens": 1000, "facade_estimated_tokens": 1200},
    ])
    nodes = ir.refine_node(router, original, nodes=[original], package="example", mod_id="test",
                           requirements={f"R{i}": "Own balance" for i in range(1, 7)},
                           reason="OUTPUT_BUDGET_EXHAUSTED", budget=2000)
    assert [n["symbol"] for n in nodes] == ["PlayerCreditsPartStore", "PlayerCredits"]
    assert nodes[1]["public_api"] == original["public_api"]
    assert nodes[1]["depends_on"] == ["PlayerCreditsPartStore"]
    assert all("nodes" not in c[1]["properties"] for c in router.calls)


def test_resource_location_is_a_separate_namespaced_decision():
    router = NativeRouter([
        {"symbol": "ItemNames", "kind": "resource"}, BEHAVIOR, DEPENDENCIES,
        {"public_api": [], "activation": False, "estimated_tokens": 300},
        {"resource_path": "src/main/resources/assets/other/lang/en_us.json"},
        {"resource_path": "src/main/resources/assets/test/lang/en_us.json"},
    ])
    graph = compile_with(router)
    assert graph["nodes"][0]["path"] == "src/main/resources/assets/test/lang/en_us.json"
    assert router.calls[-1][1]["required"] == ["resource_path"]


def test_dependency_cycle_repairs_only_edge_and_keeps_selected_owner():
    router = NativeRouter([
        {"symbol": "Trade", "kind": "java"}, BEHAVIOR, {"depends_on": ["Wallet"]}, INTERFACE,
        BEHAVIOR, {"depends_on": ["Trade"]}, DEPENDENCIES, INTERFACE,
    ])
    graph = compile_with(router)
    assert [n["symbol"] for n in graph["nodes"]] == ["Wallet", "Trade"]
    assert router.calls[6][1]["required"] == ["depends_on"]
    assert "cycle" in router.calls[6][2]["correct_only"]["depends_on"]


def test_packet_does_not_split_fenced_examples_into_new_work():
    context = {"R1": "# UI", "R2": "- Display this literal example:", "R3": "  ```markdown",
               "R4": "# Not a heading", "R5": "- Not another behavior", "R6": "  ```",
               "R7": "- Close the screen."}
    packet = work_packet(context, context)
    assert list(packet["requirements"]) == ["R1", "R2", "R3", "R4", "R5", "R6"]


def test_native_rejection_of_otherwise_valid_arguments_is_not_swallowed():
    router = NativeRouter([NativeToolDecisionRejected("select_implementation_owner", [{
        "original_tool": "select_implementation_owner", "failure_code": "WRONG_NATIVE_CONTRACT",
        "raw_arguments": json.dumps({"symbol": "Wallet", "kind": "java"}),
    }])])
    with pytest.raises(NativeToolDecisionRejected):
        compile_with(router)
    assert len(router.calls) == 1


def test_old_cached_pseudo_java_contract_is_rejected_before_source_generation(tmp_path):
    from test_implementation_ir import graph_project, node

    from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
    module, main = graph_project(tmp_path)
    before = main.read_bytes()
    request = module.config["implementation_graph_request"]
    raw = node(api=["class PlayerCredits"])
    raw["path"] = "src/main/java/example/PlayerCredits.java"
    graph = {"source_text": request["text"], "nodes": [raw],
             "requirements": ir.source_requirements(request["text"])}
    cache = tmp_path / f".minecraft_ai/implementation-ir/{ir.digest(request)}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"request_hash": ir.digest(request), "graph": graph,
                                "graph_hash": ir.digest(graph), "blocked_decodes": [], "refinements": 0}))
    router = NativeRouter([])
    with pytest.raises(ir.ImplementationGraphError, match="INVALID_NODE"):
        CustomModuleGenerator(router).generate(tmp_path, module=module)
    assert not router.calls
    assert main.read_bytes() == before
