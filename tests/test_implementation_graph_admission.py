"""Model/host graph contract tests, including invalid native tool responses."""
from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from test_implementation_ir import DESIGN, TARGET, Decisions, node

from minecraft_mod_ai import implementation_ir as ir
from minecraft_mod_ai.model_adapters.base import (
    GenerationResponse,
    NativeToolDecisionRejected,
    ToolCall,
)
from minecraft_mod_ai.model_router import ModelRouter


def compile_graph(router, **kwargs):
    return ir.compile_graph(router, text=DESIGN, package="example", mod_id="test", target=TARGET, **kwargs)


@pytest.mark.parametrize("field,value", [("public_api", []), ("resource_path", "src/main/java/example/PlayerCredits.java")])
def test_model_schema_rejects_the_same_java_contract_errors_as_host(field, value):
    invalid = node()
    invalid[field] = value
    errors = list(Draft202012Validator(ir.PAGE_SCHEMA).iter_errors({"nodes": [invalid], "done": True}))
    assert errors, "The model schema must express the host's Java API/path contract"
    assert any(field in str(e) for e in errors)


@pytest.mark.parametrize("field,value", [("public_api", []), ("resource_path", "src/main/java/example/PlayerCredits.java")])
def test_invalid_page_receives_field_diagnostic_then_continues_without_dropping_requirements(field, value):
    invalid = node()
    invalid[field] = value
    router = Decisions([{"nodes": [invalid], "done": True}, {"nodes": [node()], "done": True}])
    graph = compile_graph(router)
    assert graph["nodes"][0]["public_api"] == node()["public_api"]
    assert graph["source_text"] == DESIGN
    feedback = router.calls[1][1]["validation_feedback"]
    assert field in json.dumps(feedback)
    assert feedback["rejected_page"]["nodes"][0] == invalid
    assert feedback["diagnostics"][0]["node"] == "PlayerCredits"


def test_rejected_page_is_atomic_and_valid_sibling_is_preserved():
    first = node()
    second = node("TradeService", dependencies=["PlayerCredits"])
    invalid = {**second, "public_api": []}
    router = Decisions([{"nodes": [first, invalid], "done": True},
                        {"nodes": [first, second], "done": True}])
    graph = compile_graph(router)
    assert [n["symbol"] for n in graph["nodes"]] == ["PlayerCredits", "TradeService"]
    assert router.calls[1][1]["accepted_nodes"] == []
    assert router.calls[1][1]["validation_feedback"]["preserve_nodes"] == [first]


def test_same_rejected_page_stops_without_a_third_identical_decode():
    invalid = {**node(), "public_api": []}
    router = Decisions([{"nodes": [invalid], "done": True}] * 5)
    with pytest.raises(ir.ImplementationGraphError, match="NO_PROGRESS") as caught:
        compile_graph(router)
    assert len(router.calls) == 2
    assert "public_api" in str(caught.value)


def test_premature_done_requests_only_remaining_coverage():
    first = node(refs=["R1", "R2", "R3"])
    second = node("TradeService", refs=["R4", "R5", "R6"])
    router = Decisions([{"nodes": [first], "done": True}, {"nodes": [second], "done": True}])
    graph = compile_graph(router)
    assert len(graph["nodes"]) == 2
    next_page = router.calls[1][1]
    assert next_page["remaining_requirements"] == ["R4", "R5", "R6"]
    assert next_page["accepted_nodes"][0]["symbol"] == "PlayerCredits"


def test_later_requirements_monotonically_extend_an_accepted_owner():
    first = node(refs=["R1", "R2", "R3"], api=["public static int balance()"])
    first["obligations"] = ["Own the balance state."]

    extension = node(
        refs=["R4", "R5", "R6"],
        api=["public static int balance()", "public static boolean spend(int amount)"],
    )
    extension["obligations"] = ["Reject spending when balance is insufficient."]

    router = Decisions([
        {"nodes": [first], "done": False},
        {"nodes": [extension], "done": True},
    ])
    graph = compile_graph(router)

    assert len(graph["nodes"]) == 1
    merged = graph["nodes"][0]
    assert merged["symbol"] == "PlayerCredits"
    assert merged["requirements"] == ["R1", "R2", "R3", "R4", "R5", "R6"]
    assert merged["obligations"] == [
        "Own the balance state.",
        "Reject spending when balance is insufficient.",
    ]
    assert merged["public_api"] == [
        "public static int balance()",
        "public static boolean spend(int amount)",
    ]
    assert len(router.calls) == 2


def test_duplicate_owner_cannot_rewrite_an_existing_api_contract():
    first = node(refs=["R1", "R2", "R3"], api=["public static int balance()"])
    conflicting = node(refs=["R4", "R5", "R6"], api=["public static long balance()"])

    router = Decisions([
        {"nodes": [first], "done": False},
        {"nodes": [conflicting], "done": True},
        {"nodes": [conflicting], "done": True},
    ])
    with pytest.raises(ir.ImplementationGraphError, match="DUPLICATE_CONTRACT_CONFLICT"):
        compile_graph(router)
    assert len(router.calls) == 3


def test_repeated_accepted_owner_is_ignored_when_same_page_adds_real_new_work():
    refs = ir.source_requirements(DESIGN)
    accepted_raw = node(refs=["R1", "R2", "R3"])
    accepted = ir.validate_node(
        accepted_raw, package="example", mod_id="test", refs=set(refs)
    )
    duplicate_raw = copy.deepcopy(accepted_raw)
    new_raw = node("TradeService", refs=["R4", "R5", "R6"], dependencies=["PlayerCredits"])

    combined = ir._admit_graph_page(
        {"nodes": [duplicate_raw, new_raw], "done": True},
        accepted=[accepted],
        package="example",
        mod_id="test",
        requirements=refs,
    )

    assert [item["symbol"] for item in combined] == ["PlayerCredits", "TradeService"]
    assert combined[0] == accepted


def test_completed_pages_survive_failure_and_resume_without_replanning():
    first = node(refs=["R1", "R2", "R3"])
    second = node("TradeService", refs=["R4", "R5", "R6"])
    saved = []

    class Interrupted(Decisions):
        def generate_tool_decision(self, *args, **kwargs):
            if not self.pages:
                raise ConnectionError("transport stopped")
            return super().generate_tool_decision(*args, **kwargs)

    with pytest.raises(ConnectionError):
        compile_graph(Interrupted([{"nodes": [first], "done": False}]), checkpoint=lambda s: saved.append(copy.deepcopy(s)))
    assert saved[-1]["nodes"][0]["symbol"] == "PlayerCredits"
    router = Decisions([{"nodes": [second], "done": True}])
    graph = compile_graph(router, resume=saved[-1])
    assert len(graph["nodes"]) == 2
    assert len(router.calls) == 1
    assert router.calls[0][1]["page"] == 2


def test_native_router_retains_schema_rejection_fields_for_compiler_feedback():
    raw = json.dumps({"nodes": [{**node(), "public_api": []}], "done": True})
    rejection = {"failure_code": "TOOL_SCHEMA_INVALID", "original_tool": "compile_implementation_graph",
                 "raw_arguments": raw, "error": "nodes.0.public_api must be non-empty"}

    class Router(ModelRouter):
        def __init__(self):
            pass

        def _generation_adapter(self, role):
            return SimpleNamespace(adapter="llama_cpp"), SimpleNamespace(generate_turn=lambda request: GenerationResponse(
                tool_calls=(ToolCall("rejected", "__mmm_rejected_tool_call__", rejection),)))

        def _generation_scope(self, config):
            return nullcontext()

    with pytest.raises(NativeToolDecisionRejected) as caught:
        Router().generate_tool_decision("planner", [{"role": "user", "content": "compile"}],
                                       tool_name="compile_implementation_graph", parameters=ir.PAGE_SCHEMA)
    assert getattr(caught.value, "rejections", None) == (rejection,)
    assert "nodes.0.public_api" in str(caught.value)


def test_terminal_failure_is_not_retried_on_resume():
    invalid = {**node(), "public_api": []}
    router = Decisions([{"nodes": [invalid], "done": True}] * 2)
    saved = []
    with pytest.raises(ir.ImplementationGraphError, match="NO_PROGRESS"):
        compile_graph(router, checkpoint=lambda s: saved.append(copy.deepcopy(s)))
    with pytest.raises(ir.ImplementationGraphError, match="NO_PROGRESS"):
        compile_graph(router, resume=saved[-1])
    assert len(router.calls) == 2


@pytest.mark.parametrize(
    "stale_version",
    ["mmm/implementation-ir-draft-v1", "mmm/implementation-ir-draft-v2"],
)
def test_stale_terminal_checkpoint_is_invalidated_after_ir_contract_change(stale_version):
    invalid = {**node(), "public_api": []}
    saved = []
    failing = Decisions([{"nodes": [invalid], "done": True}] * 2)
    with pytest.raises(ir.ImplementationGraphError, match="NO_PROGRESS"):
        compile_graph(failing, checkpoint=lambda state: saved.append(copy.deepcopy(state)))

    stale = copy.deepcopy(saved[-1])
    stale["schema_version"] = stale_version
    valid = node()
    router = Decisions([{"nodes": [valid], "done": True}])
    graph = compile_graph(router, resume=stale)

    assert len(router.calls) == 1
    assert graph["nodes"][0]["symbol"] == valid["symbol"]


def test_different_invalid_responses_have_a_finite_correction_budget():
    pages = [{"nodes": [{**node(f"Owner{i}"), "public_api": []}], "done": True} for i in range(4)]
    router = Decisions(pages)
    with pytest.raises(ir.ImplementationGraphError, match="CORRECTION_LIMIT"):
        compile_graph(router)
    assert len(router.calls) == 3


def test_forward_dependencies_are_resolved_on_later_pages():
    consumer = node("TradeService", dependencies=["PlayerCredits"])
    router = Decisions([{"nodes": [consumer], "done": True}, {"nodes": [node()], "done": True}])
    graph = compile_graph(router)
    assert router.calls[1][1]["unresolved_dependencies"] == ["PlayerCredits"]
    assert [n["symbol"] for n in graph["nodes"]] == ["PlayerCredits", "TradeService"]


def test_completed_coverage_can_finish_with_an_empty_terminal_page():
    graph = compile_graph(Decisions([{"nodes": [node()], "done": False}, {"nodes": [], "done": True}]))
    assert len(graph["nodes"]) == 1


def test_native_output_limit_does_not_enter_schema_correction_loop():
    from minecraft_mod_ai.llama_finish_reason_contract import (
        OUTPUT_EXHAUSTED,
        LlamaCompletionBoundaryError,
    )

    class Exhausted(Decisions):
        def generate_tool_decision(self, *args, **kwargs):
            self.calls.append(kwargs)
            raise LlamaCompletionBoundaryError("limit", kind=OUTPUT_EXHAUSTED, completion_tokens=8192, max_tokens=8192)

    router = Exhausted([])
    with pytest.raises(LlamaCompletionBoundaryError):
        compile_graph(router)
    assert len(router.calls) == 3
    assert all("validation_feedback" not in str(call) for call in router.calls)


def test_output_limit_reduces_scope_and_recovers():
    from minecraft_mod_ai.llama_finish_reason_contract import (
        OUTPUT_EXHAUSTED,
        LlamaCompletionBoundaryError,
    )

    first_slice = node("PlayerCredits", refs=["R1", "R2"])
    second_slice = node("TradeService", refs=["R3", "R4", "R5", "R6"], dependencies=["PlayerCredits"])
    calls = []

    class ReducingRouter(Decisions):
        def generate_tool_decision(self, role, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            calls.append(payload)
            req_count = len(payload["requirements"])
            # The initial batch has all 6 requirements and hits output budget limit
            if req_count > 2:
                raise LlamaCompletionBoundaryError("output ceiling", kind=OUTPUT_EXHAUSTED,
                                                   completion_tokens=8192, max_tokens=8192)
            if "R1" in payload["requirements"]:
                return {"nodes": [first_slice], "done": False}
            return {"nodes": [second_slice], "done": True}

    router = ReducingRouter([])
    graph = compile_graph(router)
    assert len(graph["nodes"]) == 2
    # Verify Call 1 hit the ceiling, Call 2 and Call 3 had strictly smaller requirements and succeeded
    assert len(calls) == 3
    assert len(calls[0]["requirements"]) > len(calls[1]["requirements"])
    assert len(calls[1]["requirements"]) <= 2
    assert "validation_feedback" not in calls[1]
    assert [n["symbol"] for n in graph["nodes"]] == ["PlayerCredits", "TradeService"]


def test_executor_persists_admitted_pages_before_a_planning_transport_failure(tmp_path):
    from test_implementation_ir import graph_project

    from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator

    module, main = graph_project(tmp_path)
    original = main.read_bytes()

    class Interrupted(Decisions):
        def generate_tool_decision(self, *args, **kwargs):
            if not self.pages:
                raise ConnectionError("transport stopped")
            return super().generate_tool_decision(*args, **kwargs)

    router = Interrupted([{"nodes": [node(refs=["R1"])], "done": False}])
    generator = CustomModuleGenerator(router)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            generator.generate(tmp_path, module=module)
    assert len(router.calls) == 1
    saved = json.loads(next((tmp_path / ".minecraft_ai/implementation-ir").glob("*.json")).read_text())
    assert saved["compilation"]["nodes"][0]["symbol"] == "PlayerCredits"
    assert saved["compilation"]["page"] == 2
    assert main.read_bytes() == original
    assert not (main.parent / "PlayerCredits.java").exists()


@pytest.mark.parametrize("overrides", [
    {}, {"estimated_tokens": 1600.0}, {"public_api": []}, {"resource_path": "wrong.java"},
    {"activation": True}, {"symbol": "invalid symbol"}, {"requirements": ["R999"]},
    {"kind": "resource", "public_api": [], "resource_path": "src/main/resources/assets/test/lang/en_us.json"},
    {"kind": "resource", "public_api": [], "resource_path": "src/main/resources/assets/other/lang/en_us.json"},
    {"kind": "resource", "public_api": [], "resource_path": "src/main/resources/assets/test/../x.json"},
])
def test_host_and_model_use_one_node_schema(overrides):
    raw = {**node(), **overrides}
    refs = ir.source_requirements(DESIGN)
    schema = ir._page_schema({"requirements": refs, "mod_id": "test"})
    errors = list(Draft202012Validator(schema).iter_errors({"nodes": [raw], "done": True}))
    if errors:
        with pytest.raises(ir.ImplementationGraphError):
            ir.validate_node(raw, package="example", mod_id="test", refs=set(refs))
    else:
        admitted = ir.validate_node(raw, package="example", mod_id="test", refs=set(refs))
        assert admitted["symbol"] == raw["symbol"]


def test_valid_sibling_cannot_drift_through_two_invalid_corrections():
    first, second = node(), node("TradeService")
    invalid_second = {**second, "public_api": []}
    changed_first = {**first, "responsibility": "Different responsibility"}
    router = Decisions([{"nodes": [first, invalid_second], "done": True},
                        {"nodes": [changed_first, invalid_second], "done": True},
                        {"nodes": [changed_first, second], "done": True}])
    with pytest.raises(ir.ImplementationGraphError, match="ACCEPTED_SIBLING_DRIFT"):
        compile_graph(router)
    assert router.calls[2][1]["validation_feedback"]["preserve_nodes"] == [first]


def test_oversized_page_can_be_corrected_into_multiple_pages():
    nodes = [node(f"Owner{i}") for i in range(5)]
    router = Decisions([{"nodes": nodes, "done": True},
                        {"nodes": nodes[:4], "done": False},
                        {"nodes": nodes[4:], "done": True}])
    graph = compile_graph(router)
    assert len(graph["nodes"]) == 5
    assert router.calls[1][1]["validation_feedback"]["preserve_nodes"] == []


def test_semantic_error_can_correct_a_provisional_sibling_after_schema_repair():
    wallet = node(dependencies=["TradeService"])
    trade = node("TradeService", dependencies=["PlayerCredits"])
    invalid = {**trade, "public_api": []}
    router = Decisions([{"nodes": [wallet, invalid], "done": True},
                        {"nodes": [wallet, trade], "done": True},
                        {"nodes": [node(), trade], "done": True}])
    graph = compile_graph(router)
    assert len(graph["nodes"]) == 2
    assert router.calls[1][1]["validation_feedback"]["preserve_nodes"] == [wallet]
    assert router.calls[2][1]["validation_feedback"]["preserve_nodes"] == []


def test_large_authored_design_deterministic_units_and_checkpoint_continuation():
    sections = [
        "behavior_contract", "state_model", "algorithm", "integration",
        "authority_and_network", "persistence", "resources_and_ui",
        "failure_and_limits", "reuse_assessment", "verification"
    ]
    design_lines = []
    for s in sections:
        design_lines.append(f"# {s}")
        for i in range(1, 4):
            design_lines.append(f"Requirement for {s} rule {i}.")
    large_design = "\n".join(design_lines)

    units = ir.decompose_authored_units(large_design)
    assert len(units) >= 10
    assert all(len(u["requirements"]) <= ir.MAX_UNIT_REQUIREMENTS for u in units)

    all_refs = ir.source_requirements(large_design)
    assert len(all_refs) == len(design_lines)

    checkpoints = []
    calls = []

    class MultiUnitRouter(Decisions):
        def generate_tool_decision(self, role, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            calls.append(payload)
            # Ensure payload enforces bounded requirements limit
            assert len(payload["requirements"]) <= ir.MAX_BATCH_REQUIREMENTS
            unit_id = payload["unit_ids"][0] if payload["unit_ids"] else "terminal"
            sym = f"Component{len(calls)}"
            node_inst = {
                "symbol": sym, "kind": "java", "resource_path": "",
                "responsibility": f"Implement {unit_id}",
                "requirements": list(payload["requirements"].keys()),
                "obligations": ["Implement assigned unit behavior."],
                "public_api": [f"public static void exec{len(calls)}()"],
                "depends_on": [f"Component{len(calls) - 1}"] if len(calls) > 1 else [],
                "activation": False, "estimated_tokens": 800,
            }
            # Set done=True only on the last remaining unit
            is_last = len(payload["remaining_unit_ids"]) == 0
            return {"nodes": [node_inst], "done": is_last,
                    "continuation": {"remaining_unit_ids": payload["remaining_unit_ids"]}}

    router = MultiUnitRouter([])
    graph = ir.compile_graph(router, text=large_design, package="example", mod_id="test", target=TARGET,
                             checkpoint=lambda s: checkpoints.append(copy.deepcopy(s)))

    assert len(graph["nodes"]) == len(calls)
    assert len(checkpoints) >= len(units)
    assert set().union(*(set(n["requirements"]) for n in graph["nodes"])) == set(all_refs.keys())

