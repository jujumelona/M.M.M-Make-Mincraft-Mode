"""Model/host graph contract tests, including invalid native tool responses."""
from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from itertools import pairwise
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
    errors = list(Draft202012Validator(ir.PAGE_SCHEMA).iter_errors({"nodes": [invalid]}))
    assert errors, "The model schema must express the host's Java API/path contract"
    assert any(field in str(e) for e in errors)


def test_model_schema_does_not_expose_pagination_controls():
    assert set(ir.PAGE_SCHEMA["properties"]) == {"nodes"}
    legacy = ir._canonicalize_schema_page({
        "nodes": [node()],
        "done": True,
        "continuation": {"remaining_unit_ids": ["legacy"]},
    })
    assert set(legacy) == {"nodes"}


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


def test_premature_done_is_ignored_and_only_active_work_is_sent_next():
    first = node(refs=["R1", "R2", "R3"])
    second = node("TradeService", refs=["R4", "R5", "R6"])
    router = Decisions([{"nodes": [first], "done": True}, {"nodes": [second], "done": True}])
    graph = compile_graph(router)
    assert len(graph["nodes"]) == 2
    next_page = router.calls[1][1]
    assert list(next_page["requirements"]) == ["R4", "R5", "R6"]
    assert "remaining_requirements" not in next_page
    assert "remaining_unit_ids" not in next_page
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


def test_duplicate_owner_preserves_existing_api_contract_without_repair():
    first = node(refs=["R1", "R2", "R3"], api=["public static int balance()"])
    conflicting = node(refs=["R4", "R5", "R6"], api=["public static long balance()"])

    router = Decisions([
        {"nodes": [first], "done": False},
        {"nodes": [conflicting], "done": True},
    ])
    graph = compile_graph(router)

    assert len(router.calls) == 2
    merged = graph["nodes"][0]
    assert merged["public_api"] == ["public static int balance()"]
    assert merged["requirements"] == ["R1", "R2", "R3", "R4", "R5", "R6"]


def test_logged_packet_log_return_type_drift_is_host_frozen_without_repair():
    requirements = {
        "R111": "Persist packet logs.",
        "R112": "Load persisted packet logs.",
        "R113": "Continue authority persistence.",
        "R114": "Continue authority persistence.",
        "R115": "Continue authority persistence.",
        "R116": "Continue authority persistence.",
        "R117": "Continue authority persistence.",
        "R118": "Continue authority persistence.",
    }

    def raw(refs, api):
        return {
            "symbol": "AuthorityAndNetwork_Part2_Persistence",
            "kind": "java",
            "resource_path": "",
            "responsibility": "Own authority and network persistence.",
            "requirements": list(refs),
            "obligations": ["Persist and load packet logs."],
            "public_api": list(api),
            "depends_on": [],
            "activation": False,
            "estimated_tokens": 900,
        }

    accepted = ir.validate_node(
        raw(
            ["R111", "R112"],
            ["public static PacketLog loadPacketLog(String shipId)"],
        ),
        package="example",
        mod_id="test",
        refs=set(requirements),
    )

    combined = ir._admit_graph_page(
        {
            "nodes": [
                raw(
                    ["R113", "R114", "R115", "R116", "R117", "R118"],
                    ["public static void loadPacketLog(String shipId)"],
                )
            ],
            "done": True,
        },
        accepted=[accepted],
        package="example",
        mod_id="test",
        requirements=requirements,
    )

    assert len(combined) == 1
    assert combined[0]["public_api"] == [
        "public static PacketLog loadPacketLog(String shipId)"
    ]
    assert combined[0]["requirements"] == [
        "R111", "R112", "R113", "R114", "R115", "R116", "R117", "R118"
    ]


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


def test_logged_page8_to_page9_duplicate_owner_pattern_merges_r34():
    requirements = {
        "R31": "Enforce resource limits.",
        "R32": "Cheat detection is a non-goal.",
        "R34": "Apply the next behavior-contract requirement.",
    }

    def raw(symbol, refs, api, dependencies=()):
        return {
            "symbol": symbol,
            "kind": "java",
            "resource_path": "",
            "responsibility": f"Own {symbol} behavior.",
            "requirements": list(refs),
            "obligations": [f"Implement {symbol} behavior."],
            "public_api": list(api),
            "depends_on": list(dependencies),
            "activation": False,
            "estimated_tokens": 900,
        }

    accepted_raw = [
        # The old log's pseudo class declaration is now rejected independently.
        # This fixture isolates monotonic owner extension with a legal member.
        raw("BehaviorContractPart6", ["R31", "R32"], ["public static final String PLAYER_KIND"]),
        raw(
            "ActorFactoryPart6",
            ["R31", "R32"],
            ["public static Object createPlayer()"],
            ["BehaviorContractPart6"],
        ),
        raw(
            "ActorRegistryPart6",
            ["R31", "R32"],
            ["public static void registerActor(Object actor)"],
            ["BehaviorContractPart6", "ActorFactoryPart6"],
        ),
        raw(
            "ResourceLimitValidator",
            ["R31"],
            ["public static boolean validateCreditsBalance(int amount)"],
            ["BehaviorContractPart6", "ActorRegistryPart6"],
        ),
    ]
    accepted = [
        ir.validate_node(item, package="example", mod_id="test", refs=set(requirements))
        for item in accepted_raw
    ]

    page9 = [
        raw("BehaviorContractPart6", ["R34"], ["public static final String PLAYER_KIND"]),
        raw(
            "ActorFactoryPart6",
            ["R34"],
            ["public static Object createPlayer()"],
            ["BehaviorContractPart6"],
        ),
        raw(
            "ActorRegistryPart6",
            ["R34"],
            ["public static void registerActor(Object actor)"],
            ["BehaviorContractPart6", "ActorFactoryPart6"],
        ),
        raw(
            "ResourceLimitValidator",
            ["R34"],
            ["public static boolean validateCreditsBalance(int amount)"],
            ["BehaviorContractPart6", "ActorRegistryPart6"],
        ),
    ]

    combined = ir._admit_graph_page(
        {"nodes": page9, "done": True},
        accepted=accepted,
        package="example",
        mod_id="test",
        requirements=requirements,
    )

    assert len(combined) == 4
    by_symbol = {item["symbol"]: item for item in combined}
    assert by_symbol["BehaviorContractPart6"]["requirements"] == ["R31", "R32", "R34"]
    assert by_symbol["ActorFactoryPart6"]["requirements"] == ["R31", "R32", "R34"]
    assert by_symbol["ActorRegistryPart6"]["requirements"] == ["R31", "R32", "R34"]
    assert by_symbol["ResourceLimitValidator"]["requirements"] == ["R31", "R34"]


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
    [
        "mmm/implementation-ir-draft-v1",
        "mmm/implementation-ir-draft-v2",
        "mmm/implementation-ir-draft-v3",
        "mmm/implementation-ir-draft-v4",
        "mmm/implementation-ir-draft-v5",
        "mmm/implementation-ir-draft-v6",
        "mmm/implementation-ir-draft-v7",
        "mmm/implementation-ir-draft-v8",
        "mmm/implementation-ir-draft-v9",
        "mmm/implementation-ir-draft-v10",
    ],
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


def test_different_invalid_responses_stop_when_repair_measure_does_not_improve():
    pages = [{"nodes": [{**node(f"Owner{i}"), "public_api": []}], "done": True} for i in range(4)]
    router = Decisions(pages)
    with pytest.raises(ir.ImplementationGraphError, match="NO_PROGRESS"):
        compile_graph(router)
    assert len(router.calls) == 2


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


def test_output_limit_recursively_bisects_only_the_active_unit():
    from minecraft_mod_ai.llama_finish_reason_contract import (
        OUTPUT_EXHAUSTED,
        LlamaCompletionBoundaryError,
    )

    calls = []

    class ReducingRouter(Decisions):
        def generate_tool_decision(self, role, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            calls.append(payload)
            active_refs = list(payload["requirements"])
            if len(active_refs) > 2:
                raise LlamaCompletionBoundaryError(
                    "output ceiling",
                    kind=OUTPUT_EXHAUSTED,
                    completion_tokens=8192,
                    max_tokens=8192,
                )
            symbol = "PlayerCredits" if "R1" in active_refs else "TradeService"
            dependencies = [] if symbol == "PlayerCredits" else ["PlayerCredits"]
            return {
                "nodes": [node(symbol, refs=active_refs, dependencies=dependencies)],
                "done": False,
            }

    router = ReducingRouter([])
    graph = compile_graph(router)

    assert [len(call["requirements"]) for call in calls] == [6, 3, 1, 2, 3, 1, 2]
    assert all("validation_feedback" not in call for call in calls)
    assert [n["symbol"] for n in graph["nodes"]] == ["PlayerCredits", "TradeService"]
    assert set().union(*(set(n["requirements"]) for n in graph["nodes"])) == {
        "R1", "R2", "R3", "R4", "R5", "R6"
    }


def test_output_limit_shrinks_requirement_window_for_missing_dependency_resolution():
    from minecraft_mod_ai.llama_finish_reason_contract import (
        OUTPUT_EXHAUSTED,
        LlamaCompletionBoundaryError,
    )

    consumer = node("TradeService", dependencies=["MissingService"])
    calls = []

    class DependencyRecovery(Decisions):
        def generate_tool_decision(self, role, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            calls.append(payload)
            if len(calls) == 1:
                return {"nodes": [consumer], "done": False}
            if len(payload["requirements"]) > 1:
                raise LlamaCompletionBoundaryError(
                    "output ceiling",
                    kind=OUTPUT_EXHAUSTED,
                    completion_tokens=8192,
                    max_tokens=8192,
                )
            missing = node(
                "MissingService",
                refs=list(payload["requirements"]),
                api=["public static void provide()"],
            )
            return {"nodes": [missing], "done": True}

    graph = compile_graph(DependencyRecovery([]))

    assert {item["symbol"] for item in graph["nodes"]} == {
        "TradeService",
        "MissingService",
    }
    recovery_sizes = [len(call["requirements"]) for call in calls[1:]]
    assert recovery_sizes[-1] == 1
    assert all(
        later < earlier
        for earlier, later in pairwise(recovery_sizes)
    )


def test_generic_ir_checkpoint_persists_admitted_page_before_transport_failure():
    saved = []

    class Interrupted(Decisions):
        def generate_tool_decision(self, *args, **kwargs):
            if not self.pages:
                raise ConnectionError("transport stopped")
            return super().generate_tool_decision(*args, **kwargs)

    router = Interrupted([{"nodes": [node(refs=["R1"])], "done": False}])
    with pytest.raises(ConnectionError):
        compile_graph(router, checkpoint=lambda state: saved.append(copy.deepcopy(state)))
    assert router.calls
    assert saved[-1]["nodes"][0]["symbol"] == "PlayerCredits"
    assert saved[-1]["page"] == 2

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
    errors = list(Draft202012Validator(schema).iter_errors({"nodes": [raw]}))
    if errors:
        with pytest.raises(ir.ImplementationGraphError):
            ir.validate_node(raw, package="example", mod_id="test", refs=set(refs))
    else:
        admitted = ir.validate_node(raw, package="example", mod_id="test", refs=set(refs))
        assert admitted["symbol"] == raw["symbol"]


def test_valid_sibling_drift_is_ignored_during_schema_correction():
    first, second = node(), node("TradeService")
    invalid_second = {**second, "public_api": []}
    changed_first = {**first, "responsibility": "Different responsibility"}
    router = Decisions([
        {"nodes": [first, invalid_second], "done": True},
        {"nodes": [changed_first, second], "done": True},
    ])

    graph = compile_graph(router)

    by_symbol = {item["symbol"]: item for item in graph["nodes"]}
    assert by_symbol[first["symbol"]]["responsibility"] == first["responsibility"]
    assert by_symbol[second["symbol"]]["public_api"] == second["public_api"]
    assert router.calls[1][1]["validation_feedback"]["preserve_nodes"] == [first]


def test_page_node_count_is_not_rejected_by_an_arbitrary_schema_cap():
    nodes = [node(f"Owner{i}") for i in range(5)]
    router = Decisions([{"nodes": nodes, "done": True}])
    graph = compile_graph(router)
    assert len(graph["nodes"]) == 5
    assert len(router.calls) == 1


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


def test_graph_has_no_legacy_fixed_node_count_limit():
    nodes = [node(f"Owner{index}") for index in range(129)]
    graph = compile_graph(Decisions([{"nodes": nodes, "done": True}]))
    assert len(graph["nodes"]) == len(nodes)


def test_progress_driven_compilation_can_cross_the_legacy_page_count():
    section_count = 40
    lines = []
    for index in range(section_count):
        lines.extend((f"# section_{index}", f"Requirement {index}."))
    design = "\n".join(lines)
    calls = []

    class Router(Decisions):
        def generate_tool_decision(self, role, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            calls.append(payload)
            symbol = f"Component{len(calls)}"
            return {
                "nodes": [{
                    "symbol": symbol,
                    "kind": "java",
                    "resource_path": "",
                    "responsibility": f"Implement {payload['unit_ids'][0]}",
                    "requirements": list(payload["requirements"]),
                    "obligations": ["Implement the active semantic unit."],
                    "public_api": [f"public static void run{len(calls)}()"],
                    "depends_on": [],
                    "activation": False,
                    "estimated_tokens": 800,
                }],
                "done": False,
            }

    graph = ir.compile_graph(
        Router([]),
        text=design,
        package="example",
        mod_id="test",
        target=TARGET,
    )

    assert len(calls) == section_count
    assert len(graph["nodes"]) == section_count
    assert set().union(*(set(item["requirements"]) for item in graph["nodes"])) == set(
        ir.source_requirements(design)
    )


def test_large_authored_design_uses_execution_schema_and_keeps_review_sections_as_context():
    sections = [
        "behavior_contract", "state_model", "algorithm", "integration",
        "authority_and_network", "persistence", "resources_and_ui",
        "failure_and_limits", "reuse_assessment", "verification"
    ]
    design_lines = []
    for section in sections:
        design_lines.append(f"# {section}")
        for index in range(1, 4):
            design_lines.append(f"Requirement for {section} rule {index}.")
    large_design = "\n".join(design_lines)

    units = ir.decompose_authored_units(large_design)
    expected_execution = [
        "state_model", "behavior_contract", "algorithm", "authority_and_network",
        "persistence", "resources_and_ui", "failure_and_limits", "integration",
    ]
    assert [unit["title"] for unit in units] == expected_execution
    assert all(
        "reuse_assessment" not in unit["title"] and "verification" not in unit["title"]
        for unit in units
    )

    checkpoints = []
    calls = []

    class MultiUnitRouter(Decisions):
        def generate_tool_decision(self, role, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            calls.append(payload)
            assert len(payload["unit_ids"]) == 1
            unit_id = payload["unit_ids"][0]
            symbol = f"Component{len(calls)}"
            return {
                "nodes": [{
                    "symbol": symbol,
                    "kind": "java",
                    "resource_path": "",
                    "responsibility": f"Implement {unit_id}",
                    "requirements": list(payload["requirements"]),
                    "obligations": ["Implement assigned unit behavior."],
                    "public_api": [f"public static void exec{len(calls)}()"],
                    "depends_on": [],
                    "activation": False,
                    "estimated_tokens": 800,
                }],
                "done": False,
            }

    graph = ir.compile_authored_graph(
        MultiUnitRouter([]),
        text=large_design,
        package="example",
        mod_id="test",
        target=TARGET,
        checkpoint=lambda state: checkpoints.append(copy.deepcopy(state)),
    )

    execution_refs = set().union(*(set(unit["requirements"]) for unit in units))
    assert len(graph["nodes"]) == len(units)
    assert len(checkpoints) >= len(units)
    assert set().union(*(set(n["requirements"]) for n in graph["nodes"])) == execution_refs


def test_repeated_authored_concern_obligations_merge_source_provenance():
    requirements = {"R1": "first state requirement", "R2": "second state requirement"}
    instruction = json.dumps({
        "concern": "variables",
        "concern_template": "feature/state_model/variables",
        "rules": ["Preserve supplied evidence identifiers."],
        "section": "state_model",
        "section_instruction": "Implement domain state containers.",
        "task": "Resolve exactly one variables record.",
    }, ensure_ascii=False)

    def obligation(requirement_id):
        return json.dumps({
            "instruction": instruction,
            "source_requirements": {
                requirement_id: requirements[requirement_id],
            },
        }, ensure_ascii=False)

    first = node(refs=["R1"])
    first["obligations"] = [obligation("R1")]
    accepted = ir.validate_node(
        first, package="example", mod_id="test", refs=set(requirements)
    )

    second = node(refs=["R2"])
    second["obligations"] = [obligation("R2")]
    combined = ir._admit_graph_page(
        {"nodes": [second]},
        accepted=[accepted],
        package="example",
        mod_id="test",
        requirements=requirements,
    )

    assert len(combined) == 1
    assert combined[0]["requirements"] == ["R1", "R2"]
    assert len(combined[0]["obligations"]) == 1
    payload = json.loads(combined[0]["obligations"][0])
    assert payload["source_requirements"] == requirements
