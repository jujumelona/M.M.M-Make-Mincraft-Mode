from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import custom_module_generator as direct
from minecraft_mod_ai import implementation_ir as ir
from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules
from minecraft_mod_ai.implementation_ir import (
    ImplementationGraphError,
    OutputBudgetExhausted,
    compile_graph,
    source_requirements,
)
from minecraft_mod_ai.llama_finish_reason_contract import (
    OUTPUT_EXHAUSTED,
    LlamaCompletionBoundaryError,
)

TARGET = {"minecraft_version": "1.21.1", "loader": "fabric", "mappings": "1.21.1+build.3"}
DESIGN = "# implementation\nAdd credits and purchase once.\nPlayerCredits owns balances.\nReject purchases when balance is insufficient.\nPersist the balance.\nExpose the balance API."


def node(symbol="PlayerCredits", *, refs=None, cost=1600, dependencies=(), api=None, activation=False):
    return {"symbol": symbol, "kind": "java", "resource_path": "",
            "responsibility": "Own the player balance" if symbol == "PlayerCredits" else symbol,
            "requirements": refs or list(source_requirements(DESIGN)),
            "obligations": ["Keep state in its declared owner and implement the assigned behavior."],
            "public_api": api or ["public static int balance()"], "depends_on": list(dependencies),
            "activation": activation, "estimated_tokens": cost}


class Decisions:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def generate_tool_decision(self, role, messages, **kwargs):
        self.calls.append((kwargs["tool_name"], json.loads(messages[-1]["content"])))
        return copy.deepcopy(self.pages.pop(0))

    def generate_implementation_decision(self, name, payload, *, state=None, checkpoint=None):
        del state, checkpoint
        return self.generate_tool_decision(
            "planner",
            [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            tool_name=name,
            parameters=ir._page_schema(payload),
        )


def compile_with(router):
    return compile_graph(router, text=DESIGN, package="example", mod_id="test", target=TARGET)


def test_graph_combines_sections_by_owner_and_orders_actual_dependencies():
    trade = node("TradeService", dependencies=["PlayerCredits"], refs=["R2", "R6"])
    wallet = node(refs=["R1", "R3", "R4", "R5"])
    router = Decisions([{"nodes": [trade, wallet], "done": True}])
    graph = compile_with(router)
    assert [n["symbol"] for n in graph["nodes"]] == ["PlayerCredits", "TradeService"]
    assert graph["source_text"] == DESIGN
    assert len(graph["nodes"]) == 2  # Three headings do not imply three files.
    assert graph["nodes"][0]["depends_on"] == []


def test_schema_canonicalization_strips_member_body_without_decode_retry():
    bad = node("PlayerCredits", api=["public static int balance() { return 1; }"])
    payload = {"requirements": source_requirements(DESIGN), "mod_id": "test"}
    normalized = ir._canonicalize_schema_page({"nodes": [bad]}, payload)
    assert normalized["nodes"][0]["public_api"] == ["public static int balance()"]
    assert not ir._schema_diagnostics(normalized, ir._page_schema(payload))


def test_schema_repair_scope_freezes_valid_sibling_and_replaces_only_invalid_node():
    stable = node("BehaviorContract", api=["public static int original()"])
    invalid = node("ActorState", api=[
        "public enum ActorState {IDLE, ACTIVE, INACTIVE, DESTROYED}"
    ])
    payload = {"requirements": source_requirements(DESIGN), "mod_id": "test"}
    page = {"nodes": [stable, invalid]}
    schema = ir._page_schema(payload)
    diagnostics = ir._schema_diagnostics(page, schema)
    preserve, repair_count = ir._schema_repair_scope(page, schema, diagnostics)
    assert preserve == [stable]
    assert repair_count == 1

    fixed = copy.deepcopy(invalid)
    fixed["public_api"] = [
        "public static final ActorState IDLE",
        "public static final ActorState ACTIVE",
        "public static final ActorState INACTIVE",
        "public static final ActorState DESTROYED",
    ]
    drifted = copy.deepcopy(stable)
    drifted["public_api"] = ["public static int drifted()"]
    merged = ir._merge_scoped_schema_repair(
        {"nodes": [drifted, fixed]},
        {"preserve_nodes": preserve, "repair_count": repair_count},
    )
    assert merged["nodes"][0] == stable
    assert merged["nodes"][1] == fixed

def test_schema_repair_freezes_valid_siblings_and_merges_only_invalid_nodes():
    stable = node("BehaviorContract", api=["public static int original()"])
    invalid = node("ActorState")
    invalid["public_api"] = []
    drifted = copy.deepcopy(stable)
    drifted["public_api"] = ["public static int drifted()"]
    fixed = copy.deepcopy(invalid)
    fixed["public_api"] = ["public static final ActorState IDLE"]

    router = Decisions([
        {"nodes": [stable, invalid], "done": True},
        # A small model may redundantly rewrite the valid sibling. The host must
        # ignore that rewrite and merge only the corrected invalid node.
        {"nodes": [drifted, fixed], "done": True},
    ])
    graph = compile_with(router)
    by_symbol = {item["symbol"]: item for item in graph["nodes"]}
    assert by_symbol["BehaviorContract"]["public_api"] == ["public static int original()"]
    assert by_symbol["ActorState"]["public_api"] == ["public static final ActorState IDLE"]
    assert len(router.calls) == 2
    feedback = router.calls[1][1]["validation_feedback"]
    assert feedback["preserve_nodes"] == [stable]
    assert feedback["repair_count"] == 1

def test_oversized_task_decomposes_before_any_coder_request():
    oversized = node(cost=8000)
    helper = node("PlayerCreditsPartStore", cost=1800)
    facade = node(cost=1200, dependencies=[helper["symbol"]])
    facade["obligations"] = ["Delegate storage to PlayerCreditsPartStore while retaining the balance API."]
    router = Decisions([{"nodes": [oversized], "done": True},
                        {"nodes": [helper, facade], "done": True}])
    router.implementation_output_budget = 4000
    graph = compile_with(router)
    assert [name for name, _ in router.calls] == ["compile_implementation_graph", "decompose_implementation_node"]
    assert len(graph["nodes"]) == 2
    assert router.calls[1][1]["reason"] == "preflight_output_budget"


def test_preflight_refines_multiple_levels_without_decoding_oversized_intermediate():
    parent = node(cost=16000)
    helper = node("PlayerCreditsPartStore", cost=6000)
    facade = node(cost=1200, dependencies=[helper["symbol"]])
    facade["obligations"] = ["Delegate balance storage through the helper API."]
    inner = node("PlayerCreditsPartStorePartBalances", cost=1600)
    helper_facade = copy.deepcopy(helper)
    helper_facade.update(estimated_tokens=1200, depends_on=[inner["symbol"]],
                         obligations=["Delegate storage to the smaller balance owner."])
    router = Decisions([{"nodes": [parent], "done": True},
                        {"nodes": [facade, helper], "done": True},
                        {"nodes": [helper_facade, inner], "done": True}])
    router.implementation_output_budget = 4000
    graph = compile_with(router)
    assert [n["symbol"] for n in graph["nodes"]] == [inner["symbol"], helper["symbol"], parent["symbol"]]
    assert len(router.calls) == 3


@pytest.mark.parametrize("fault", ["cycle", "unknown", "coverage", "path", "duplicate", "api"])
def test_invalid_graph_is_rejected_before_source_generation(fault):
    first, second = node(), node("TradeService")
    if fault == "cycle":
        first["depends_on"], second["depends_on"] = ["TradeService"], ["PlayerCredits"]
    elif fault == "unknown":
        first["depends_on"] = ["Missing"]
    elif fault == "coverage":
        first["requirements"] = second["requirements"] = ["R1"]
    elif fault == "path":
        first.update(kind="resource", resource_path="src/main/resources/assets/test/../../other/x.json", public_api=[])
    elif fault == "duplicate":
        second = first
    else:
        first["public_api"] = []
    with pytest.raises(ImplementationGraphError):
        compile_with(Decisions([{"nodes": [first, second], "done": True}] * 3))


def test_pagination_termination_is_host_owned_and_rejects_no_progress():
    consumer = node(dependencies=["MissingService"])
    repeated = copy.deepcopy(consumer)
    router = Decisions([
        {"nodes": [consumer], "done": False},
        {"nodes": [repeated], "done": False},
    ])
    with pytest.raises(ImplementationGraphError, match="PAGE_NO_PROGRESS"):
        compile_with(router)
    assert len(router.calls) == 2


@pytest.mark.parametrize("fault", ["same", "api", "lost", "helper"])
def test_refinement_cannot_repeat_task_drop_coverage_or_break_consumers(fault):
    original = node(cost=8000)
    helper = node("PlayerCreditsPartStore", cost=1200)
    facade = node(cost=1000, dependencies=[helper["symbol"]])
    if fault == "same":
        facade = original
    elif fault == "api":
        facade["public_api"] = ["public static long balance()"]
    elif fault == "lost":
        facade["requirements"] = helper["requirements"] = ["R1"]
    else:
        helper["symbol"] = "Unrelated"
    router = Decisions([{"nodes": [original], "done": True},
                        {"nodes": [helper, facade], "done": True},
                        {"nodes": [helper, facade], "done": True}])
    router.implementation_output_budget = 4000
    with pytest.raises(ImplementationGraphError):
        compile_with(router)


def test_wrapped_output_boundary_propagates_without_compiler_repair(tmp_path, monkeypatch):
    from test_direct_custom_module_generator import _adapter, _module, _project
    root, path, symbol = _project(tmp_path)
    original = (root / path).read_bytes()
    calls = []

    class Router:
        def generate_text(self, *args, **kwargs):
            calls.append(args)
            try:
                raise LlamaCompletionBoundaryError("limit", kind=OUTPUT_EXHAUSTED,
                                                   completion_tokens=4096, max_tokens=4096)
            except LlamaCompletionBoundaryError as exc:
                raise RuntimeError("wrapped backend error") from exc

    class NoCompiler:
        def __init__(self, *args):
            pass

        def compile_java(self, *args):
            pytest.fail("Output exhaustion is not a compiler error")

    monkeypatch.setattr(direct, "adapter_for_target", lambda *args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", NoCompiler)
    with pytest.raises(OutputBudgetExhausted):
        direct.CustomModuleGenerator(Router()).generate(root, module=_module(path, symbol), minecraft_version="1.21.1", loader="fabric")
    assert len(calls) == 1
    assert (root / path).read_bytes() == original


def graph_project(tmp_path):
    modules, _ = _compile_new_authored_modules(AuthoredPlan("economy", DESIGN),
                                              mod_id="test", package_name="example", target=TARGET)
    main = tmp_path / "src/main/java/example/TestMod.java"
    main.parent.mkdir(parents=True)
    main.write_text("package example; public final class TestMod { public void onInitialize() {} }", encoding="utf-8")
    return modules[0], main


def test_relabeling_estimate_without_moving_work_is_not_decomposition():
    original = node(cost=8000)
    facade = node(cost=1000, dependencies=["PlayerCreditsPartStore"])
    helper = node("PlayerCreditsPartStore", cost=1000)
    router = Decisions([{"nodes": [original], "done": True},
                        {"nodes": [facade, helper], "done": True},
                        {"nodes": [facade, helper], "done": True}])
    router.implementation_output_budget = 4000
    with pytest.raises(ImplementationGraphError, match="UNCHANGED_WORK"):
        compile_with(router)


def test_unused_helper_cannot_disguise_same_whole_file_work():
    original = node(cost=8000)
    facade = node(cost=1000)
    facade["obligations"] = ["Delegate to a helper."]
    helper = node("PlayerCreditsPartStore", cost=1000)
    router = Decisions([{"nodes": [original], "done": True},
                        {"nodes": [facade, helper], "done": True},
                        {"nodes": [facade, helper], "done": True}])
    router.implementation_output_budget = 4000
    with pytest.raises(ImplementationGraphError, match="UNUSED_SPLIT_HELPER"):
        compile_with(router)
