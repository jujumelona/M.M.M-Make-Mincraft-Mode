from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import custom_module_generator as direct
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
from minecraft_mod_ai.model_adapters.base import NativeToolDecisionRejected

TARGET = {"minecraft_version": "1.21.1", "loader": "fabric", "mappings": "1.21.1+build.3"}
DESIGN = "# behavior_contract\nAdd credits and purchase once.\n# state_model\nPlayerCredits owns balances.\n# verification\nReject purchases when balance is insufficient."


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


def test_native_schema_rejection_canonicalizes_declaration_body_without_retry():
    bad = node("ActorState", api=["public enum ActorState {IDLE, ACTIVE, INACTIVE, DESTROYED}"])

    class RejectOnce:
        def __init__(self):
            self.calls = []

        def generate_tool_decision(self, role, messages, **kwargs):
            self.calls.append((kwargs["tool_name"], json.loads(messages[-1]["content"])))
            raise NativeToolDecisionRejected(
                kwargs["tool_name"],
                [{
                    "original_tool": kwargs["tool_name"],
                    "raw_arguments": json.dumps({"nodes": [bad], "done": True}),
                    "failure_code": "TOOL_DECISION_SCHEMA_INVALID",
                    "error": "public_api declaration contains a body",
                }],
            )

    router = RejectOnce()
    graph = compile_with(router)
    assert len(router.calls) == 1
    assert graph["nodes"][0]["symbol"] == "ActorState"
    assert graph["nodes"][0]["public_api"] == ["public enum ActorState"]


def test_schema_repair_freezes_valid_siblings_and_merges_only_invalid_nodes():
    stable = node("BehaviorContract", api=["public static int original()"])
    invalid = node("ActorState")
    invalid["public_api"] = []
    drifted = copy.deepcopy(stable)
    drifted["public_api"] = ["public static int drifted()"]
    fixed = copy.deepcopy(invalid)
    fixed["public_api"] = ["public enum ActorState"]

    router = Decisions([
        {"nodes": [stable, invalid], "done": True},
        # A small model may redundantly rewrite the valid sibling. The host must
        # ignore that rewrite and merge only the corrected invalid node.
        {"nodes": [drifted, fixed], "done": True},
    ])
    graph = compile_with(router)
    by_symbol = {item["symbol"]: item for item in graph["nodes"]}
    assert by_symbol["BehaviorContract"]["public_api"] == ["public static int original()"]
    assert by_symbol["ActorState"]["public_api"] == ["public enum ActorState"]
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


def test_pagination_has_explicit_completion_and_detects_repetition():
    router = Decisions([{"nodes": [node()], "done": False},
                        {"nodes": [node()], "done": False},
                        {"nodes": [node()], "done": False}])
    with pytest.raises(ImplementationGraphError, match="DUPLICATE_OWNER"):
        compile_with(router)
    assert len(router.calls) == 3


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


def test_real_java_execution_after_budget_decomposition_and_api_handoff(tmp_path, monkeypatch):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("Java compiler/runtime unavailable")
    module, main = graph_project(tmp_path)
    wallet = node(cost=3000, api=["public static int balance()", "public static boolean spend(int amount)"])
    helper = node("PlayerCreditsPartStore", cost=1000, api=wallet["public_api"])
    facade = copy.deepcopy(wallet)
    facade.update(estimated_tokens=1100, depends_on=[helper["symbol"]])
    facade["obligations"] = ["Delegate storage and transactions to PlayerCreditsPartStore through the frozen API."]
    trade = node("TradeService", dependencies=["PlayerCredits"], api=["public static void initialize()"], activation=True)
    router = Decisions([{"nodes": [wallet, trade], "done": True},
                        {"nodes": [helper, facade], "done": True}])
    decodes = []
    sources = {
        helper["symbol"]: "public static int balance() { return credits; } private static int credits=10; public static boolean spend(int amount) { if(amount<0 || credits<amount) return false; credits-=amount; return true; }",
        "PlayerCredits": "public static int balance() { return PlayerCreditsPartStore.balance(); } public static boolean spend(int amount) { return PlayerCreditsPartStore.spend(amount); }",
        "TradeService": 'public static void initialize() { if(!PlayerCredits.spend(3) || PlayerCredits.balance()!=7 || PlayerCredits.spend(8) || PlayerCredits.balance()!=7) throw new AssertionError("transaction"); System.out.print("PASS"); }',
    }

    def generate_text(role, messages, **kwargs):
        prompt = messages[-1]["content"]
        symbol = next(s for s in sources if f"Exact target: src/main/java/example/{s}.java#{s}" in prompt)
        decodes.append(symbol)
        if decodes == ["PlayerCredits"]:
            raise LlamaCompletionBoundaryError("limit", kind=OUTPUT_EXHAUSTED, completion_tokens=4096, max_tokens=4096)
        return json.dumps({"content": f"package example;\npublic final class {symbol} {{ {sources[symbol]} }}", "summary": "implemented"})

    router.generate_text = generate_text
    compilations = []

    class JavacRunner:
        def __init__(self, *args):
            pass

        def compile_java(self, root):
            files = list((Path(root) / "src/main/java").rglob("*.java"))
            result = subprocess.run([javac, "-d", str(tmp_path / "classes"), *map(str, files)], capture_output=True, text=True, check=False)
            compilations.append(result)
            return SimpleNamespace(status="PASS" if result.returncode == 0 else "FAIL", commands=(), error=result.stderr)

    monkeypatch.setattr(direct, "GradleRunner", JavacRunner)
    monkeypatch.setattr(direct, "adapter_for_target", lambda *args: SimpleNamespace(minecraft_version="1.21.1", loader="fabric", java_version=17, yarn_mappings="none"))
    generator = direct.CustomModuleGenerator(router)
    result = generator.generate(tmp_path, module=module)
    assert decodes == ["PlayerCredits", "PlayerCreditsPartStore", "PlayerCredits", "TradeService"]
    assert len(router.calls) == 2
    assert result["decomposition_count"] == 1
    assert all(c.returncode == 0 for c in compilations)
    assert "TradeService.initialize();" in main.read_text()
    assert "PlayerCredits.initialize" not in main.read_text()
    assert generator.ensure_generation_live_commit(result, project_root=tmp_path)
    probe = tmp_path / "Probe.java"
    probe.write_text("public class Probe { public static void main(String[] args) { new example.TestMod().onInitialize(); } }", encoding="utf-8")
    subprocess.run([javac, "-cp", str(tmp_path / "classes"), "-d", str(tmp_path / "classes"), str(probe)], check=True, capture_output=True)
    run = subprocess.run([java, "-cp", str(tmp_path / "classes"), "Probe"], capture_output=True, text=True, check=True)
    assert run.stdout == "PASS"
    (tmp_path / result["touched_paths"][0]).write_text("tampered", encoding="utf-8")
    assert not generator.ensure_generation_live_commit(result, project_root=tmp_path)


def test_failed_decomposition_rolls_back_and_resume_does_not_repeat_exhausted_decode(tmp_path, monkeypatch):
    module, main = graph_project(tmp_path)
    before = main.read_bytes()
    oversized = node(cost=3000)
    # Same task is not a split; both first run and resumed run fail before another decode.
    pages = [{"nodes": [oversized], "done": True},
             {"nodes": [oversized], "done": True},
             {"nodes": [oversized], "done": True}]
    router = Decisions(pages)
    calls = []

    def generate_text(*args, **kwargs):
        calls.append(args)
        raise LlamaCompletionBoundaryError("limit", kind=OUTPUT_EXHAUSTED, completion_tokens=4096, max_tokens=4096)

    router.generate_text = generate_text
    monkeypatch.setattr(direct, "adapter_for_target", lambda *args: SimpleNamespace(minecraft_version="1.21.1", loader="fabric", java_version=17, yarn_mappings="none"))
    generator = direct.CustomModuleGenerator(router)
    for _ in range(2):
        with pytest.raises(ImplementationGraphError, match="DECOMPOSITION_REQUIRED"):
            generator.generate(tmp_path, module=module)
        assert main.read_bytes() == before
        assert not (main.parent / "PlayerCredits.java").exists()
    assert len(calls) == 1
    assert [n for n, _ in router.calls] == ["compile_implementation_graph", "decompose_implementation_node", "decompose_implementation_node"]


def test_resources_are_separate_artifacts_and_final_compile_failure_rolls_back(tmp_path, monkeypatch):
    module, main = graph_project(tmp_path)
    before = main.read_bytes()
    resource = node("Translations")
    resource.update(kind="resource", public_api=[], resource_path="src/main/resources/assets/test/lang/en_us.json")
    router = Decisions([{"nodes": [resource], "done": True}])
    router.generate_text = lambda *args, **kwargs: json.dumps({"content": '{"item.test.token":"Token"}', "summary": "resource"})
    outcomes = ["FAIL", "PASS"]

    class Runner:
        def __init__(self, *args):
            pass

        def compile_java(self, *args):
            return SimpleNamespace(status=outcomes.pop(0), commands=(), error="integration failed")

    monkeypatch.setattr(direct, "GradleRunner", Runner)
    generator = direct.CustomModuleGenerator(router)
    with pytest.raises(ImplementationGraphError, match="INTEGRATION_COMPILE_FAILED"):
        generator.generate(tmp_path, module=module)
    assert main.read_bytes() == before
    assert not (tmp_path / resource["resource_path"]).exists()
    result = generator.generate(tmp_path, module=module)
    assert json.loads((tmp_path / resource["resource_path"]).read_text()) == {"item.test.token": "Token"}
    assert result["operation_count"] == 2
    assert generator.ensure_generation_live_commit(result, project_root=tmp_path)


def test_relabeling_estimate_without_moving_work_is_not_decomposition():
    original = node(cost=8000)
    facade = node(cost=1000, dependencies=["PlayerCreditsPartStore"])
    helper = node("PlayerCreditsPartStore", cost=1000)
    router = Decisions([{"nodes": [original], "done": True},
                        {"nodes": [facade, helper], "done": True},
                        {"nodes": [facade, helper], "done": True}])
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
    with pytest.raises(ImplementationGraphError, match="UNUSED_SPLIT_HELPER"):
        compile_with(router)
