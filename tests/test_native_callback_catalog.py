"""Native writes must survive the actual index-to-callback production path."""
import hashlib
import json
import shutil
import subprocess

import pytest

from minecraft_mod_ai.construction_file_api_evidence import FileAPIEvidence
from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java
from minecraft_mod_ai.host_callback_body import author_callback_body
from minecraft_mod_ai.host_verified_callback_ops import verified_runtime_mutation_candidates
from minecraft_mod_ai.runtime_mutation_author import verify_native_runtime_mutation_coverage


def _symbols():
    owner = "example.Target"
    hook = "net.fabricmc.fabric.api.event.player.TestHook"
    event = "net.fabricmc.fabric.api.event.Event"
    symbols = {
        owner: {"kind": "CLASS", "owner": owner, "access": 1},
        hook: {"kind": "CLASS", "owner": hook, "access": 1537},
        "hook": {"kind": "FIELD", "owner": hook, "name": "EVENT", "access": 25,
                 "return_type": event, "generic_return_type": event + "<" + hook + ">"},
        "sam": {"kind": "METHOD", "owner": hook, "name": "run", "access": 1025,
                "parameters": [owner], "return_type": "void"},
        "register": {"kind": "METHOD", "owner": event, "name": "register", "access": 1,
                     "parameters": ["java.lang.Object"], "return_type": "void"},
        "ready": {"kind": "METHOD", "owner": owner, "name": "isReady", "access": 1,
                  "parameters": [], "return_type": "boolean", "static": False},
    }
    for i in range(90):
        symbols[f"noise{i}"] = {
            "kind": "METHOD", "owner": owner, "name": f"applyNoise{i:03d}", "access": 1,
            "parameters": ["float"], "return_type": "void", "static": False,
        }
    symbols["write"] = {
        "kind": "METHOD", "owner": owner, "name": "setHealth", "access": 1,
        "parameters": ["float"], "return_type": "void", "static": False,
    }
    return symbols


def _frame(targets=("Player health",)):
    return {
        "kind": "method", "prefix": "private boolean apply(example.Target target) {\n",
        "suffix": "\n}", "host_runtime_mutation_receipts": [],
        "host_behavior_contract": {"behavior_id": "native", "runtime_state_writes": list(targets)},
    }


class SelectingRouter:
    """Only the external model boundary is replaced; discovery/lowering are real."""
    def __init__(self, method="setHealth"):
        self.method = method
        self.pages = []
        self.prompts = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.prompts.append(messages)
        page = json.loads(messages[-1]["content"])
        if data_name == "select_verified_fabric_event":
            return {"event_binding": "net.fabricmc.fabric.api.event.player.TestHook#EVENT"}
        if "_predicate_" in data_name:
            return {"verified_call_id": parameters["properties"]["verified_call_id"]["enum"][0]}
        if "_operation_" in data_name:
            return {"action": "complete"}
        if "_runtime_mutation_" in data_name:
            self.pages.append(page)
            choice = next((c for c in page["inspected_calls"]
                           if "#" + self.method + "(" in c["signature"]), None)
            return {"verified_call_id": choice["verified_call_id"] if choice else "unavailable"}
        if "_native_value_" in data_name:
            return {"expression": "6.0f"}
        raise AssertionError(data_name)


def test_first_pass_keeps_scalar_mutator_from_actual_full_symbol_index(tmp_path):
    router = SelectingRouter()
    behavior = {"behavior_id": "native", "intent": "Change player health when ready",
                "trigger": "target ready", "effects": "Change player health",
                "runtime_state_writes": ["Player health"]}
    receipts = []
    source = compose_first_pass_java(router, {
        "language": "java", "side": "COMMON", "target_path": "src/main/java/example/Bridge.java",
        "bindings": {"request_feature_entrypoint": "example.Bridge"},
    }, [], construction_context={"symbols": _symbols(), "runtime_mutation_receipts": receipts},
        behaviors=[behavior])
    assert "target.setHealth(6.0f);" in source
    verify_native_runtime_mutation_coverage(source, [behavior], receipts)
    # The complete host catalogue must not be serialized into every model call.
    for messages in router.prompts:
        assert "host_verified_runtime_mutation_candidates" not in json.dumps(messages)

    java = shutil.which("java")
    javac = shutil.which("javac")
    if not java:
        pytest.skip("Java runtime required for compiled gameplay probe")
    compiler = [javac] if javac else [java, "com.sun.tools.javac.Main"]
    files = {
        "example/Bridge.java": source,
        "example/Target.java": "package example; public class Target { public boolean ready; public float health=20; public boolean isReady(){return ready;} public void setHealth(float value){health=value;} }",
        "net/fabricmc/api/ModInitializer.java": "package net.fabricmc.api; public interface ModInitializer {void onInitialize();}",
        "net/fabricmc/fabric/api/event/Event.java": "package net.fabricmc.fabric.api.event; public class Event<T> {public T listener; public void register(T value){listener=value;}}",
        "net/fabricmc/fabric/api/event/player/TestHook.java": "package net.fabricmc.fabric.api.event.player; public interface TestHook {net.fabricmc.fabric.api.event.Event<TestHook> EVENT=new net.fabricmc.fabric.api.event.Event<>(); void run(example.Target target);}",
        "example/Probe.java": "package example; import net.fabricmc.fabric.api.event.player.TestHook; public class Probe {public static void main(String[] args){new Bridge().onInitialize(); Target t=new Target(); TestHook.EVENT.listener.run(t); if(t.health!=20)throw new AssertionError(); t.ready=true; TestHook.EVENT.listener.run(t); if(t.health!=6)throw new AssertionError();}}",
    }
    paths = []
    for path, text in files.items():
        output = tmp_path / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        paths.append(str(output))
    classes = tmp_path / "classes"
    result = subprocess.run([*compiler, "-proc:none", "-d", str(classes), *paths],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([java, "-cp", str(classes), "example.Probe"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_native_selection_searches_past_old_candidate_and_page_cutoffs():
    # No lexical match to the obligation: relevance cannot establish absence.
    methods = [{"kind": "METHOD", "name": f"applyNoise{i:03d}", "static": False,
                "parameters": ["float"], "return_type": "void"} for i in range(90)]
    methods.append({"kind": "METHOD", "name": "zeta", "static": False,
                    "parameters": ["float"], "return_type": "void"})
    message = [{"role": "user", "content": json.dumps({
        "verified_event_callback": {"parameters": ["example.Target"]},
        "callback_input_api": [{"verified_class": "example.Target", "members": methods}],
    })}]
    router = SelectingRouter("zeta")
    result = author_callback_body(router, message, data_name="paged", frame=_frame(),
                                  effect_receipt=True)
    assert "target.zeta(6.0f);" in result
    assert len(router.pages) >= 3
    assert all(len(p["inspected_calls"]) <= 32 for p in router.pages)


def test_one_checked_invocation_can_cover_two_explicit_native_obligations():
    message = [{"role": "user", "content": json.dumps({
        "verified_event_callback": {"parameters": ["example.Target"]},
        "callback_input_api": [{"verified_class": "example.Target", "members": [{
            "kind": "METHOD", "name": "setHealth", "static": False,
            "parameters": ["float"], "return_type": "void",
        }]}],
    })}]
    frame = _frame(("Combat damage", "Player health"))
    result = author_callback_body(SelectingRouter(), message, data_name="shared", frame=frame,
                                  effect_receipt=True)
    assert result.count("target.setHealth(6.0f);") == 1
    assert {r["state_name"] for r in frame["host_runtime_mutation_receipts"]} == {
        "Combat damage", "Player health"}


def test_same_mutation_abi_in_independent_handlers_keeps_per_behavior_proof():
    behaviors, receipts, methods = [], [], []
    for identity in ("first", "second"):
        method = "applyBehavior" + hashlib.sha256(identity.encode()).hexdigest()[:20]
        behaviors.append({"behavior_id": identity, "runtime_state_writes": ["health"]})
        receipts.append({"behavior_id": identity, "state_name": "health",
                         "verified_signature": "example.Target#setHealth(float)",
                         "statement": "target.setHealth(6.0f);"})
        methods.append("boolean " + method + "(Target target){target.setHealth(6.0f);return true;}")
    source = "class Bridge {" + "".join(methods) + "}"
    verify_native_runtime_mutation_coverage(source, behaviors, receipts)
    with pytest.raises(ValueError, match="NATIVE_MUTATION_PROOF_EFFECT_DROPPED"):
        verify_native_runtime_mutation_coverage(
            source.replace("target.setHealth(6.0f);", "", 1), behaviors, receipts)


def test_nested_scalar_catalog_preserves_each_verified_receiver_identity():
    from minecraft_mod_ai.host_runtime_mutation_catalog import runtime_mutation_api_records

    player, bag = "example.Player", "example.Bag"
    symbols = {t: {"kind": "CLASS", "owner": t, "access": 1} for t in (player, bag)}
    for name in ("getMainBag", "getSpareBag"):
        symbols[name] = {"kind": "METHOD", "owner": player, "name": name, "access": 1,
                         "parameters": [], "return_type": bag, "static": False}
    symbols["write"] = {"kind": "METHOD", "owner": bag, "name": "setCount", "access": 1,
                         "parameters": ["int"], "return_type": "void", "static": False}
    symbols["hidden"] = {**symbols["write"], "name": "privateWrite", "access": 2}
    symbols["client"] = {**symbols["write"], "name": "clientWrite", "side": "CLIENT"}
    evidence = FileAPIEvidence(symbols, target_path="src/main/java/example/Bridge.java",
                               owner="example.Bridge")
    records = runtime_mutation_api_records(evidence, [player])
    message = [{"role": "user", "content": json.dumps({
        "verified_event_callback": {"parameters": [player]}, "callback_input_api": records,
    })}]
    candidates = verified_runtime_mutation_candidates(message, {
        "prefix": "private void run(example.Player player) {\n", "suffix": "\n}"})
    assert {(c["receiver"], c["name"]) for c in candidates} == {
        ("player.getMainBag()", "setCount"), ("player.getSpareBag()", "setCount")}


def test_compiler_repair_updates_only_the_repaired_native_effect_proof():
    from minecraft_mod_ai.construction_host_member_repair import repair_compiler_member
    from minecraft_mod_ai.runtime_mutation_author import RuntimeRepairBindings

    identity = "native"
    method = "applyBehavior" + hashlib.sha256(identity.encode()).hexdigest()[:20]
    source = ("package example; class Bridge {\nprivate boolean " + method +
              "(example.Target target) {\ntarget.setHealth(6.0); return true;\n}\n}")
    behaviors = [{"behavior_id": identity, "effects": "Change player health",
                  "runtime_state_writes": ["Player health"]}]
    receipts = [{"behavior_id": identity, "state_name": "Player health",
                 "verified_signature": "example.Target#setHealth(float)",
                 "statement": "target.setHealth(6.0);"}]
    revised = repair_compiler_member(
        SelectingRouter(), source,
        "Bridge.java:3: error: incompatible types: possible lossy conversion from double to float",
        filename="Bridge.java", target_path="src/main/java/example/Bridge.java",
        symbols=_symbols(), owner="example.Bridge", api_evidence={}, prerequisite_api=[],
        behaviors=behaviors, runtime_state_bindings=RuntimeRepairBindings({}, receipts), attempt=1,
    )
    assert "target.setHealth(6.0f);" in revised
    verify_native_runtime_mutation_coverage(revised, behaviors, receipts)
    assert len(receipts) == 1
    assert receipts[0]["statement"] == "target.setHealth(6.0f);"


@pytest.mark.parametrize("self_type", [False, True])
def test_root_typed_getter_keeps_one_hop_participant_without_recursive_expansion(self_type):
    from minecraft_mod_ai.host_runtime_mutation_catalog import runtime_mutation_api_records

    actor = "example.Actor"
    target = actor if self_type else "example.Target"
    symbols = {t: {"kind": "CLASS", "owner": t, "access": 1} for t in (actor, target)}
    symbols["getter"] = {"kind": "METHOD", "owner": actor, "name": "getOwnedTarget",
                         "access": 1, "parameters": [], "return_type": target, "static": False}
    symbols["write"] = {"kind": "METHOD", "owner": target, "name": "setHealth",
                        "access": 1, "parameters": ["float"], "return_type": "void", "static": False}
    parameters = [(actor, "actor")] if self_type else [(actor, "actor"), (target, "target")]
    evidence = FileAPIEvidence(symbols, target_path="src/main/java/example/Bridge.java",
                               owner="example.Bridge")
    task = {"verified_event_callback": {"parameters": [typ for typ, _ in parameters]},
            "declaration_metadata": {"parameters": [{"type": typ, "name": name}
                                                        for typ, name in parameters]},
            "callback_input_api": runtime_mutation_api_records(evidence, [typ for typ, _ in parameters])}
    frame = {"prefix": "private void run(" + ", ".join(typ + " " + name for typ, name in parameters) + ") {\n",
             "suffix": "\n}"}
    candidates = verified_runtime_mutation_candidates([], frame, api_task=task)
    expected = {"actor", "actor.getOwnedTarget()"} if self_type else {
        "target", "actor.getOwnedTarget()"}
    assert {c["receiver"] for c in candidates} == expected
