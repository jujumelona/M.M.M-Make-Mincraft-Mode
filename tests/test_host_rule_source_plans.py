"""Exercise rule sources with the same source index used in production."""
import hashlib

import pytest

from minecraft_mod_ai.construction_context import java_source_units
from minecraft_mod_ai.construction_source_symbols import inspected_project_source_symbols
from minecraft_mod_ai.host_rule_program import RuleProgram
from minecraft_mod_ai.host_rule_provenance import resolve_materialized_entity_presence


WORLD = "example.World"
ENTITY = "net.minecraft.world.entity.Entity"
ALIEN = "example.AlienCombatHostileEntity"
PATH = "src/main/java/example/AlienCombatHostileEntity.java"
SOURCE = "package example; public class AlienCombatHostileEntity extends net.minecraft.world.entity.Entity {}"


def production_index():
    symbols = {
        "java.lang.Object": {"kind": "CLASS", "owner": "java.lang.Object", "access": 1},
        WORLD: {"kind": "CLASS", "owner": WORLD, "access": 1},
        ENTITY: {"kind": "CLASS", "owner": ENTITY, "access": 1},
        "java.lang.Iterable": {"kind": "CLASS", "owner": "java.lang.Iterable", "access": 1537},
        "all_entities": {"kind": "METHOD", "owner": WORLD, "name": "getAllEntities",
                         "access": 1, "return_type": "java.lang.Iterable", "parameters": []},
    }
    project, rejected = inspected_project_source_symbols(
        {PATH: java_source_units(SOURCE, path=PATH)}, symbols)
    assert not rejected
    return {**symbols, **project}


def program(symbols, behavior=None):
    return RuleProgram(symbols, [{"type": WORLD, "name": "arg0"}], behavior or {},
                       owner="example.Bridge", target_path="src/main/java/example/Bridge.java")


def test_actual_production_source_index_materializes_live_entity_observer():
    symbols = production_index()
    assert symbols[ALIEN]["provenance"] == "materialized_source"
    assert symbols[ALIEN]["source_hash"] == hashlib.sha256(SOURCE.encode()).hexdigest()
    evidence = resolve_materialized_entity_presence("Alien presence", symbols, [WORLD])
    assert evidence is not None, "Production source evidence must not disappear at rule handoff"
    p = program(symbols)
    entities = p.value("call", [p.inputs[0]], symbol_ref=evidence["observer_ref"])
    presence = p.value("entity_presence", [entities], entity_owner=evidence["entity_owner"])
    assert "instanceof " + ALIEN in p.values[presence]["initializer"]


@pytest.mark.parametrize("missing", ["source_hash", "source_path"])
def test_materialized_label_without_source_identity_cannot_certify_entity(missing):
    symbols = production_index()
    symbols[ALIEN].pop(missing)
    assert resolve_materialized_entity_presence("Alien presence", symbols, [WORLD]) is None
    p = program(symbols)
    entities = p.value("call", [p.inputs[0]], symbol_ref="all_entities")
    with pytest.raises(ValueError, match="RULE_ENTITY_PRESENCE_TYPE_UNVERIFIED"):
        p.value("entity_presence", [entities], entity_owner=ALIEN)


def test_fallback_declaration_index_preserves_same_source_identity():
    from minecraft_mod_ai.host_prerequisite_api import java_api_declarations, inspected_prerequisite_symbols
    indexed = inspected_prerequisite_symbols([java_api_declarations(SOURCE, path=PATH)])
    assert indexed[ALIEN]["source_path"] == PATH
    assert indexed[ALIEN]["source_hash"] == hashlib.sha256(SOURCE.encode()).hexdigest()


def test_host_builds_complete_multihop_path_without_operation_by_operation_search():
    from minecraft_mod_ai.host_rule_source_plan import RuntimeSourcePlanner
    symbols = production_index()
    symbols.update({
        "example.Controls": {"kind": "CLASS", "owner": "example.Controls", "access": 1},
        "controls": {"kind": "METHOD", "owner": WORLD, "name": "getControls",
                     "access": 1, "parameters": [], "return_type": "example.Controls"},
        "reactor_ready": {"kind": "METHOD", "owner": "example.Controls", "name": "isReactorReady",
                          "access": 1, "parameters": [], "return_type": "boolean"},
    })
    paths = RuntimeSourcePlanner(symbols).plans("reactor_ready", [WORLD])
    assert len(paths) == 1
    assert paths[0]["symbol_refs"] == ["controls", "reactor_ready"]
    assert paths[0]["input_index"] == 0
    assert paths[0]["return_type"] == "boolean"


def test_captured_partial_match_is_not_a_complete_runtime_source():
    import json
    from pathlib import Path
    from minecraft_mod_ai.host_rule_source_plan import RuntimeSourcePlanner
    capture = json.loads((Path(__file__).parent / "fixtures/runtime_source_failure_071e9087.json").read_text())
    source_name = capture["behavior"]["runtime_state_reads"][0]
    getter = capture["wrong_observation"]["inspected_api"]["name"]
    symbols = production_index()
    symbols.update({
        "height": {"kind": "METHOD", "owner": WORLD, "name": getter,
                   "access": 1, "parameters": [], "return_type": "int"},
        "jna": {"kind": "METHOD", "owner": "com.sun.jna.CallbackReference",
                "name": "getCallback", "static": True, "access": 9,
                "parameters": ["java.lang.Class", "com.sun.jna.Pointer"],
                "return_type": "com.sun.jna.Callback"},
    })
    assert RuntimeSourcePlanner(symbols).plans(source_name, [WORLD]) == []
    from minecraft_mod_ai.host_rule_provenance import runtime_source_expression_proof
    p = program(symbols)
    height = p.value("call", [p.inputs[0]], symbol_ref="height")
    assert not runtime_source_expression_proof(p, height, source_name)


def test_sources_must_be_reachable_together_on_one_callback():
    from minecraft_mod_ai.host_rule_source_plan import RuntimeSourcePlanner, bind_runtime_sources
    from minecraft_mod_ai.host_rule_provenance import RuleMissingRuntimeProducer
    symbols = production_index()
    symbols.update({
        "example.Player": {"kind": "CLASS", "owner": "example.Player", "access": 1},
        "selected": {"kind": "METHOD", "owner": "example.Player", "name": "getSelectedSection",
                     "parameters": [], "access": 1, "return_type": "int"},
    })
    names = ["Alien presence", "selected_section"]
    behavior = {"behavior_id": "joint", "runtime_state_reads": names}
    bindings = {name: {"binding": "world_state"} for name in names}
    events = [{"event_id": "world", "parameters": [WORLD]},
              {"event_id": "player", "parameters": ["example.Player"]}]
    with pytest.raises(RuleMissingRuntimeProducer) as error:
        bind_runtime_sources(behavior, bindings, RuntimeSourcePlanner(symbols), events)
    assert {row["name"] for row in error.value.missing} == set(names)
    assert all(row["reason"] == "no_single_callback_supplies_required_sources"
               for row in error.value.missing)
    both = {"event_id": "both", "parameters": [WORLD, "example.Player"]}
    eligible, plans = bind_runtime_sources(behavior, bindings, RuntimeSourcePlanner(symbols), [*events, both])
    assert eligible == [both]
    assert set(plans["both"]) == set(names)


def test_derived_entity_value_is_grounded_and_pinned_beyond_recent_window():
    from minecraft_mod_ai.host_rule_author import RuleAuthor
    p = program(production_index())
    entities = p.value("call", [p.inputs[0]], symbol_ref="all_entities")
    presence = p.value("entity_presence", [entities], entity_owner=ALIEN)
    p.behavior["runtime_source_values"] = {"Alien presence": presence}
    for n in range(20):
        p.value("literal", type="int", value=n)

    class CheckPrompt:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            import json
            task = json.loads(messages[-1]["content"])
            assert presence not in [x["id"] for x in task["recent_values"]]
            assert task["runtime_sources"] == [{"name": "Alien presence", "id": presence, "type": "boolean"}]
            return {"mode": "level"}

    author = RuleAuthor(CheckPrompt(), p)
    assert author._runtime_grounded(presence)
    author.ask("rule_activation", "Choose activation", {"mode": {"type": "string", "enum": ["level"]}})


class SourceRuleRouter:
    """Deterministic semantic choices; the host must discover every receiver."""
    def __init__(self):
        self.effect_started = False
        self.literal = False
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        import json
        task = json.loads(messages[-1]["content"])
        self.calls.append(data_name)
        purpose = task.get("subgoal", "")
        if data_name == "rule_observation":
            return {"observation": "world_state"}
        if data_name == "rule_activation":
            return {"mode": "level"}
        if data_name == "rule_effect_operation":
            action = "complete" if self.effect_started else "assign"
            self.effect_started = True
            return {"operation": action}
        if data_name == "rule_value_operation":
            if purpose.startswith("Original trigger"):
                assert "call" not in parameters["properties"]["operation"]["enum"]
                return {"operation": "use"}
            if purpose.startswith("Choose or construct"):
                exists = any(x["purpose"] == "effects" for x in task["recent_values"])
                return {"operation": "use" if exists else "field"}
            if purpose == "Value to assign":
                return {"operation": "use" if self.literal else "literal"}
        if data_name in {"rule_api_query", "rule_api_page"}:
            assert self.effect_started, "Source construction must never ask for global API search"
            return {"query": "effects"} if data_name == "rule_api_query" else {"symbol": "s0"}
        if data_name == "rule_operand":
            values = task.get("values", [])
            if purpose.startswith("Original trigger"):
                return {"value_ref": task["runtime_sources"][0]["id"]}
            if purpose.startswith("Receiver"):
                return {"value_ref": "v0"}
            if purpose.startswith("Choose or construct"):
                return {"value_ref": next(x["id"] for x in values if x["purpose"] == "effects")}
            if purpose == "Value to assign":
                return {"value_ref": next(x["id"] for x in values if x["purpose"] == purpose)
                        if self.literal else "build"}
        if data_name == "rule_literal_type":
            return {"type": "int"}
        if data_name == "rule_literal":
            self.literal = True
            return {"value": 1}
        if data_name == "rule_assignment_operator":
            return {"operator": "+="}
        raise AssertionError((data_name, purpose))

    def generate_java_source(self, *args, **kwargs):
        raise AssertionError("No Java source fallback")


def test_production_entity_handoff_compiles_and_executes_rule(tmp_path, monkeypatch):
    import json
    from test_host_rule_feature import LEVEL, compile_run, index
    from minecraft_mod_ai.construction_host_source import compose_host_java_source

    symbols = {**production_index(), **index()}
    symbols["all_entities"] = {**symbols["all_entities"], "owner": LEVEL}
    path = tmp_path / PATH
    path.parent.mkdir(parents=True)
    path.write_text(SOURCE)
    router = SourceRuleRouter()

    def verify(source, *args, **kwargs):
        compile_run(tmp_path, source, """
            ServerLevel w=new ServerLevel(); tick(w,0); check(w.effects==0);
            w.items.add(new net.minecraft.world.entity.Entity()); tick(w,1); check(w.effects==0);
            w.items.add(new AlienCombatHostileEntity()); tick(w,2); check(w.effects==1);
            tick(w,3); check(w.effects==2); w.items.clear(); tick(w,4); check(w.effects==2);
        """, extra_sources={
            "net/minecraft/world/entity/Entity.java": "package net.minecraft.world.entity; public class Entity {}",
            "example/AlienCombatHostileEntity.java": SOURCE,
            "net/minecraft/server/level/ServerLevel.java": (
                "package net.minecraft.server.level; public class ServerLevel { public long time; public int effects; "
                "public java.util.List<net.minecraft.world.entity.Entity> items=new java.util.ArrayList<>(); "
                "public java.lang.Iterable<net.minecraft.world.entity.Entity> getAllEntities(){return items;} }"),
        })

    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._verify_full_source", verify)
    monkeypatch.setattr("minecraft_mod_ai.construction_coverage.record_authored_source_calls", lambda *a, **kw: None)
    source = compose_host_java_source(router, {
        "language": "java", "side": "COMMON", "target_path": "src/main/java/example/Bridge.java",
        "bindings": {"request_feature_entrypoint": "example.Bridge",
                     "runtime_state_bindings_json": json.dumps({"Alien presence": {"binding": "world_state"}})},
    }, {"symbols": symbols, "project_root": str(tmp_path), "target": {"loader": "fabric"},
        "java_classpath": [str(tmp_path)]}, package="example", name="Bridge", owner="example.Bridge",
        behaviors=[{"behavior_id": "production-entity", "intent": "Observe live entity presence",
                    "trigger": "Alien presence", "effects": "Increment effects",
                    "state_reads": [], "state_writes": [], "runtime_state_reads": ["Alien presence"]}],
        source_context=[{"path": PATH}])
    assert "instanceof " + ALIEN in source
    assert router.calls[:4] == ["rule_observation", "rule_activation", "rule_value_operation", "rule_operand"]
    receipts = list((tmp_path / ".mmm/construction_rule_ir").glob("*.json"))
    record = json.loads(receipts[0].read_text())["members"]
    assert record["rules"][0]["runtime_source_plans"]["Alien presence"]["entity_owner"] == ALIEN
    assert record["runtime_verified"] is False


def test_complete_multihop_source_skips_absent_receiver_then_reads_live_values(tmp_path):
    from test_host_rule_feature import LEVEL, compile_run, index
    from minecraft_mod_ai.host_rule_feature import compose_rule_feature

    symbols = index()
    symbols.update({
        "example.Controls": {"kind": "CLASS", "owner": "example.Controls", "access": 1},
        "controls": {"kind": "METHOD", "owner": LEVEL, "name": "getControls",
                     "access": 1, "parameters": [], "return_type": "example.Controls"},
        "reactor_ready": {"kind": "METHOD", "owner": "example.Controls", "name": "isReactorReady",
                          "access": 1, "parameters": [], "return_type": "boolean"},
    })

    source = compose_rule_feature(SourceRuleRouter(), {
        "side": "COMMON", "target_path": "src/main/java/example/Bridge.java",
        # Real materialized source signatures, not a speculative planner label,
        # are the authority for this concrete receiver→observation path.
        "bindings": {"runtime_state_bindings_json": "{}"},
    }, {"symbols": symbols, "project_root": str(tmp_path)}, package="example", name="Bridge",
        behaviors=[{"behavior_id": "multihop", "trigger": "Reactor is ready", "effects": "Increment effects",
                    "state_reads": [], "state_writes": [], "runtime_state_reads": ["reactor_ready"]}], state_access=[])
    compile_run(tmp_path, source, """
        ServerLevel w=new ServerLevel(); tick(w,0); check(w.effects==0);
        w.controls=new Controls(); tick(w,1); check(w.effects==0);
        w.controls.ready=true; tick(w,2); check(w.effects==1);
        w.controls=null; tick(w,3); check(w.effects==1);
    """, extra_sources={
        "example/Controls.java": "package example; public class Controls { public boolean ready; public boolean isReactorReady(){return ready;} }",
        "net/minecraft/server/level/ServerLevel.java": (
            "package net.minecraft.server.level; public class ServerLevel { public long time; public int effects; "
            "public example.Controls controls; public example.Controls getControls(){return controls;} }"),
    })


def test_object_source_predicate_can_inspect_its_pinned_receiver():
    from collections import deque
    import json
    from minecraft_mod_ai.host_rule_author import RuleAuthor
    symbols = production_index()
    inventory = "example.Inventory"
    symbols.update({
        inventory: {"kind": "CLASS", "owner": inventory, "access": 1},
        "inventory": {"kind": "METHOD", "owner": WORLD, "name": "getInventory",
                      "access": 1, "parameters": [], "return_type": inventory},
        "empty": {"kind": "METHOD", "owner": inventory, "name": "isEmpty",
                  "access": 1, "parameters": [], "return_type": "boolean"},
        "unrelated": {"kind": "METHOD", "owner": WORLD, "name": "isEmpty",
                      "access": 1, "parameters": [], "return_type": "boolean"},
    })
    p = program(symbols)
    held = p.value("call", [p.inputs[0]], symbol_ref="inventory")
    p.behavior.update(runtime_source_values={"inventory": held}, runtime_source_symbols=["inventory"])
    replies = deque([
        ("rule_value_operation", {"operation": "call"}),
        ("rule_api_query", {"query": "isEmpty"}),
        ("rule_api_page", {"symbol": "s0"}),
        ("rule_operand", {"value_ref": held}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v2"}),
    ])

    class SelectInventory:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            expected, reply = replies.popleft()
            assert data_name == expected
            if data_name == "rule_api_page":
                choices = json.loads(messages[-1]["content"])["symbols"]
                assert len(choices) == 1 and choices[0]["owner"] == inventory
            return reply

    result = RuleAuthor(SelectInventory(), p).value("Inspect live inventory", expected="boolean", require_grounded=True)
    assert not replies
    assert p.checker.nodes[result]["attributes"]["symbol_ref"] == "empty"
    assert p.checker.nodes[result]["input_refs"] == [held]


def test_two_world_inputs_offer_two_complete_entity_presence_sources():
    from minecraft_mod_ai.host_rule_source_plan import RuntimeSourcePlanner
    paths = RuntimeSourcePlanner(production_index()).plans("Alien presence", [WORLD, WORLD])
    assert len(paths) == 2
    assert {path["input_index"] for path in paths} == {0, 1}
    assert all(path["entity_owner"] == ALIEN for path in paths)


def test_missing_planner_binding_still_uses_inspected_receiver_producer():
    from minecraft_mod_ai.host_rule_source_plan import (
        RuntimeSourcePlanner, bind_runtime_sources,
    )
    symbols = production_index()
    symbols["example.Controls"] = {
        "kind": "CLASS", "owner": "example.Controls", "access": 1}
    symbols["controls"] = {
        "kind": "METHOD", "owner": WORLD, "name": "getControls",
        "access": 1, "parameters": [], "return_type": "example.Controls",
    }
    symbols["ready"] = {
        "kind": "METHOD", "owner": "example.Controls",
        "name": "isReactorReady", "access": 1, "parameters": [],
        "return_type": "boolean",
    }
    behavior = {"behavior_id": "ready", "runtime_state_reads": ["reactor_ready"]}
    event = {"event_id": "world", "parameters": [WORLD]}
    eligible, sources = bind_runtime_sources(
        behavior, {}, RuntimeSourcePlanner(symbols), [event])
    assert eligible == [event]
    assert sources["world"]["reactor_ready"][0]["symbol_refs"] == [
        "controls", "ready",
    ]


def test_contradictory_explicit_runtime_binding_never_authorizes_a_producer():
    from minecraft_mod_ai.host_rule_source_plan import (
        RuntimeSourcePlanner, bind_runtime_sources,
    )
    from minecraft_mod_ai.host_rule_provenance import RuleMissingRuntimeProducer
    with pytest.raises(RuleMissingRuntimeProducer, match="runtime_binding_namespace_conflict"):
        bind_runtime_sources(
            {"behavior_id": "conflict", "runtime_state_reads": ["reactor_ready"]},
            {"reactor_ready": {"binding": "canonical_state"}},
            RuntimeSourcePlanner(production_index()),
            [{"event_id": "world", "parameters": [WORLD]}],
        )
