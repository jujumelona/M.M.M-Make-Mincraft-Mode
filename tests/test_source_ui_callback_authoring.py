"""Host-seeded source callbacks require real structured Java implementation."""
from copy import deepcopy
import json

import pytest

from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
from minecraft_mod_ai.source_ui_actions import source_ui_callback
from minecraft_mod_ai.source_ui_actions_java import render_source_ui_action_files
from minecraft_mod_ai.typed_action_java import render_action_dispatch_java
from minecraft_mod_ai.typed_plan_java import render_typed_plan_java
from test_incremental_java_value_integration import compile_run, literal, local, real_builder
from test_incremental_java_values import ValueCoder, member_named, symbols
from test_source_ui_actions_java import _event_fixture, source_plan


def callback_spec(records=None):
    plan, _, _ = source_plan()
    record = {"source_requirement_id": "feature_a", "trigger": "Player confirms operation",
              "server_action": {"intent": "Apply requested operation", "inputs": [
                  {"name": "amount", "label": "Amount", "type": "int", "minimum": 1, "maximum": 10}]}}
    records = [record] if records is None else records
    callbacks = [{**source_ui_callback(row, package="test"), "source_ui_record": deepcopy(row)}
                 for row in records]
    return {"target_path": "src/main/java/test/Feature.java", "bindings": {
        "package_name": "test", "class_name": "Feature", "parent_requirement_id": "feature_a",
        "source_ui_callbacks_json": json.dumps(callbacks)}}, callbacks, plan


def builder_for(spec, steps=(), *, program=None, behaviors=None, api=None):
    return IncrementalJavaBuilder(ValueCoder(steps), spec, [], {"symbols": api or {}}, behaviors or [
        {"behavior_id": "original", "intent": "Preserve original operation conditions and effects",
         "conditions": ["actor is eligible"]}], program=program)


def test_callbacks_seed_exact_listener_abi_and_accepted_registration_before_model_choice():
    spec, callbacks, _ = callback_spec()
    builder = builder_for(spec)
    registrar, listener = builder.program["members"]
    assert registrar["registrar"]
    assert listener["event"] == callbacks[0]
    assert listener["metadata"]["return_type"] == "int"
    assert listener["metadata"]["parameters"] == [
        {"type": "net.minecraft.commands.CommandSourceStack", "name": "source"},
        {"type": "int", "name": "input_0"}]
    assert listener["behavior"] == registrar["behavior"]
    assert listener["implementation_status"] == "pending"
    assert list(builder.pending) == [listener, registrar]
    registration, = registrar["body"]
    assert registration == {"id": registration["id"], "op": "register",
                            "event_owner": callbacks[0]["event_owner"],
                            "event_field": callbacks[0]["event_field"], "handler": listener["name"]}
    assert builder.history[registration["id"]] is registration
    assert {listener["id"], registration["id"]} <= builder.identities
    assert not builder.router.requests
    assert callbacks[0]["event_id"] not in {row["event_id"] for row in builder.api.callbacks()}
    builder.member = listener
    builder.scope = {"locals": listener["metadata"]["parameters"]}
    context = builder.payload()
    assert context["current_member"]["event"]["source_ui_record"] == callbacks[0]["source_ui_record"]
    assert context["assigned_behavior"] == registrar["behavior"]


def test_checkpoint_does_not_seed_again_or_discard_accepted_program_work():
    spec, _, _ = callback_spec()
    builder = builder_for(spec)
    program = deepcopy(builder.program)
    restored = builder_for(spec, program=program)
    assert restored.program == program
    assert not restored.pending
    assert len(restored.program["members"]) == 2


def generated_namespace_spec():
    spec, callbacks, _ = callback_spec()
    record = callbacks[0]["source_ui_record"]
    callback = {**source_ui_callback(record, package="demo"), "source_ui_record": record}
    spec["target_path"] = "src/main/java/demo/generated/Feature.java"
    spec["bindings"].update(package_name="demo.generated", source_ui_package_name="demo",
                            source_ui_callbacks_json=json.dumps([callback]))
    return spec, callback


def test_source_ui_callback_namespace_can_differ_from_generated_feature_namespace():
    spec, callback = generated_namespace_spec()
    builder = builder_for(spec)
    assert builder.program["package"] == "demo.generated"
    registrar, listener = builder.program["members"]
    assert listener["event"] == callback
    assert registrar["body"][0]["event_owner"] == "demo.AuthoredSourceActions"
    assert list(builder.pending) == [listener, registrar]


@pytest.mark.parametrize("changed", ["source_ui_package_name", "callback_abi"])
def test_explicit_source_ui_namespace_rejects_changed_package_or_callback_abi(changed):
    spec, callback = generated_namespace_spec()
    if changed == "source_ui_package_name":
        spec["bindings"][changed] = "demo.generated"
    else:
        record = callback["source_ui_record"]
        spec["bindings"]["source_ui_callbacks_json"] = json.dumps([
            {**source_ui_callback(record, package="demo.generated"), "source_ui_record": record}])
    with pytest.raises(ValueError, match="SOURCE_UI_CALLBACK_ABI_MISMATCH"):
        builder_for(spec)


@pytest.mark.parametrize("field,value", [
    ("event_owner", "foreign.AuthoredSourceActions"),
    ("parameters", ["java.lang.Object", "int"]),
    ("parameter_names", ["source", "amount"]),
    ("return_type", "void"),
    ("parent_requirement_id", "foreign"),
    ("trigger", "Altered original trigger"),
    ("source_requirement_id", "foreign"),
])
def test_callback_binding_rejects_foreign_owner_or_modified_abi(field, value):
    spec, callbacks, _ = callback_spec()
    callbacks[0][field] = value
    spec["bindings"]["source_ui_callbacks_json"] = json.dumps(callbacks)
    with pytest.raises(ValueError, match="SOURCE_UI_CALLBACK"):
        builder_for(spec)


def test_callback_binding_rejects_a_foreign_source_and_duplicate_action():
    spec, callbacks, _ = callback_spec()
    spec["bindings"]["parent_requirement_id"] = "foreign"
    with pytest.raises(ValueError, match="SOURCE_UI_CALLBACK"):
        builder_for(spec)
    spec["bindings"]["parent_requirement_id"] = "feature_a"
    spec["bindings"]["source_ui_callbacks_json"] = json.dumps(callbacks * 2)
    with pytest.raises(ValueError, match="SOURCE_UI_CALLBACK"):
        builder_for(spec)


@pytest.mark.parametrize("raw", ["{}", "null", "[null]", "[{}]", "not json"])
def test_callback_binding_rejects_malformed_host_metadata(raw):
    spec, _, _ = callback_spec()
    spec["bindings"]["source_ui_callbacks_json"] = raw
    with pytest.raises(ValueError, match="SOURCE_UI_CALLBACK"):
        builder_for(spec)


def test_all_input_types_keep_fixed_names_and_original_metadata_in_callback_context():
    spec, callbacks, _ = callback_spec()
    record = callbacks[0]["source_ui_record"]
    record["server_action"]["inputs"] += [
        {"name": "enabled", "label": "Enabled", "type": "boolean"},
        {"name": "epoch", "label": "Epoch", "type": "long", "minimum": 0, "maximum": 100},
        {"name": "ratio", "label": "Ratio", "type": "double", "minimum": 0, "maximum": 1},
        {"name": "name", "label": "Name", "type": "string", "max_length": 20, "choices": []}]
    spec, callbacks, _ = callback_spec([record])
    callbacks[0]["parent_requirement_id"] = "feature_a"
    spec["bindings"]["source_ui_callbacks_json"] = json.dumps(callbacks)
    listener = builder_for(spec).program["members"][-1]
    assert listener["event"] == callbacks[0]
    assert [row["type"] for row in listener["metadata"]["parameters"]] == [
        "net.minecraft.commands.CommandSourceStack", "int", "boolean", "long", "double", "java.lang.String"]
    assert [row["name"] for row in listener["metadata"]["parameters"]] == [
        "source", "input_0", "input_1", "input_2", "input_3", "input_4"]


def test_absent_or_empty_callback_metadata_preserves_normal_builder_work():
    spec, _, _ = callback_spec()
    spec["bindings"].pop("source_ui_callbacks_json")
    normal = builder_for(spec)
    spec["bindings"]["source_ui_callbacks_json"] = "[]"
    empty = builder_for(spec)
    assert normal.program == empty.program
    assert list(normal.pending) == list(empty.pending)
    assert len(empty.program["members"]) == 1


def test_seed_preserves_non_ui_work_history_counters_and_original_bodies():
    from minecraft_mod_ai.source_ui_callback_authoring import seed_source_ui_callbacks
    builder = real_builder([])
    original = builder.program["members"][0]
    node = {"id": "n10", "op": "call", "receiver": "this", "method": "other", "arguments": []}
    original["body"].append(node)
    builder.history[node["id"]] = node
    builder.identities.add(node["id"])
    spec, callbacks, _ = callback_spec()
    # Recompute the fixture ABI for this real builder's package.
    row = callbacks[0]["source_ui_record"]
    spec["bindings"]["source_ui_callbacks_json"] = json.dumps([
        {**source_ui_callback(row, package="demo"), "source_ui_record": row}])
    builder.spec["bindings"].update(spec["bindings"])
    before = deepcopy(original)
    seed_source_ui_callbacks(builder)
    assert original["body"][0] == before["body"][0]
    assert {key: value for key, value in original.items() if key != "body"} == {
        key: value for key, value in before.items() if key != "body"}
    assert builder.history["n10"] is node
    assert builder.allocate("n") != "n10"
    assert len(builder.program["members"]) == 2


def effect_steps(field_index):
    return [("kind", {"op": "field"}), ("field", {"type": "int",
             "purpose": "Retain the source action's requested amount", "static": False, "final": False}),
            *literal(0), ("kind", {"op": "assign"}), ("assign", {"operator": "+="}),
            ("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "this"}),
            ("field_member", member_named("mmmField" + str(field_index))), *local("input_0"),
            ("kind", {"op": "assign"}), ("assign", {"operator": "+="}),
            ("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "value"}),
            *local("source"), ("field_member", member_named("effect")), *local("input_0"),
            ("kind", {"op": "return"}), *local("input_0")]


def test_real_structured_callbacks_execute_original_ui_input_with_two_actions(tmp_path):
    spec, callbacks, plan = callback_spec()
    first = callbacks[0]["source_ui_record"]
    second = deepcopy(first)
    second["trigger"] = "Player confirms second operation"
    second["server_action"]["intent"] = "Apply second requested operation"
    spec, callbacks, _ = callback_spec([first, second])
    api = symbols("net.minecraft.commands.CommandSourceStack", {
        "kind": "FIELD", "owner": "net.minecraft.commands.CommandSourceStack",
        "name": "effect", "return_type": "int", "access": 1})
    builder = builder_for(spec, [*effect_steps(1), *effect_steps(2), ("kind", {"op": "complete"}),
                                ("kind", {"op": "complete"})], api=api, behaviors=[
        {"behavior_id": "first", "intent": "Original source effect"},
        {"behavior_id": "other", "intent": "Retain separate non UI work"}])
    source, program = builder.finish()
    assert all(row["implementation_status"] == "complete" for row in program["members"] if row["kind"] == "method")
    assert [len(row["body"]) for row in program["members"] if row.get("registrar")] == [2, 0]
    assert builder_for(spec, program=program).program == program
    assert not any(task["requested_page"] == "register" for task, _ in builder.router.requests)
    listener_requests = [task for task, _ in builder.router.requests if "event" in task["current_member"]]
    assert {task["current_member"]["event"]["action_id"] for task in listener_requests} == {
        row["action_id"] for row in callbacks}
    assert all(task["current_member"]["event"]["source_ui_record"] in [first, second]
               for task in listener_requests)
    # Render the host transport for both original UI actions.
    from minecraft_mod_ai.source_ui_actions import _source_ui_delegate_body
    original_function = deepcopy(plan["functions"][0])
    original_binding = deepcopy(plan["event_bindings"][0])
    plan["functions"], plan["event_bindings"] = [], []
    for index, callback in enumerate(callbacks):
        function, binding = deepcopy(original_function), deepcopy(original_binding)
        function.update(id="action_" + str(index), body=_source_ui_delegate_body(callback["action_id"], callback["inputs"]))
        binding["function"] = function["id"]
        binding["entry_point_index"] = index
        binding["config"]["literal"] = "operation_" + str(index)
        binding["config"]["action"]["id"] = callback["action_id"]
        plan["functions"].append(function)
        plan["event_bindings"].append(binding)
    files = render_source_ui_action_files(plan, package="test", minecraft_version="26.2")
    runtime = deepcopy(plan)
    runtime["event_bindings"] = []
    _, caps, _ = source_plan()
    files["test/AuthoredProgram.java"] = render_typed_plan_java(runtime, package="test", capabilities=caps)
    files["test/AuthoredActions.java"] = render_action_dispatch_java(plan, package="test")
    files.update(_event_fixture())
    files["test/Feature.java"] = source
    files["net/fabricmc/api/ModInitializer.java"] = "package net.fabricmc.api; public interface ModInitializer {void onInitialize();}"
    files["test/Harness.java"] = '''package test;
public class Harness {
 public static void main(String[] args) throws Exception {
  Feature feature = new Feature(); feature.onInitialize();
  var source = new net.minecraft.commands.CommandSourceStack();
  if (AuthoredActions.invoke("FIRST", source, java.util.Map.of("amount", 3)) != 3) throw new AssertionError();
  if (AuthoredActions.invoke("SECOND", source, java.util.Map.of("amount", 4)) != 4) throw new AssertionError();
  if (source.effect != 7) throw new AssertionError("missing authored effect");
  var firstState = Feature.class.getDeclaredField("mmmField1"); firstState.setAccessible(true);
  var secondState = Feature.class.getDeclaredField("mmmField2"); secondState.setAccessible(true);
  if (firstState.getInt(feature) != 3 || secondState.getInt(feature) != 4) throw new AssertionError("missing new authored state");
  feature.onInitialize();
  try { AuthoredActions.invoke("FIRST", source, java.util.Map.of("amount", 3)); throw new AssertionError("duplicate listener"); }
  catch (IllegalStateException expected) {}
  if (source.effect != 7) throw new AssertionError("duplicate effect");
 }
}'''.replace("FIRST", callbacks[0]["action_id"]).replace("SECOND", callbacks[1]["action_id"])
    compile_run(tmp_path, files, "test.Harness")
