"""Requested features are assembled from operations, never whole Java bodies."""
from __future__ import annotations

import json
import subprocess
import shutil

import pytest

from test_direct_feature_source import _fixture
from incremental_value_model import structured_values


@structured_values
class OperationCoder:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        structured_values(cls)

    def generate_java_source(self, *args, **kwargs):
        raise AssertionError("Requested-feature values must use structured model pages")

    def __init__(self):
        self.requests = []
        self.expression_requests = []
        self.steps = {
            "m1": iter([("field", {"type": "int", "purpose": "Count distinct eligible claims", "static": False, "final": False}),
                        ("register", {})]),
            "m2": iter([("if", {}), ("assign", {"operator": "+="})]),
        }
        self.completed = set()

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        task = json.loads(messages[-1]["content"])
        self.requests.append((data_name, task, parameters))
        assert len(json.dumps(messages, ensure_ascii=False)) < 16000
        assert "available_fabric_callbacks" not in task
        if data_name.endswith("_kind"):
            member = task["current_member"]["id"]
            position = task["block_path"]
            if member == "m2" and position:
                return {"op": "return"} if not task.get("accepted_operations") else {"op": "complete"}
            try:
                self.current_op, self.details = next(self.steps[member])
                return {"op": self.current_op}
            except StopIteration:
                return {"op": "complete"}
        if data_name.endswith("_register"):
            return {"event_id": "example.Events#ACTION"}
        if data_name.endswith("_otherwise"):
            return {"enabled": False}
        return self.details

    def value_answer(self, role, messages, *, data_name, source_frame):
        # Only leaf expressions may ever reach source decoding, never a method.
        assert source_frame["kind"] == "field"
        task = json.loads(messages[-1]["content"])
        self.expression_requests.append(task)
        slot = task["expression_slot"]
        name = task["scope"]["locals"][0]["name"] if task["scope"]["locals"] else ""
        expression = {"initializer": "0", "condition": f"{name} == null || {name}.isBlank()",
                      "target": "this.mmmField1", "value": "1"}[slot]
        return source_frame["prefix"] + expression + source_frame["suffix"]


def _inputs():
    behavior, spec, context = _fixture()
    context["symbols"].update({
        "java.lang.String": {"kind": "CLASS", "owner": "java.lang.String", "access": 1},
        "String#isBlank": {"kind": "METHOD", "owner": "java.lang.String", "name": "isBlank",
                            "parameters": [], "return_type": "boolean", "access": 1}})
    callback = {"event_id": "example.Events#ACTION", "event_owner": "example.Events", "event_field": "ACTION",
                "event_interface": "java.util.function.Consumer<String>", "method": "accept",
                "parameters": ["String"], "parameter_names": ["arg0"], "return_type": "void", "throws": [], "type_parameters": []}
    messages = [{"role": "user", "content": json.dumps({"requirement": behavior["intent"],
                 "assigned_behaviors": [behavior], "available_fabric_callbacks": [callback]})}]
    return behavior, spec, context, messages


def test_operation_author_writes_new_state_and_listener_without_a_member_source_request(tmp_path):
    from minecraft_mod_ai.incremental_java_author import compose_incremental_feature
    behavior, spec, context, messages = _inputs()
    coder = OperationCoder()
    source, program = compose_incremental_feature(coder, spec, messages, construction_context=context, behaviors=[behavior])
    assert "this.mmmField1 += 1 ;" in " ".join(source.split())
    assert "example.Events .ACTION .register(this::" in " ".join(source.split())
    assert len(program["members"]) == 3
    assert coder.expression_requests and all("current_operation" in p for p in coder.expression_requests)
    assert not any("submit_java_construction_plan" in name for name, _, _ in coder.requests)
    _compile_claims(tmp_path, source)


def _compile_claims(tmp_path, source, *, execute=True):
    java = shutil.which("java")
    if java is None:
        pytest.skip("Java runtime required")
    files = {"example/Feature.java": source,
             "net/fabricmc/api/ModInitializer.java": "package net.fabricmc.api; public interface ModInitializer {void onInitialize();}",
             "example/Events.java": "package example; public class Events {public static final Bus ACTION = new Bus(); public static class Bus {java.util.function.Consumer<String> listener; public void register(java.util.function.Consumer<String> v){listener=v;} public void fire(String v){listener.accept(v);}}}",
             "example/Harness.java": 'package example; public class Harness {public static void main(String[] args) throws Exception {Feature f=new Feature();f.onInitialize();Events.ACTION.fire("");Events.ACTION.fire("alpha");Events.ACTION.fire("beta");var v=Feature.class.getDeclaredField("mmmField1");v.setAccessible(true);if(v.getInt(f)!=2)throw new AssertionError("claims");}}'}
    paths=[]
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        paths.append(str(p))
    result=subprocess.run([java,"-m","jdk.compiler/com.sun.tools.javac.Main","-d",str(tmp_path),*paths],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    if not execute:
        return
    result=subprocess.run([java,"-cp",str(tmp_path),"example.Harness"],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_javac_error_repairs_only_one_ast_value_and_preserves_accepted_work(tmp_path):
    from copy import deepcopy
    from minecraft_mod_ai.incremental_java_author import compose_incremental_feature, IncrementalJavaBuilder
    from minecraft_mod_ai.incremental_java_repair import repair_incremental_program
    behavior, spec, context, messages = _inputs()
    source, program = compose_incremental_feature(OperationCoder(), spec, messages, construction_context=context, behaviors=[behavior])
    member = next(m for m in program["members"] if m["id"] == "m2")
    member["body"][-1]["value"] = "unresolvedAmount"
    broken, _ = IncrementalJavaBuilder(None, spec, messages, context, [behavior], program).render()
    before = deepcopy(program)
    with pytest.raises(AssertionError) as error:
        _compile_claims(tmp_path, broken)
    diagnostic = str(error.value)

    @structured_values
    class RepairCoder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert "repair_target" in data_name
            return {"repair": "mapped_operand"}

        def value_answer(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            assert source_frame["kind"] == "field"
            assert task["repair_slot"] == "value"
            assert "unresolvedAmount" in task["javac_diagnostic"]
            assert task["rejected_operation"]["operator"] == "+="
            assert task["operation_stack"][-1]["target"] == "this.mmmField1"
            return source_frame["prefix"] + "1" + source_frame["suffix"]

    revised, accepted = repair_incremental_program(RepairCoder(), broken, diagnostic,
        program=program, spec=spec, messages=messages, construction_context=context, behaviors=[behavior])
    _compile_claims(tmp_path, revised)
    before["members"][-1]["body"][-1]["value"] = "1"
    assert accepted == before
    assert program["members"][-1]["body"][-1]["value"] == "unresolvedAmount"


def test_operand_requests_include_selected_call_prior_effects_and_enclosing_branch():
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    from minecraft_mod_ai.incremental_java_operations import author_operation
    behavior, spec, context, messages = _inputs()

    @structured_values
    class OperandCoder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            if data_name.endswith("_call"):
                return {"receiver": "this", "method": "newHelper", "result_type": ""}
            if data_name.endswith("_arguments_count"):
                assert task["operation_stack"][-1]["method"] == "newHelper"
                return {"count": 2}
            raise AssertionError(data_name)

        def value_answer(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            node = task["operation_stack"][-1]
            assert node["receiver"] == "this" and node["method"] == "newHelper"
            assert task["operation_stack"][0]["condition"] == "ready"
            assert task["accepted_operations"][0]["value"] == "7"
            if task["expression_slot"] == "argument:1":
                assert node["arguments"] == ["1"]
            return source_frame["prefix"] + "1" + source_frame["suffix"]

    builder = IncrementalJavaBuilder(OperandCoder(), spec, messages, context, [behavior])
    builder.member = builder.program["members"][0]
    from minecraft_mod_ai.incremental_java_model import initial_scope
    builder.scope = initial_scope(builder.member)
    from minecraft_mod_ai.incremental_java_model import method_metadata
    helper = builder.new_method(owner="Feature", purpose="Apply both inputs", behavior=behavior, name="newHelper",
                                metadata=method_metadata(parameters=[{"type": "int", "name": "arg0"}, {"type": "int", "name": "arg1"}]))
    helper["implementation_status"] = "complete"
    builder.drafts = [{"id": "guard", "op": "if", "condition": "ready", "body": []}]
    builder.block = [{"id": "prior", "op": "assign", "target": "this.stock", "operator": "+=", "value": "7"}]
    result = author_operation(builder, "call")
    assert result["arguments"] == ["1", "1"]


def test_live_capacity_and_parser_correction_keep_the_incremental_operand_context():
    from minecraft_mod_ai.construction_java_context import scoped_java_member_messages
    from minecraft_mod_ai.construction_source_repair import focused_java_correction_messages
    frame = {"kind": "field", "prefix": "private boolean __value = ", "suffix": ";"}
    task = {"host_incremental_java": True, "immutable_signature": frame,
            "expression_slot": "condition", "scope": {"locals": [{"type": "String", "name": "arg0"}]},
            "operation_stack": [{"op": "if", "condition": "arg0 != null"}],
            "current_declaration": {}, "retained_typed_behavior": {"ui_owned_records": [{"action": "existingInput"}]}}
    messages = [{"role": "user", "content": json.dumps(task)}]
    projected = scoped_java_member_messages(messages, frame)
    assert json.loads(projected[-1]["content"]) == task
    corrected = focused_java_correction_messages(messages, frame=frame,
        response='{"implementation":"bad(}"}', diagnostic="missing expression")
    payload = json.loads(corrected[-1]["content"])
    assert payload["scope"] == task["scope"]
    assert payload["operation_stack"] == task["operation_stack"]
    assert payload["retained_typed_behavior"] == task["retained_typed_behavior"]
