"""Repair the declaration responsible for an error, retaining its accepted users."""
from __future__ import annotations

import json

from test_incremental_java_dependency_repair import _setup, _repair
from test_incremental_java_render import _java, _field


def _with_members(body, members, symbols=None):
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    data = _setup(body)
    _, program, spec, context, messages, behaviors = data
    if symbols:
        context["symbols"].update(symbols)
        context["incremental_state"].pop("api", None)
    program["members"].extend(members)
    source, _ = IncrementalJavaBuilder(None, spec, messages, context, behaviors, program).render()
    return source, *data[1:]


def test_a_callers_assignment_can_correct_the_fields_final_modifier(tmp_path):
    field = _field("counter", "int", "0")
    field["metadata"]["final"] = True
    body = [{"id": "increment", "op": "assign", "target": "this.counter", "operator": "+=", "value": "1"}]
    data = _with_members(body, [field])
    failed = _java(tmp_path, data[0])
    assert failed.returncode != 0 and "final variable counter" in failed.stderr

    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("repair_target"):
                return {"repair": "declaration"}
            if data_name.endswith("repair_related_declaration"):
                return {"declaration_id": "Demo:counter"}
            if data_name.endswith("repair_declaration_slot"):
                task = json.loads(messages[-1]["content"])
                assert task["current_member"]["name"] == "counter"
                assert not task["scope"]["locals"]
                return {"declaration_slot": "final"}
            return {"final": False}
    source, program = _repair(Coder(), data, failed.stderr)
    assert program["members"][0]["body"] == body
    assert _java(tmp_path, source).returncode == 0


def test_api_signatures_can_be_looked_up_during_repair_before_choosing_the_correct_method(tmp_path):
    body = [{"id": "convert", "op": "call", "receiver": "Integer", "method": "toMissingString",
             "arguments": ["1"], "result_type": "String", "name": "value"}]
    symbols = {"java.lang.Integer": {"kind": "CLASS", "owner": "java.lang.Integer", "access": 1},
               "java.lang.Integer#toString(int)": {"kind": "METHOD", "owner": "java.lang.Integer",
                "name": "toString", "access": 9, "static": True, "parameters": ["int"], "return_type": "String"}}
    data = _with_members(body, [], symbols)
    failed = _java(tmp_path, data[0])
    assert failed.returncode != 0 and "toMissingString" in failed.stderr

    class Coder:
        looked_up = False

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            if data_name.endswith("repair_target"):
                if not self.looked_up:
                    self.looked_up = True
                    return {"repair": "lookup"}
                assert task["requested_api_page"]["members"][0]["name"] == "toString"
                return {"repair": "mapped_operand"}
            if data_name.endswith("_lookup"):
                return {"query": "java.lang.Integer#toString", "offset": 0}
            return {"name": "toString"}
    source, program = _repair(Coder(), data, failed.stderr)
    assert program["members"][0]["body"][0]["arguments"] == ["1"]
    assert _java(tmp_path, source).returncode == 0


def test_unknown_record_component_type_can_be_corrected_without_replacing_the_record(tmp_path):
    record = {"id": "record", "owner": "Demo", "kind": "record", "name": "Entry", "body": [], "purpose": "Retain an entry", "behavior": {},
              "metadata": {"visibility": "private", "static": True, "interfaces": [],
                           "components": [{"type": "MissingType", "name": "value"}]}}
    data = _with_members([], [record])
    failed = _java(tmp_path, data[0])
    assert failed.returncode != 0 and "MissingType" in failed.stderr

    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("repair_target"):
                return {"repair": "declaration"}
            if data_name.endswith("repair_declaration_slot"):
                return {"declaration_slot": "component:0"}
            return {"type": "String"}
    source, program = _repair(Coder(), data, failed.stderr)
    assert program["members"][-1]["metadata"]["components"] == [{"type": "String", "name": "value"}]
    assert _java(tmp_path, source).returncode == 0


def test_class_completion_can_add_its_missing_interface_override(tmp_path):
    nested = {"id": "nested", "owner": "Demo", "kind": "class", "name": "Listener", "body": [],
              "purpose": "Run an action", "behavior": {},
              "metadata": {"visibility": "private", "static": True, "superclass": "", "interfaces": ["Runnable"]}}
    data = _with_members([], [nested])
    data[3]["symbols"].update({
        "java.lang.Runnable": {"kind": "CLASS", "owner": "java.lang.Runnable", "access": 0x601},
        "java.lang.Runnable#run()": {"kind": "METHOD", "owner": "java.lang.Runnable", "name": "run",
                                     "return_type": "void", "parameters": [], "access": 0x401}})
    data[3]["incremental_state"].pop("api", None)
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    source, _ = IncrementalJavaBuilder(None, data[2], data[4], data[3], data[5], data[1]).render()
    data = (source, *data[1:])
    failed = _java(tmp_path, data[0])
    assert failed.returncode != 0 and "does not override abstract method" in failed.stderr

    class Coder:
        declared = False

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("_override"):
                return {"override_id": "override0"}
            task = json.loads(messages[-1]["content"])
            if task["current_member"]["kind"] == "class" and not self.declared:
                self.declared = True
                return {"op": "override"}
            return {"op": "complete"}
    source, program = _repair(Coder(), data, failed.stderr)
    assert program["members"][-1]["name"] == "run"
    assert _java(tmp_path, source).returncode == 0
