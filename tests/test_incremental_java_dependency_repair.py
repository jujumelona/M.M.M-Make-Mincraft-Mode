"""Compiler corrections can complete nested blocks and introduce dependencies."""
from __future__ import annotations

import json

from test_incremental_java_author import _inputs
from test_incremental_java_render import _program, _method, _java


def _setup(body):
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    behavior, spec, context, messages = _inputs()
    spec["target_path"] = "src/main/java/sample/Demo.java"
    spec["bindings"].update(class_name="Demo", package_name="sample")
    program = _program([_method("work", body, registrar=True)])
    builder = IncrementalJavaBuilder(None, spec, messages, context, [behavior], program)
    source, _ = builder.render()
    return source, program, spec, context, messages, [behavior]


def _repair(coder, data, diagnostic):
    from minecraft_mod_ai.incremental_java_repair import repair_incremental_program
    source, program, spec, context, messages, behaviors = data
    return repair_incremental_program(coder, source, diagnostic, program=program, spec=spec,
                                     construction_context=context, messages=messages, behaviors=behaviors)


def test_missing_lambda_return_is_added_inside_the_lambda_without_replacing_its_work(tmp_path):
    body = [{"id": "lambda", "op": "lambda", "type": "java.util.function.IntSupplier",
             "name": "supplier", "return_type": "int", "parameters": [],
             "body": [{"id": "local", "op": "local", "type": "int", "name": "amount", "value": "7"}]}]
    data = _setup(body)
    failed = _java(tmp_path, data[0])
    assert failed.returncode != 0 and "missing return value" in failed.stderr

    class Coder:
        def generate_java_source(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            assert {"name": "amount", "type": "int"} in task["scope"]["locals"]
            assert task["scope"]["return_type"] == "int"
            return source_frame["prefix"] + "amount" + source_frame["suffix"]
    source, program = _repair(Coder(), data, failed.stderr)
    assert program["members"][0]["body"][0]["body"][0] == body[0]["body"][0]
    assert _java(tmp_path, source).returncode == 0


def test_closure_capture_repair_can_insert_a_snapshot_before_the_enclosing_lambda(tmp_path):
    body = [{"id": "local", "op": "local", "type": "int", "name": "mmmValue1", "value": "7"},
            {"id": "lambda", "op": "lambda", "type": "Runnable", "name": "listener", "parameters": [],
             "return_type": "void", "body": [{"id": "print", "op": "call", "receiver": "System.out",
              "method": "println", "arguments": ["mmmValue1"], "result_type": ""}]},
            {"id": "mutate", "op": "assign", "target": "mmmValue1", "operator": "+=", "value": "1"}]
    data = _setup(body)
    failed = _java(tmp_path, data[0])
    assert failed.returncode != 0 and "effectively final" in failed.stderr

    class Coder:
        inserting = True

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("repair_target"):
                return {"repair": "insert_before" if self.inserting else "mapped_operand"}
            if data_name.endswith("repair_insert_location"):
                return {"before_node": "lambda"}
            if data_name.endswith("repair_insert_kind"):
                return {"op": "local"}
            return {"type": "int"}

        def generate_java_source(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            if self.inserting:
                assert task["scope"]["locals"] == [{"type": "int", "name": "mmmValue1"}]
            else:
                assert any(v["name"] == "mmmValue2" for v in task["scope"]["locals"])
            return source_frame["prefix"] + ("mmmValue1" if self.inserting else "mmmValue2") + source_frame["suffix"]
    coder = Coder()
    source, program = _repair(coder, data, failed.stderr)
    assert program["members"][0]["body"][1]["value"] == "mmmValue1"
    assert program["members"][0]["body"][1]["name"] == "mmmValue2"
    failed = _java(tmp_path, source)
    assert failed.returncode != 0
    coder.inserting = False
    source, program = _repair(coder, (source, program, *data[2:]), failed.stderr)
    assert program["members"][0]["body"][-1] == body[-1]
    assert _java(tmp_path, source).returncode == 0


def test_unreachable_call_can_be_moved_before_the_guard_while_retaining_all_effects(tmp_path):
    body = [{"id": "guard", "op": "if", "condition": "System.nanoTime() > 0",
             "body": [{"id": "yes", "op": "return", "value": ""}],
             "otherwise": [{"id": "no", "op": "return", "value": ""}]},
            {"id": "print", "op": "call", "receiver": "System.out", "method": "println", "arguments": ["1"], "result_type": ""}]
    data = _setup(body)
    failed = _java(tmp_path, data[0])
    assert failed.returncode != 0 and "unreachable statement" in failed.stderr

    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("repair_target"):
                return {"repair": "relocate"}
            if data_name.endswith("repair_move_anchor"):
                return {"anchor_id": "guard"}
            return {"position": "before"}
    source, program = _repair(Coder(), data, failed.stderr)
    assert program["members"][0]["body"] == [body[1], body[0]]
    assert _java(tmp_path, source).returncode == 0
