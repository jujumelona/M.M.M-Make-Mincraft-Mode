"""Accepted work survives local metadata mistakes and dependency corrections."""
from __future__ import annotations

import json

from test_incremental_java_author import OperationCoder, _inputs, _compile_claims


def _builder(coder):
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    from minecraft_mod_ai.incremental_java_model import initial_scope
    behavior, spec, context, messages = _inputs()
    builder = IncrementalJavaBuilder(coder, spec, messages, context, [behavior])
    builder.member = builder.program["members"][0]
    builder.scope = initial_scope(builder.member)
    return builder


def test_invalid_call_head_is_corrected_locally_before_authoring_arguments():
    from minecraft_mod_ai.incremental_java_operations import author_operation
    class Coder:
        calls = 0

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            if data_name.endswith("_call"):
                self.calls += 1
                if self.calls == 2:
                    assert "operand_diagnostic" in task
                    assert task["rejected_operand_page"]["receiver"] == "this; escaped()"
                return {"receiver": "this; escaped()" if self.calls == 1 else "this",
                        "method": "helper", "result_type": ""}
            return {"count": 0}
    coder = Coder()
    node = author_operation(_builder(coder), "call")
    assert coder.calls == 2
    assert node["receiver"] == "this" and node["arguments"] == []


def test_void_parameter_is_corrected_as_type_metadata_instead_of_becoming_source():
    class Coder:
        types = iter(["void", "String"])

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("parameters_count"):
                assert json.loads(messages[-1]["content"])["requested_page"] == "parameters_count"
                return {"count": 1}
            value = next(self.types)
            task = json.loads(messages[-1]["content"])
            if value == "String":
                assert "void" in task["operand_diagnostic"]
            return {"type": value, "purpose": "World identity"}
    assert _builder(Coder()).parameters() == [{"type": "String", "name": "arg0", "purpose": "World identity"}]


def test_cancelled_event_selection_preserves_accepted_local_and_returns_to_authoring():
    class Coder:
        steps = iter(["local", "register", "complete"])

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("_kind"):
                return {"op": next(self.steps)}
            if data_name.endswith("_register"):
                return {"event_id": "@cancel"}
            return {"type": "int"}

        def generate_java_source(self, role, messages, *, data_name, source_frame):
            return source_frame["prefix"] + "3" + source_frame["suffix"]
    builder = _builder(Coder())
    body = builder.author_block(builder.member, builder.scope, [])
    assert len(body) == 1 and body[0]["op"] == "local" and body[0]["value"] == "3"
    assert builder.scope["locals"] == [{"name": body[0]["name"], "type": "int"}]
    assert len(builder.program["members"]) == 1


def test_new_helper_is_declared_called_and_implemented_without_existing_gameplay_code(tmp_path):
    from minecraft_mod_ai.incremental_java_author import compose_incremental_feature
    class HelperCoder(OperationCoder):
        def __init__(self):
            super().__init__()
            self.steps["m1"] = iter([
                ("field", {"type": "int", "purpose": "Count eligible claims", "static": False, "final": False}),
                ("method", {"return_type": "void", "purpose": "Count only nonempty identifiers", "static": False}),
                ("register", {})])
            self.steps["m3"] = iter([("call", {"receiver": "this", "method": "mmmHelper2", "result_type": ""})])

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("parameters_count"):
                return {"count": 1}
            if data_name.endswith("_parameter"):
                return {"type": "String", "purpose": "Observed identifier"}
            return super().generate_structured_data(role, messages, data_name=data_name, parameters=parameters)

        def generate_java_source(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            if task["expression_slot"].startswith("argument:"):
                assert task["operation_stack"][-1]["call_contract"]["parameters"][0]["type"] == "String"
                return source_frame["prefix"] + task["scope"]["locals"][0]["name"] + source_frame["suffix"]
            return super().generate_java_source(role, messages, data_name=data_name, source_frame=source_frame)
    behavior, spec, context, messages = _inputs()
    coder = HelperCoder()
    source, program = compose_incremental_feature(coder, spec, messages, construction_context=context, behaviors=[behavior])
    assert any(member["name"] == "mmmHelper2" and member["body"] for member in program["members"])
    _compile_claims(tmp_path, source)


def test_event_selection_can_reach_late_pages_without_sending_the_catalog():
    from minecraft_mod_ai.incremental_java_events import select_event
    class Coder:
        pages = 0

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            assert len(json.dumps(task)) < 12000
            if data_name.endswith("register_lookup"):
                return {"query": "*", "offset": 2980}
            assert len(task["event_index"]) == 10
            self.pages += 1
            return {"event_id": "@next" if self.pages == 1 else task["event_index"][-1]["event_id"]}
    coder = Coder()
    builder = _builder(coder)
    event = builder.api.callbacks()[0]
    builder.api._callbacks = [{**event, "event_id": f"example.Events#ACTION_{index:04}"} for index in range(3000)]
    assert select_event(builder)["event_id"] == "example.Events#ACTION_2999"
    assert coder.pages == 2


def test_large_parent_override_catalog_is_paged_until_the_last_method_is_selected():
    from minecraft_mod_ai.incremental_java_declarations import _select_override
    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            assert len(json.dumps(task)) < 16000
            assert len(task["override_candidates"]) == 10
            return {"override_id": "@next" if task["override_offset"] < 490 else "override499"}
    rows = [{"kind": "METHOD", "owner": "Parent", "name": f"method{index}", "parameters": [],
             "return_type": "void", "access": 1} for index in range(500)]
    assert _select_override(_builder(Coder()), rows)["name"] == "method499"
