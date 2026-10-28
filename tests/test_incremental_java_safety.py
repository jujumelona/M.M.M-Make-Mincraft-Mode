"""Model-driven Java authoring must not recurse forever without implementing work."""
from __future__ import annotations

import json

import pytest

from test_incremental_java_author import _inputs

from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
from minecraft_mod_ai.incremental_java_model import initial_scope
from minecraft_mod_ai.incremental_java_schema import operation_options


def test_plain_lexical_block_is_not_a_model_action():
    behavior, spec, context, messages = _inputs()
    builder = IncrementalJavaBuilder(None, spec, messages, context, [behavior])
    member = builder.program["members"][0]
    choices = operation_options(member, initial_scope(member), [], set())
    assert "block" not in choices
    assert "synchronized" in choices
    assert {"if", "try", "method", "register", "call"} <= set(choices)


def test_deep_new_helper_chain_is_deferred_instead_of_recursing():
    from minecraft_mod_ai.incremental_java_liveness import _MAX_ACTIVE_MEMBER_DEPTH

    class Coder:
        def __init__(self):
            self.declared = set()
            self.active_depths = []

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            page = task["requested_page"]
            if page == "kind":
                member = task["current_member"]["id"]
                choices = parameters["properties"]["op"]["enum"]
                self.active_depths.append(task["authoring_progress"]["active_helper_depth"])
                if member not in self.declared and "method" in choices:
                    self.declared.add(member)
                    return {"op": "method"}
                return {"op": "complete"}
            if page == "method":
                return {
                    "return_type": "void",
                    "purpose": "distinct helper for " + task["current_member"]["id"],
                    "static": False,
                }
            if page == "parameters_count":
                return {"count": 0}
            raise AssertionError(page)

    behavior, spec, context, messages = _inputs()
    coder = Coder()
    builder = IncrementalJavaBuilder(coder, spec, messages, context, [behavior])
    source, program = builder.finish()
    assert source
    assert max(coder.active_depths) == _MAX_ACTIVE_MEMBER_DEPTH
    assert len(program["members"]) == _MAX_ACTIVE_MEMBER_DEPTH
    assert not builder.active_members
    assert all(member.get("implementation_status") == "complete" for member in program["members"])


def test_declarations_without_an_executable_statement_fail_explicitly():
    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            page = task["requested_page"]
            if page == "kind":
                choices = parameters["properties"]["op"]["enum"]
                if task["current_member"]["id"] == "m1" and "method" in choices:
                    return {"op": "method"}
                return {"op": "complete"}
            if page == "method":
                return {
                    "return_type": "void",
                    "purpose": "helper number " + str(task["accepted_operations_total"]) +
                               " " + str(task["authoring_progress"]["consecutive_declarations_without_statement"]),
                    "static": False,
                }
            if page == "parameters_count":
                return {"count": 0}
            raise AssertionError(page)

    behavior, spec, context, messages = _inputs()
    builder = IncrementalJavaBuilder(Coder(), spec, messages, context, [behavior])
    with pytest.raises(RuntimeError, match="JAVA_INCREMENTAL_DECLARATION_ONLY_STALL"):
        builder.finish()
    assert not builder.active_members
    assert builder.member is None
    assert builder.program["members"][0]["implementation_status"] == "failed"


def test_control_flow_depth_is_constrained_and_scope_restored():
    from minecraft_mod_ai.incremental_java_liveness import _MAX_CONTROL_FLOW_DEPTH

    class Coder:
        def __init__(self):
            self.visited = set()
            self.depths = []

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            page = task["requested_page"]
            if page == "kind":
                path = tuple(task["block_path"])
                depth = task["authoring_progress"]["control_flow_depth"]
                self.depths.append(depth)
                if path not in self.visited and "if" in parameters["properties"]["op"]["enum"]:
                    self.visited.add(path)
                    return {"op": "if"}
                return {"op": "complete"}
            if page == "otherwise":
                return {"enabled": False}
            raise AssertionError(page)

    behavior, spec, context, messages = _inputs()
    coder = Coder()
    builder = IncrementalJavaBuilder(coder, spec, messages, context, [behavior])
    builder.expression = lambda *args, **kwargs: "true"
    source, program = builder.finish()
    assert source.count("if (") >= _MAX_CONTROL_FLOW_DEPTH
    assert max(coder.depths) == _MAX_CONTROL_FLOW_DEPTH
    assert not builder.active_members
    assert len(program["members"][0]["body"]) == 1
