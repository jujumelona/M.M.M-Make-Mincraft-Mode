"""Declarations must lead to executable work rather than an expanding backlog."""
from __future__ import annotations

import json
from incremental_value_model import structured_values

import pytest

from test_incremental_java_author import OperationCoder, _inputs, _compile_claims
from test_incremental_java_recovery import _builder


def test_callback_feedback_reports_that_registration_already_connects_listener(tmp_path):
    from minecraft_mod_ai.incremental_java_author import compose_incremental_feature

    @structured_values
    class Coder(OperationCoder):
        connected = False

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            if data_name.endswith("_kind") and task["current_member"]["id"] == "m1" and task.get("accepted_operations"):
                self.connected = True
                assert task["last_declaration"]["next_action"] == "continue_caller"
            return super().generate_structured_data(role, messages, data_name=data_name, parameters=parameters)

    behavior, spec, context, messages = _inputs()
    coder = Coder()
    source, _ = compose_incremental_feature(coder, spec, messages, construction_context=context, behaviors=[behavior])
    assert coder.connected
    _compile_claims(tmp_path, source)


def test_helper_body_is_authored_before_parent_selects_another_operation(tmp_path):
    from minecraft_mod_ai.incremental_java_author import compose_incremental_feature

    @structured_values
    class Coder(OperationCoder):
        def __init__(self):
            super().__init__()
            self.helper_started = False
            self.parent_turns = 0
            self.steps["m1"] = iter([
                ("field", {"type": "int", "purpose": "Count claims", "static": False, "final": False}),
                ("method", {"return_type": "void", "purpose": "Count eligible identifiers", "static": False}),
                ("register", {}),
            ])
            self.steps["m3"] = iter([("call", {"receiver": "this", "method": "mmmHelper2", "result_type": ""})])

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            if data_name.endswith("_kind"):
                if task["current_member"]["id"] == "m2":
                    self.helper_started = True
                if task["current_member"]["id"] == "m1":
                    self.parent_turns += 1
                    if self.parent_turns == 3:
                        assert self.helper_started, "parent resumed before its new helper was implemented"
                        helper = next(row for row in task["generated_declarations"] if row["id"] == "m2")
                        assert helper["implementation_status"] == "complete"
            if data_name.endswith("_parameters_count"):
                return {"count": 1}
            if data_name.endswith("_parameter"):
                return {"type": "String", "purpose": "Observed identifier"}
            return super().generate_structured_data(role, messages, data_name=data_name, parameters=parameters)

        def value_answer(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            if task["expression_slot"].startswith("argument:"):
                return source_frame["prefix"] + task["scope"]["locals"][0]["name"] + source_frame["suffix"]
            return super().value_answer(role, messages, data_name=data_name, source_frame=source_frame)

    behavior, spec, context, messages = _inputs()
    source, program = compose_incremental_feature(Coder(), spec, messages, construction_context=context, behaviors=[behavior])
    assert len(program["members"]) == 4
    _compile_claims(tmp_path, source)


@pytest.mark.parametrize("repeat_owner", ["m1", "m2"])
def test_repeated_helper_declaration_reuses_body_and_connects_real_callback(tmp_path, repeat_owner):
    from minecraft_mod_ai.incremental_java_author import compose_incremental_feature

    @structured_values
    class Coder(OperationCoder):
        def __init__(self):
            super().__init__()
            self.steps["m1"] = iter([
                ("field", {"type": "int", "purpose": "Count claims", "static": False, "final": False}),
                ("method", {"return_type": "void", "purpose": "Increment the claim count", "static": False}),
                ("register", {}),
            ])
            self.steps["m2"] = iter([("assign", {"operator": "+="})])
            if repeat_owner == "m2":
                self.steps["m1"] = iter([
                    ("field", {"type": "int", "purpose": "Count claims", "static": False, "final": False}),
                    ("method", {"return_type": "void", "purpose": "Increment the claim count", "static": False}),
                    ("register", {}),
                ])
                self.steps["m2"] = iter([
                    ("method", {"return_type": "void", "purpose": "Increment the claim count", "static": False}),
                    ("assign", {"operator": "+="}),
                ])
            self.steps["m3"] = iter([("if", {}), ("call", {"receiver": "this", "method": "mmmHelper2", "result_type": ""})])
            self.reused = False

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            if data_name.endswith("_kind"):
                if (task["current_member"]["id"] == "m1"
                        and task.get("progress_frontier", {}).get("unconnected_member_ids")):
                    self.reused = True
                    assert task["progress_frontier"]["unconnected_member_ids"] == ["m2"]
                    assert "method" not in parameters["properties"]["op"]["enum"]
                if task["current_member"]["id"] == "m2" and task.get("last_declaration"):
                    self.reused = True
                    assert task["last_declaration"]["outcome"] == "active"
                    assert task["last_declaration"]["next_action"] == "implement_current_work"
                if task["current_member"]["id"] == "m3" and task["block_path"]:
                    return {"op": "return"}
                if task["current_member"]["id"] == "m3":
                    assert task["scope"]["locals"], "duplicate zero-argument helper was queued instead of the callback"
                # This also bounds the unfixed reproduction without a hanging test.
                assert task["current_member"]["id"] in {"m1", "m2", "m3"}, "duplicate helper was queued as new code"
            if data_name.endswith("_parameters_count"):
                return {"count": 0}
            return super().generate_structured_data(role, messages, data_name=data_name, parameters=parameters)

    behavior, spec, context, messages = _inputs()
    coder = Coder()
    source, program = compose_incremental_feature(coder, spec, messages, construction_context=context, behaviors=[behavior])
    assert coder.reused
    assert len(program["members"]) == 4
    assert sum(member["name"] == "mmmHelper2" for member in program["members"]) == 1
    _compile_claims(tmp_path, source)


def test_same_signature_with_reversed_parameter_roles_is_distinct_work():
    from minecraft_mod_ai.incremental_java_model import method_metadata

    builder = _builder(None)
    roles = [{"type": "int", "name": "arg0", "purpose": "Minuend"},
             {"type": "int", "name": "arg1", "purpose": "Subtrahend"}]
    reversed_roles = [{**roles[0], "purpose": "Subtrahend"}, {**roles[1], "purpose": "Minuend"}]
    kwargs = {"owner": "Feature", "behavior": builder.member["behavior"], "purpose": "Compute the difference"}
    first = builder.new_method(metadata=method_metadata("int", roles), **kwargs)
    reversed_work = builder.new_method(metadata=method_metadata("int", reversed_roles), **kwargs)
    assert reversed_work is not first
    assert len(builder.program["members"]) == 3


def test_duplicate_override_does_not_prevent_implementing_another_interface_method(tmp_path):
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    from minecraft_mod_ai.incremental_java_model import initial_scope
    from test_incremental_java_dependency_repair import _setup
    from test_incremental_java_render import _java

    _, program, spec, context, messages, behaviors = _setup([])
    context["symbols"].update({
        "sample.TwoMethods": {"kind": "CLASS", "owner": "sample.TwoMethods", "access": 0x601},
        **{"sample.TwoMethods#" + name + "()": {
            "kind": "METHOD", "owner": "sample.TwoMethods", "name": name,
            "return_type": "int", "parameters": [], "access": 0x401}
            for name in ("first", "second")},
    })
    context["incremental_state"].pop("api", None)
    nested = {"id": "nested", "owner": "Demo", "kind": "class", "name": "Implementation", "body": [],
              "purpose": "Implement both interface operations", "behavior": {},
              "metadata": {"visibility": "private", "static": True, "superclass": "", "interfaces": ["sample.TwoMethods"]}}
    program["members"].append(nested)

    @structured_values
    class Coder:
        steps = iter(["override", "override", "override", "complete"])
        overrides = iter(["first", "first", "second"])

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            if data_name.endswith("_override"):
                name = next(self.overrides)
                return {"override_id": next(key for key, row in task["override_candidates"].items() if row["name"] == name)}
            if task["current_member"]["id"] == "nested":
                return {"op": next(self.steps)}
            return {"op": "return"}

        def value_answer(self, role, messages, *, data_name, source_frame):
            name = json.loads(messages[-1]["content"])["current_member"]["name"]
            return source_frame["prefix"] + ("1" if name == "first" else "2") + source_frame["suffix"]

    builder = IncrementalJavaBuilder(Coder(), spec, messages, context, behaviors, program)
    nested = builder.program["members"][-1]
    builder.author_block(nested, initial_scope(nested), [], body=nested["body"])
    source, program = builder.render()
    assert {member["name"] for member in program["members"] if member["owner"] == "Implementation"} == {"first", "second"}
    result = _java(tmp_path, source, "interface TwoMethods { int first(); int second(); }")
    assert result.returncode == 0, result.stderr


def test_equivalent_helper_requests_share_identity_but_distinct_work_does_not():
    from minecraft_mod_ai.incremental_java_model import method_metadata

    builder = _builder(None)
    metadata = method_metadata()
    kwargs = {"owner": "Feature", "behavior": builder.member["behavior"], "metadata": metadata}
    first = builder.new_method(purpose="Increment the count", **kwargs)
    repeated = builder.new_method(purpose="  increment   THE count  ", **kwargs)
    different = builder.new_method(purpose="Reset the count", **kwargs)
    assert repeated is first
    assert different is not first
    assert [member["purpose"] for member in builder.program["members"][1:]] == ["Increment the count", "Reset the count"]


def test_completed_child_restores_parent_scope_and_draft_stack():
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    from minecraft_mod_ai.incremental_java_model import initial_scope

    @structured_values
    class Coder:
        steps = {"m1": iter(["method", "local", "complete"]), "m2": iter(["complete"])}

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            member = task["current_member"]["id"]
            if data_name.endswith("_kind"):
                if member == "m2":
                    assert task["operation_stack"] == []
                    assert task["scope"]["locals"] == []
                return {"op": next(self.steps[member])}
            if data_name.endswith("_method"):
                return {"return_type": "void", "purpose": "Prepare state", "static": True}
            if data_name.endswith("_parameters_count"):
                return {"count": 0}
            return {"type": "int"}

        def value_answer(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            assert task["current_member"]["id"] == "m1"
            assert task["scope"]["locals"] == [{"type": "int", "name": "parentValue"}]
            assert task["operation_stack"][0]["id"] == "parentGuard"
            return source_frame["prefix"] + "parentValue" + source_frame["suffix"]

    behavior, spec, context, messages = _inputs()
    builder = IncrementalJavaBuilder(Coder(), spec, messages, context, [behavior])
    member = builder.program["members"][0]
    scope = initial_scope(member)
    scope["locals"].append({"type": "int", "name": "parentValue"})
    builder.drafts = [{"id": "parentGuard", "op": "if", "condition": "true"}]
    builder.frontier_unconnected = ["previousHelper"]
    builder.frontier_feedback = {"state": "outer_block"}
    body = builder.author_block(member, scope, ["parentGuard", "body"])
    assert body[0]["value"] == "parentValue"
    assert builder.drafts == [{"id": "parentGuard", "op": "if", "condition": "true"}]
    assert builder.frontier_unconnected == ["previousHelper"]
    assert builder.frontier_feedback == {"state": "outer_block"}


def test_frontier_rejects_unconnected_distinct_helpers_but_accepts_connected_growth():
    """No operation-count cap: code growth must consume the previous dependency."""
    from types import SimpleNamespace
    from minecraft_mod_ai.incremental_java_progress import frontier_options, helper_delegation_allowed

    current = {"id": "root", "kind": "method", "implementation_status": "authoring",
               "body": [{"id": "n1", "op": "call", "method": "existingMethod", "arguments": []}]}
    helper = {"id": "child", "kind": "method", "name": "generatedHelper",
              "implementation_status": "complete", "body": []}
    builder = SimpleNamespace(program={"members": [current, helper]}, member=current)
    choices = ["method", "call", "if", "complete"]
    blocked = frontier_options(builder, current, current["body"], choices, {"child"})
    assert blocked == ["call", "if", "complete"]
    assert not helper_delegation_allowed(builder)
    assert builder.frontier_feedback["unconnected_member_ids"] == ["child"]
    assert builder.frontier_feedback["state"] == "implementation_required"

    # The same function can create arbitrarily many *connected* helpers.
    # Nested calls are real references and clear the dependency.
    current["body"].append({
        "id": "n2", "op": "if", "condition": "true",
        "body": [{"id": "n3", "op": "call", "method": "generatedHelper", "arguments": []}],
    })
    assert frontier_options(builder, current, current["body"], choices, {"child"}) == choices
    assert helper_delegation_allowed(builder)
    assert builder.frontier_feedback == {}


def test_frontier_prevents_unimplemented_recursive_helper_chain():
    from types import SimpleNamespace
    from minecraft_mod_ai.incremental_java_progress import frontier_options, helper_delegation_allowed

    parent = {"id": "m_parent", "kind": "method", "implementation_status": "authoring",
              "body": []}
    child = {"id": "m_child", "kind": "method", "implementation_status": "authoring",
             "body": []}
    builder = SimpleNamespace(program={"members": [parent, child]}, member=child)
    choices = ["method", "call", "assign", "complete"]
    assert frontier_options(builder, child, child["body"], choices, set()) == [
        "call", "assign", "complete"
    ]
    assert not helper_delegation_allowed(builder)
    assert builder.frontier_feedback["waiting_ancestor_id"] == "m_parent"

    # Merely declaring locals or returning void is not an effect.
    parent["body"].extend([{"op": "local", "name": "unused", "value": "1"},
                           {"op": "return", "value": ""}])
    assert "method" not in frontier_options(builder, child, child["body"], choices, set())

    parent["body"].append({"op": "call", "method": "verifiedOperation", "arguments": []})
    assert frontier_options(builder, child, child["body"], choices, set()) == choices
    assert helper_delegation_allowed(builder)


def test_class_layout_and_initial_registrar_can_declare_members():
    from types import SimpleNamespace
    from minecraft_mod_ai.incremental_java_progress import frontier_options

    registrar = {"id": "m1", "kind": "method", "registrar": True,
                 "implementation_status": "authoring", "body": []}
    layout = {"id": "t1", "kind": "class", "implementation_status": "authoring",
              "body": []}
    builder = SimpleNamespace(program={"members": [registrar, layout]}, member=layout)
    choices = ["method", "field", "type", "complete"]
    assert frontier_options(builder, layout, [], choices, {"unused"}) == choices
    builder.member = registrar
    assert frontier_options(builder, registrar, [], choices, set()) == choices


def test_helper_only_call_cycles_do_not_count_as_implementation():
    from types import SimpleNamespace
    from minecraft_mod_ai.incremental_java_progress import frontier_options

    caller = {"id": "a", "name": "helperA", "kind": "method",
              "implementation_status": "authoring",
              "body": [{"op": "call", "method": "helperB", "arguments": []}]}
    callee = {"id": "b", "name": "helperB", "kind": "method",
              "implementation_status": "complete",
              "body": [{"op": "call", "method": "helperA", "arguments": []}]}
    child = {"id": "c", "name": "helperC", "kind": "method",
             "implementation_status": "authoring", "body": []}
    builder = SimpleNamespace(program={"members": [caller, callee, child]}, member=child)
    assert "method" not in frontier_options(builder, child, [], ["method", "call", "complete"], set())
    assert builder.frontier_feedback["waiting_ancestor_id"] == "a"

    # A real leaf makes the existing delegation productive.
    callee["body"].append({"op": "assign", "target": "this.value", "value": "2"})
    assert "method" in frontier_options(builder, child, [], ["method", "call", "complete"], set())


def test_helper_name_in_string_literal_is_not_a_call_reference():
    from types import SimpleNamespace
    from minecraft_mod_ai.incremental_java_progress import frontier_options

    owner = {"id": "parent", "kind": "method", "implementation_status": "authoring",
             "body": [{"op": "local", "name": "label", "value": '"helperTarget("'}]}
    child = {"id": "helper", "kind": "method", "name": "helperTarget",
             "implementation_status": "complete", "body": []}
    builder = SimpleNamespace(program={"members": [owner, child]}, member=owner)
    # The parser must not take an unrelated literal as a completed dependency.
    assert "method" not in frontier_options(builder, owner, owner["body"], ["method", "call"], {"helper"})
