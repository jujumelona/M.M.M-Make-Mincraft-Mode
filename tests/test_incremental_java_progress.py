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
                if task["current_member"]["id"] == "m1" and task.get("last_declaration", {}).get("outcome") == "reused":
                    self.reused = True
                    assert task["last_declaration"]["declaration"]["name"] == "mmmHelper2"
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
    body = builder.author_block(member, scope, ["parentGuard", "body"])
    assert body[0]["value"] == "parentValue"
    assert builder.drafts == [{"id": "parentGuard", "op": "if", "condition": "true"}]
