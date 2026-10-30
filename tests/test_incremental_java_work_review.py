"""Valid implementations and evidence pages must survive work-phase recovery."""
from __future__ import annotations

import json
import subprocess

import pytest
from jsonschema import Draft202012Validator

from incremental_value_model import structured_values
from test_incremental_java_author import OperationCoder, _compile_claims, _inputs
from test_incremental_java_context import _fixture, _program
from test_incremental_java_work_ownership import _world_builder


def test_large_state_selection_can_follow_its_advertised_lookup_pages():
    from minecraft_mod_ai.incremental_java_schema import SCHEMAS

    names = ["canonical_storage_" + str(i) for i in range(24)]
    methods = [{"name": "readState_" + name,
                "signature": "public static int readState_" + name + "();"} for name in names]
    context = _fixture({"host_state_access": [{"owner": "demo.State", "methods": methods}]})
    program = _program()
    program["members"][0]["behavior"]["state_reads"] = names
    view = context.view(program["members"][0], {}, program)
    page = view["host_state_access_page"]
    seen = list(view["host_state_access"][0]["methods"])
    while page["next_offset"] is not None:
        request = {"query": page["query"], "offset": page["next_offset"]}
        Draft202012Validator(SCHEMAS["lookup"]).validate(request)
        page = context.work_lookup(**request)
        seen.extend(page["host_state_access"][0]["methods"])
    assert seen == methods


def test_fallback_state_headers_do_not_grant_behavior_call_permission():
    from minecraft_mod_ai.host_typed_state_access import validate_behavior_state_provenance

    context = _fixture({"host_state_access": [{"owner": "demo.State", "methods": [
        {"name": "writeState_unrelated", "signature": "public static void writeState_unrelated(int value);"}]}]})
    program = _program()
    behavior = program["members"][0]["behavior"]
    behavior.update(state_reads=[], state_writes=[], runtime_state_writes=["Observed subject"])
    view = context.view(program["members"][0], {}, program)
    assert view["state_access_permissions"]["mode"] == "declaration_discovery"
    assert view["state_access_permissions"]["writable"] == []
    with pytest.raises(ValueError, match="JAVA_BEHAVIOR_STATE_PROVENANCE_MISMATCH"):
        validate_behavior_state_provenance("void apply() { demo.State.writeState_unrelated(1); }",
                                           context.state_access, behavior)


@pytest.mark.parametrize("receiver_type", ["net.fabricmc.api.ModInitializer", "Child"])
def test_initializer_upcast_subclass_and_method_reference_are_not_callees(receiver_type):
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    from minecraft_mod_ai.incremental_java_values_members import _api_page
    from minecraft_mod_ai.incremental_java_values_model import JavaReceiver
    from minecraft_mod_ai.incremental_java_work import lifecycle_call_diagnostic

    behavior, spec, context, messages = _inputs()
    owner = "net.fabricmc.api.ModInitializer"
    context["symbols"].update({
        owner: {"kind": "CLASS", "owner": owner, "access": 0x601},
        owner + "#onInitialize()": {"kind": "METHOD", "owner": owner, "name": "onInitialize",
                                     "return_type": "void", "parameters": [], "access": 0x401},
    })
    builder = IncrementalJavaBuilder(None, spec, messages, context, [behavior])
    builder.member = builder.program["members"][0]
    builder.program["members"].append({"id": "child", "owner": "Feature", "name": "Child",
        "kind": "class", "metadata": {"superclass": "Feature", "interfaces": [], "static": True}, "body": []})
    for reference in (False, True):
        receiver = JavaReceiver("initializer", receiver_type, unbound=reference)
        rows, _ = _api_page(builder, receiver, "METHOD", "", 0, {"nonvoid": False, "writable": False})
        assert "onInitialize" not in {row["name"] for row in rows.values()}
    assert lifecycle_call_diagnostic(builder, {"op": "call", "receiver": "initializer",
        "receiver_type": receiver_type, "method": "onInitialize",
        "call_contract": {"owner": owner, "name": "onInitialize", "parameters": []}})


@pytest.mark.parametrize("node", [
    {"id": "n1", "op": "local", "name": "removed", "type": "boolean", "value": "world.removeEntity(entity)"},
    {"id": "n1", "op": "if", "condition": "world.tryApply()", "body": [], "otherwise": []},
    {"id": "n1", "op": "assign", "target": "removed", "operator": "=", "value": "world.tryApply()"},
])
def test_effectful_expression_can_complete_and_counts_as_progress(node):
    from minecraft_mod_ai.incremental_java_work import work_options

    builder = _world_builder()
    builder.member.pop("registrar")
    builder.member["metadata"]["parameters"] = [{"name": "removed", "type": "boolean"}]
    builder.scope["locals"] = [{"name": "removed", "type": "boolean"}]
    options = work_options(builder, builder.member, builder.scope, [], [node], ["call", "complete", "return"])
    assert options == ["call", "complete", "return"]
    builder._record_operation_progress(builder.member, node, builder.scope)
    assert builder._progress_epoch[builder.member["id"]] == 1


def test_pure_local_copy_still_cannot_complete_world_work():
    from minecraft_mod_ai.incremental_java_work import work_options

    builder = _world_builder()
    builder.member.pop("registrar")
    body = [{"op": "local", "name": "copy", "type": "boolean", "value": "false"}]
    assert work_options(builder, builder.member, builder.scope, [], body,
                        ["call", "complete", "return"]) == ["call"]


def test_deferred_lambda_body_is_not_an_executed_effect():
    from minecraft_mod_ai.incremental_java_work import work_options

    builder = _world_builder()
    builder.member.pop("registrar")
    body = [{"op": "lambda", "name": "later", "body": [
        {"op": "call", "receiver": "world", "method": "apply", "arguments": []}]}]
    assert work_options(builder, builder.member, builder.scope, [], body,
                        ["call", "complete", "return"]) == ["call"]


def test_effectful_initializer_is_executed_by_the_generated_listener(tmp_path):
    from minecraft_mod_ai.incremental_java_author import compose_incremental_feature

    @structured_values
    class Coder(OperationCoder):
        def __init__(self):
            super().__init__()
            self.steps = {
                "m1": iter([("register", {})]),
                "m2": iter([
                    ("field", {"type": "int", "purpose": "Count accepted inputs", "static": False, "final": False}),
                    ("method", {"return_type": "boolean", "purpose": "Apply the observed input", "static": False}),
                    ("if", {}), ("local", {"type": "boolean"}),
                ]),
                "m3": iter([("assign", {"operator": "+="}), ("return", {})]),
            }

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("_parameters_count"):
                return {"count": 0}
            return super().generate_structured_data(role, messages, data_name=data_name, parameters=parameters)

        def value_answer(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            if task["current_member"]["id"] == "m2" and task["expression_slot"] == "value":
                return "this.mmmHelper3()"
            if task["current_member"]["id"] == "m3" and task["current_operation"] == "return":
                return "true"
            return super().value_answer(role, messages, data_name=data_name, source_frame=source_frame)

    behavior, spec, context, messages = _inputs()
    behavior["runtime_state_reads"] = ["Observed subject"]
    context["runtime_state_bindings"] = {"Observed subject": {"binding": "world_state"}}
    source, _ = compose_incremental_feature(Coder(), spec, messages, construction_context=context, behaviors=[behavior])
    _compile_claims(tmp_path, source)


def test_supporting_empty_override_is_not_forced_to_invent_gameplay(tmp_path):
    from minecraft_mod_ai.incremental_java_declarations import _override
    from minecraft_mod_ai.incremental_java_progress import finish_member
    from test_incremental_java_render import _java

    @structured_values
    class Coder:
        selections = iter(["override0", "override1"])

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name.endswith("_override"):
                return {"override_id": next(self.selections)}
            task = json.loads(messages[-1]["content"])
            if task["current_member"]["name"] == "active":
                return {"op": "return"}
            return {"op": "complete"}

        def value_answer(self, role, messages, *, data_name, source_frame):
            return "1"

    builder = _world_builder(Coder())
    builder.member = {**builder.member, "id": "adapter", "owner": "Demo", "kind": "class", "name": "Adapter",
        "registrar": False, "metadata": {"visibility": "public", "static": True, "interfaces": ["sample.Hooks"]},
        "purpose": "Support a two-hook observer adapter"}
    builder.program.update(package="sample", class_name="Demo", interface="sample.Lifecycle", members=[builder.member])
    builder.api.override = lambda _: [
        {"kind": "METHOD", "owner": "sample.Hooks", "name": "active", "parameters": [], "return_type": "int", "access": 0x401},
        {"kind": "METHOD", "owner": "sample.Hooks", "name": "idle", "parameters": [], "return_type": "void", "access": 0x401},
    ]
    for _ in range(2):
        method = _override(builder)
        finish_member(builder, method)
    assert method["body"] == [] and method["implementation_status"] == "complete"
    source, _ = builder.render()
    compiled = _java(tmp_path, source, """
interface Hooks { int active(); void idle(); }
class Harness { public static void main(String[] args) {
    var adapter = new Demo.Adapter(); adapter.idle();
    if (adapter.active() != 1) throw new AssertionError("active result");
} }
""")
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(["java", "-cp", str(tmp_path), "sample.Harness"], capture_output=True, text=True, timeout=30)
    assert executed.returncode == 0, executed.stderr
