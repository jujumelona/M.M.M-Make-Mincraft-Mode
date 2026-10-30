"""Production-log regressions for a small coder's work ownership and inputs."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from incremental_value_model import structured_values
from test_incremental_java_author import OperationCoder, _compile_claims, _inputs
from test_incremental_java_context import _fixture, _program
from test_incremental_java_recovery import _builder


def _world_builder(coder=None):
    builder = _builder(coder)
    behavior = builder.member["behavior"]
    behavior["runtime_state_reads"] = ["Observed subject"]
    builder.api.bindings["Observed subject"] = {"binding": "world_state"}
    return builder


@pytest.mark.parametrize("plan_location", ["payload", "context"])
def test_work_order_survives_into_operation_context_and_all_pages(plan_location):
    plan = {"feature_id": "f1", "behavior_ids": ["claim"],
            "intents": ["Observe first, then commit the confirmed change"],
            "effects": ["Write the confirmed result"],
            "state_reads": ["canonical" + str(i) for i in range(24)],
            "state_writes": ["balance"],
            "acceptance": {"runtime": "required_behavior_acceptance"}}
    original = deepcopy(plan)
    payload = {"feature_work_order": plan} if plan_location == "payload" else {}
    context = _fixture(payload, **({"feature_work_order": plan} if plan_location == "context" else {}))
    program = _program()
    view = context.view(program["members"][0], {}, program)
    assert view["feature_work_order"]["effects"] == ["Write the confirmed result"]
    assert view["feature_work_order"]["acceptance"] == {"runtime": "required_behavior_acceptance"}
    page = context.work_lookup("@work_order:state_reads", 20)
    assert page["values"] == ["canonical20", "canonical21", "canonical22", "canonical23"]
    assert page["total"] == 24 and page["next_offset"] is None
    assert plan == original


def test_logical_world_names_do_not_hide_actual_state_declarations():
    methods = [{"name": "readState_canonical" + str(i),
                "signature": "public static int readState_canonical" + str(i) + "();"}
               for i in range(24)]
    context = _fixture({"host_state_access": [{"owner": "demo.State", "methods": methods}]})
    program = _program()
    program["members"][0]["behavior"].update(state_reads=[], runtime_state_writes=["Observed subject"])
    view = context.view(program["members"][0], {}, program)
    assert view["host_state_access"][0]["methods"] == methods[:10]
    page = context.work_lookup("@state_access", 20)
    assert page["host_state_access"] == [{"owner": "demo.State", "methods": methods[20:]}]
    assert page["total"] == 24 and page["next_offset"] is None


def test_work_evidence_lookups_are_host_pages_not_broad_api_search():
    from minecraft_mod_ai.incremental_java_history import lookup_work

    builder = _builder(None)
    builder.api.work_order = {"state_reads": ["balance"]}
    page = lookup_work(builder, "@work_order:state_reads", 0)
    assert page["values"] == ["balance"]


def test_host_lifecycle_registrars_are_not_callable_from_generated_work():
    from minecraft_mod_ai.incremental_java_calls import _generated_methods
    from minecraft_mod_ai.incremental_java_model import method_metadata
    from minecraft_mod_ai.incremental_java_values_catalog import generated_members

    builder = _builder(None)
    sibling = {**deepcopy(builder.member), "id": "m20", "name": "registerSibling"}
    builder.program["members"].append(sibling)
    helper = builder.new_method(owner="Feature", purpose="Apply the effect", behavior={},
                                metadata=method_metadata())
    rows = generated_members(builder, "Feature", "METHOD")
    assert "registerSibling" not in {row["name"] for row in rows}
    assert helper["name"] in {row["name"] for row in rows}
    assert _generated_methods(builder, "Feature", "registerSibling") == []


def test_world_registrar_acquires_real_inputs_before_delegating():
    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            assert data_name.endswith("_kind")
            assert "method" not in parameters["properties"]["op"]["enum"]
            assert "call" not in parameters["properties"]["op"]["enum"]
            assert "complete" not in parameters["properties"]["op"]["enum"]
            assert task["current_work"]["phase"] == "bind_observation"
            raise StopIteration("observed input-first work request")

    builder = _world_builder(Coder())
    with pytest.raises(StopIteration, match="observed input-first"):
        builder.author_block(builder.member, builder.scope, [])


def test_world_callback_local_only_completion_is_redirected_to_real_effect(tmp_path):
    from minecraft_mod_ai.incremental_java_author import compose_incremental_feature

    @structured_values
    class Coder(OperationCoder):
        def __init__(self):
            super().__init__()
            self.steps["m1"] = iter([("register", {})])
            self.steps["m2"] = iter([
                ("field", {"type": "int", "purpose": "Count eligible observations", "static": False, "final": False}),
                ("local", {"type": "int"}),
            ])
            self.corrected = False

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            if (data_name.endswith("_kind") and task["current_member"]["id"] == "m2"
                    and not task["block_path"] and task["accepted_operations_total"] >= 1):
                options = parameters["properties"]["op"]["enum"]
                # Reproduce the actual small coder's premature completion when
                # offered; otherwise implement the concrete observed effect.
                if not self.corrected and "complete" in options:
                    return {"op": "complete"}
                if not self.corrected:
                    self.corrected = True
                    assert "return" not in options
                    return {"op": "if"}
                if any(node["op"] == "assign" for node in task["accepted_operations"]):
                    return {"op": "complete"}
                self.details = {"operator": "+="}
                return {"op": "assign"}
            return super().generate_structured_data(role, messages, data_name=data_name, parameters=parameters)

    behavior, spec, context, messages = _inputs()
    behavior["runtime_state_reads"] = ["Observed subject"]
    context["runtime_state_bindings"] = {"Observed subject": {"binding": "world_state"}}
    coder = Coder()
    source, _ = compose_incremental_feature(coder, spec, messages,
        construction_context=context, behaviors=[behavior])
    assert coder.corrected, "local-only callback was falsely completed"
    _compile_claims(tmp_path, source)


def test_world_bound_supporting_type_layout_can_complete():
    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            return {"op": "complete"}

    builder = _world_builder(Coder())
    builder.member.update(kind="class", registrar=False)
    assert builder.author_block(builder.member, builder.scope, []) == []


def test_world_bound_empty_rejection_branch_can_return():
    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert "return" in parameters["properties"]["op"]["enum"]
            return {"op": "return"}

    builder = _world_builder(Coder())
    builder.member.pop("registrar")
    body = builder.author_block(builder.member, builder.scope, ["n1", "body"])
    assert [node["op"] for node in body] == ["return"]


def test_seeded_world_observer_can_complete_without_duplicate_registration():
    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            assert task["current_work"]["phase"] == "implement_effects"
            return {"op": "complete"}

    builder = _world_builder(Coder())
    original = {"id": "n1", "op": "register", "event_owner": "example.Events",
                "event_field": "ACTION", "handler": "onAction"}
    builder.member["body"] = [original]
    assert builder.author_block(builder.member, builder.scope, [], body=builder.member["body"]) == [original]


def test_inherited_mod_lifecycle_is_not_a_generated_gameplay_call():
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    from minecraft_mod_ai.incremental_java_values_members import _api_page
    from minecraft_mod_ai.incremental_java_values_model import JavaReceiver

    behavior, spec, context, messages = _inputs()
    context["symbols"].update({
        "net.fabricmc.api.ModInitializer": {"kind": "CLASS", "owner": "net.fabricmc.api.ModInitializer", "access": 0x601},
        "net.fabricmc.api.ModInitializer#onInitialize()": {
            "kind": "METHOD", "owner": "net.fabricmc.api.ModInitializer", "name": "onInitialize",
            "return_type": "void", "parameters": [], "access": 0x401},
    })
    builder = IncrementalJavaBuilder(None, spec, messages, context, [behavior])
    builder.member = builder.program["members"][0]
    rows, _ = _api_page(builder, JavaReceiver("this", "Feature"), "METHOD", "", 0,
                        {"nonvoid": False, "writable": False})
    assert "onInitialize" not in {row["name"] for row in rows.values()}


@pytest.mark.parametrize("receiver", ["this", "example.Feature"])
def test_late_lifecycle_call_is_rejected_without_accepting_it(monkeypatch, receiver):
    from minecraft_mod_ai.incremental_java_author import _safe_author_operation
    from minecraft_mod_ai import incremental_java_operations as operations

    builder = _builder(None)
    sibling = {**deepcopy(builder.member), "id": "m20", "name": "registerSibling"}
    builder.program["members"].append(sibling)
    monkeypatch.setattr(operations, "author_operation", lambda *_: {
        "id": "n1", "op": "call", "receiver": receiver, "method": "registerSibling",
        "call_contract": {"owner": "example.Feature", "parameters": []}})
    assert _safe_author_operation(builder, builder.member, "call") is None
    assert builder.rejection["operand_diagnostic"] == "JAVA_HOST_LIFECYCLE_CALL: m20"


def test_late_root_initializer_call_is_not_an_implementation(monkeypatch):
    from minecraft_mod_ai import incremental_java_operations as operations
    from minecraft_mod_ai.incremental_java_author import _safe_author_operation

    builder = _builder(None)
    monkeypatch.setattr(operations, "author_operation", lambda *_: {
        "id": "n1", "op": "call", "receiver": "this", "receiver_type": "Feature",
        "method": "onInitialize", "call_contract": {
            "owner": "net.fabricmc.api.ModInitializer", "parameters": []}})
    assert _safe_author_operation(builder, builder.member, "call") is None


def test_external_same_named_method_is_not_the_host_lifecycle(monkeypatch):
    from minecraft_mod_ai import incremental_java_operations as operations
    from minecraft_mod_ai.incremental_java_author import _safe_author_operation

    builder = _builder(None)
    sibling = {**deepcopy(builder.member), "id": "m20", "name": "registerSibling"}
    builder.program["members"].append(sibling)
    node = {"id": "n1", "op": "call", "receiver": "external.Feature", "method": "registerSibling",
            "call_contract": {"owner": "external.Feature", "parameters": []}}
    monkeypatch.setattr(operations, "author_operation", lambda *_: node)
    assert _safe_author_operation(builder, builder.member, "call") == node


@pytest.mark.parametrize("offset", [-1, "0", True])
def test_evidence_lookup_rejects_invalid_page_offsets(offset):
    context = _fixture()
    with pytest.raises(ValueError, match="JAVA_API_LOOKUP_OFFSET_INVALID"):
        context.work_lookup("@state_access", offset)
