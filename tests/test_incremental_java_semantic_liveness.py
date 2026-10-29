"""Regression coverage for semantic liveness and recursive Java expression depth."""
from __future__ import annotations

from contextlib import ExitStack
import pytest

from minecraft_mod_ai.incremental_java_author import _semantic_operation_key
from minecraft_mod_ai.incremental_java_values_model import (
    MAX_VALUE_EXPRESSION_NESTING, value_frame,
)
from test_incremental_java_recovery import _builder


def test_local_preparation_does_not_count_as_runtime_effect():
    scope = {"locals": [{"name": "temporary", "type": "int"}]}
    assert _semantic_operation_key(
        {"id": "n1", "op": "local", "name": "temporary", "value": "0"}, scope
    ) is None
    assert _semantic_operation_key(
        {"id": "n2", "op": "assign", "target": "temporary", "value": "1"}, scope
    ) is None
    assert _semantic_operation_key(
        {"id": "n3", "op": "call", "method": "calculate", "result_type": "int"}, scope
    ) is not None
    assert _semantic_operation_key(
        {"id": "n4", "op": "assign", "target": "this.count", "value": "1"}, scope
    ) is not None


def test_repeated_operations_share_semantic_signature_not_node_identity():
    scope = {"locals": []}
    first = {"id": "n1", "op": "call", "receiver": "this",
             "method": "observe", "arguments": [], "result_type": ""}
    second = {**first, "id": "n2147"}
    assert _semantic_operation_key(first, scope) == _semantic_operation_key(second, scope)


def test_local_churn_followed_by_duplicate_call_reaches_stall_guard(monkeypatch):
    from minecraft_mod_ai import incremental_java_operations as operations

    class Coder:
        def __init__(self):
            self.requests = 0

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert data_name.endswith("_kind")
            options = parameters["properties"]["op"]["enum"]
            self.requests += 1
            return {"op": "local" if "local" in options else "call"}

    coder = Coder()
    builder = _builder(coder)

    def fake_operation(builder, operation):
        identity = builder.allocate("n")
        if operation == "local":
            name = "mmmValue" + builder.allocate("v")[1:]
            builder.scope["locals"].append({"name": name, "type": "int"})
            return {"id": identity, "op": "local", "name": name, "value": "0"}
        return {"id": identity, "op": "call", "method": "observe",
                "receiver": "this", "arguments": [], "result_type": ""}

    monkeypatch.setattr(operations, "author_operation", fake_operation)
    with pytest.raises(RuntimeError, match="INCREMENTAL_JAVA_NO_SEMANTIC_PROGRESS"):
        builder.author_block(builder.member, builder.scope, [])
    assert coder.requests <= 130
    assert builder._progress_epoch.get(builder.member["id"], 0) == 1


def test_value_expression_nesting_is_bounded_without_mutating_stack():
    builder = _builder(None)
    builder.value_stack = []
    with ExitStack() as stack:
        for i in range(MAX_VALUE_EXPRESSION_NESTING):
            stack.enter_context(value_frame(builder, "operand:" + str(i), "int"))
        with pytest.raises(RuntimeError, match="JAVA_VALUE_EXPRESSION_NESTING_EXHAUSTED"):
            with value_frame(builder, "one-more", "int"):
                pass
        assert len(builder.value_stack) == MAX_VALUE_EXPRESSION_NESTING
    assert builder.value_stack == []


def test_local_catalog_page_cycle_is_rejected(monkeypatch):
    from minecraft_mod_ai import incremental_java_values as values

    builder = _builder(None)
    builder.scope["locals"] = [
        {"name": "local" + str(i), "type": "int"} for i in range(20)
    ]

    def choose(_builder, schema, suffix, extra=None):
        assert suffix == "local_value"
        return {"name": "next_page" if extra["local_offset"] == 0 else "previous_page"}

    monkeypatch.setattr(values, "request", choose)
    with pytest.raises(RuntimeError, match="JAVA_LOCAL_SELECTION_NAVIGATION_CYCLE"):
        values.ValueAuthor(builder)._local()


def test_member_catalog_page_cycle_is_rejected(monkeypatch):
    from minecraft_mod_ai import incremental_java_values_members as members
    from minecraft_mod_ai.incremental_java_values_model import JavaReceiver

    builder = _builder(None)

    def catalog(_builder, receiver, kind, query, location, filters):
        if location[1] == 0:
            return ({"m1": {"name": "entry"}}, {"total": 20}, ("api", 10))
        return ({"m2": {"name": "entry2"}}, {"total": 20}, None)

    def choose(_builder, schema, suffix, extra=None):
        assert suffix == "field_member"
        return {"member_id": "next_page" if not extra.get("member_page", {}).get("offset") else "previous_page"}

    # Supply the real page offset in the test catalog, so selection alternates.
    def catalog_with_offset(_builder, receiver, kind, query, location, filters):
        rows, page, following = catalog(_builder, receiver, kind, query, location, filters)
        return rows, {**page, "offset": location[1]}, following

    monkeypatch.setattr(members, "_member_page", catalog_with_offset)
    monkeypatch.setattr(members, "request", choose)
    with pytest.raises(RuntimeError, match="JAVA_MEMBER_SELECTION_NAVIGATION_CYCLE"):
        members.select_member(builder, JavaReceiver("this", "Feature"), "FIELD", "field_member")


def test_repair_discovery_cannot_replace_a_concrete_edit_forever():
    from minecraft_mod_ai.incremental_java_repair import _select_repair

    class Builder:
        lookups = 0

        def ask(self, schema, suffix):
            options = schema["properties"]["repair"]["enum"]
            return {"repair": "lookup" if "lookup" in options else "mapped_operand"}

        def lookup(self):
            self.lookups += 1

    builder = Builder()
    assert _select_repair(builder, ["mapped_operand"]) == "mapped_operand"
    assert builder.lookups == 2


def test_inserted_repair_operations_share_original_error_budget():
    from minecraft_mod_ai.incremental_java_repair import _count_attempt

    builder = _builder(None)
    member = builder.member
    _count_attempt(builder, member, {"id": "n1"}, "value")
    _count_attempt(builder, member, {"id": "n2", "_repair_origin": "n1"}, "value")
    with pytest.raises(ValueError, match="JAVA_INCREMENTAL_OPERATION_REPAIR_EXHAUSTED"):
        _count_attempt(builder, member, {"id": "n3", "_repair_origin": "n1"}, "value")



def test_unimplemented_registrar_is_mapped_to_a_repairable_source_diagnostic():
    from minecraft_mod_ai.incremental_java_pipeline import verify_full_source

    builder = _builder(None)
    source, program = builder.render()
    builder.context["incremental_state"]["program"] = program

    with pytest.raises(ValueError, match="CONSTRUCTION_FILE_BEHAVIOR_NOOP"):
        verify_full_source(
            source, builder.spec, builder.context,
            package=program["package"], name=program["class_name"],
            verifier=lambda *args, **kwargs: "verified",
            state_validators=(lambda *args: None, lambda *args: None),
        )


def test_no_effect_callback_mapping_uses_member_identity_not_guessed_event():
    from minecraft_mod_ai.incremental_java_state import _repair_entrypoint

    callback = {"id": "m2", "event": {"event_id": "known"},
                "behavior": {"behavior_id": "damage_effect"}}
    selected, diagnostic = _repair_entrypoint(
        {"members": [callback]},
        "INCREMENTAL_JAVA_NO_OBSERVABLE_ACTION: entrypoint=m2; no effects",
    )
    assert selected is callback
    assert diagnostic == "CONSTRUCTION_FILE_BEHAVIOR_NOOP: damage_effect"


def test_navigation_tracker_allows_legal_backtracking_but_detects_cycle():
    from minecraft_mod_ai.incremental_java_navigation import record_navigation_transition

    history = {}
    moves = [("page0", "next"), ("page1", "previous"), ("page0", "previous"),
             ("generated", "next"), ("page0", "search")]
    for state, action in moves:
        record_navigation_transition(history, state, action, error="CYCLE")
    assert history[("page0", "next")] == 1
    # The first matching transition in 'moves' was already recorded.
    record_navigation_transition(history, "page0", "next", error="CYCLE")
    with pytest.raises(RuntimeError, match="CYCLE"):
        record_navigation_transition(history, "page0", "next", error="CYCLE")



def test_captured_calls_remain_potential_effects_without_fresh_signatures():
    # Minecraft mutators such as block-placement methods may return a boolean.
    scope = {"locals": []}
    first = {"id": "n1", "op": "call", "receiver": "world", "method": "setBlock",
             "arguments": ["pos", "state"], "result_type": "boolean",
             "name": "mmmValue1"}
    second = {**first, "id": "n2", "name": "mmmValue947"}
    assert _semantic_operation_key(first, scope) == _semantic_operation_key(second, scope)
    assert _semantic_operation_key(first, scope) is not None
    distinct = {**first, "arguments": ["otherPos", "state"]}
    assert _semantic_operation_key(distinct, scope) != _semantic_operation_key(first, scope)


def test_duplicate_registration_does_not_reset_semantic_progress():
    scope = {"locals": []}
    first = {"id": "n1", "op": "register", "event_owner": "demo.Events",
             "event_field": "CLICK", "handler": "mmmHelper1"}
    second = {**first, "id": "n2", "handler": "mmmHelper2"}
    assert _semantic_operation_key(first, scope) == _semantic_operation_key(second, scope)


def test_repair_cannot_spin_when_event_registration_is_unavailable(monkeypatch):
    from minecraft_mod_ai import incremental_java_operations as operations
    from minecraft_mod_ai.incremental_java_repair_slots import _replacement_operation

    calls = []

    def unavailable(_builder, operation):
        calls.append(operation)
        return None

    monkeypatch.setattr(operations, "author_operation", unavailable)
    with pytest.raises(ValueError, match="JAVA_INCREMENTAL_REPAIR_OPERATION_UNAVAILABLE"):
        _replacement_operation(object(), "register")
    assert calls == ["register", "register"]


def test_register_event_search_cycle_is_rejected():
    from minecraft_mod_ai.incremental_java_events import select_event

    class API:
        def callbacks(self):
            return [{"event_id": "demo.Events#CLICK",
                     "parameters": ["java.lang.String"]}]

    class Builder:
        api = API()
        def ask(self, _schema, suffix, _extra=None):
            if suffix == "register":
                return {"event_id": "@search"}
            assert suffix == "register_lookup"
            return {"query": "*", "offset": 0}

    with pytest.raises(RuntimeError, match="JAVA_EVENT_SELECTION_NAVIGATION_CYCLE"):
        select_event(Builder())


def test_constructor_reference_reselection_is_not_recursive(monkeypatch):
    from minecraft_mod_ai import incremental_java_values as values

    class Builder:
        value_stack = [{"slot": "reference", "expected_type": "java.lang.Object"}]
        def select_java_type(self, *_args, **_kwargs):
            return "demo.Target"

    monkeypatch.setattr(values, "request",
                        lambda _builder, _schema, _suffix: {"reference_kind": "constructor"})
    monkeypatch.setattr(values, "select_member",
                        lambda *_args, **_kwargs: "change_receiver")
    with pytest.raises(RuntimeError, match="JAVA_REFERENCE_CONSTRUCTOR_SELECTION_CYCLE"):
        values.ValueAuthor(Builder())._reference("java.util.function.Supplier")



def test_type_lookup_cannot_spin_indefinitely():
    from minecraft_mod_ai.incremental_java_type_choices import select_java_type

    builder = _builder(None)
    calls = []
    builder.lookup = lambda: calls.append("lookup")

    def choose(schema, _suffix, _extra=None):
        options = schema["properties"]["type"]["enum"]
        return {"type": "@lookup" if "@lookup" in options else "int"}

    builder.ask = choose
    assert select_java_type(builder, "field_type") == "int"
    assert calls == ["lookup", "lookup"]
    assert builder.type_stack == []


def test_type_construction_depth_guard_keeps_original_stack():
    from minecraft_mod_ai.incremental_java_type_choices import (
        select_java_type, MAX_TYPE_EXPRESSION_NESTING,
    )

    builder = _builder(None)
    builder.type_stack = [{"purpose": str(index)}
                          for index in range(MAX_TYPE_EXPRESSION_NESTING)]
    original = list(builder.type_stack)
    with pytest.raises(RuntimeError, match="JAVA_TYPE_EXPRESSION_NESTING_EXHAUSTED"):
        select_java_type(builder, "nested_type")
    assert builder.type_stack == original


def test_override_catalog_backtracking_cycle_is_bounded():
    from minecraft_mod_ai.incremental_java_declarations import _select_override

    class Builder:
        def ask(self, _schema, _suffix, extra=None):
            return {"override_id": "@next" if extra["override_offset"] == 0 else "@previous"}

    rows = [{"name": "method" + str(i), "kind": "METHOD"} for i in range(20)]
    with pytest.raises(RuntimeError, match="JAVA_OVERRIDE_SELECTION_NAVIGATION_CYCLE"):
        _select_override(Builder(), rows)


def test_repair_anchor_catalog_backtracking_cycle_is_bounded():
    from minecraft_mod_ai.incremental_java_edits import _select_anchor

    class Builder:
        def ask(self, _schema, _suffix, extra=None):
            return {"anchor_id": "@next" if extra["anchor_offset"] == 0 else "@previous"}

    nodes = [{"id": "n" + str(i), "op": "call"} for i in range(20)]
    with pytest.raises(RuntimeError, match="JAVA_REPAIR_ANCHOR_NAVIGATION_CYCLE"):
        _select_anchor(Builder(), nodes)


def test_repair_related_declaration_pagination_cycle_is_bounded():
    from minecraft_mod_ai.incremental_java_repair_slots import repair_related_declaration

    class Builder:
        def __init__(self):
            self.program = {"members": [
                {"id": "m" + str(i), "name": "M" + str(i), "kind": "method",
                 "owner": "Feature", "purpose": "test",
                 "metadata": {"return_type": "int"}} for i in range(20)
            ]}
        def ask(self, _schema, _suffix, extra=None):
            return {"declaration_id": "@next"
                    if extra["declaration_offset"] == 0 else "@previous"}

    builder = Builder()
    with pytest.raises(RuntimeError, match="JAVA_REPAIR_DECLARATION_NAVIGATION_CYCLE"):
        repair_related_declaration(builder, builder.program["members"][0])



def test_repeated_switch_cases_have_finite_structural_limit():
    from minecraft_mod_ai.incremental_java_operations import _switch

    class Builder:
        path = []
        scope = {"switch_depth": 0, "locals": []}
        def enabled(self, _suffix):
            return True
        def expression(self, _slot):
            return "1"
        def author_block(self, _member, _scope, _path):
            return []
        member = {"id": "m1", "kind": "method"}

    with pytest.raises(RuntimeError, match="JAVA_STRUCTURED_SECTION_EXHAUSTED: switch_cases"):
        _switch(Builder(), {"id": "n1", "op": "switch"})


def test_repeated_catch_clauses_have_finite_structural_limit():
    from minecraft_mod_ai.incremental_java_operations import _try

    class Builder:
        path = []
        scope = {"switch_depth": 0, "locals": []}
        member = {"id": "m1", "kind": "method"}
        def enabled(self, _suffix):
            return True
        def typed(self, *_args, **_kwargs):
            return {"type": "java.lang.Exception"}
        def allocate(self, _prefix):
            return "e1"
        def author_block(self, _member, _scope, _path):
            return []

    with pytest.raises(RuntimeError, match="JAVA_STRUCTURED_SECTION_EXHAUSTED: catch_clauses"):
        _try(Builder(), {"id": "n1", "op": "try"})



def test_type_component_count_is_bounded_by_schema():
    from minecraft_mod_ai.incremental_java_type_choices import (
        MAX_TYPE_COMPONENTS, _count,
    )

    class Builder:
        def ask(self, schema, suffix):
            assert suffix == "test_type_count"
            constraint = schema["properties"]["count"]
            assert constraint["minimum"] == 2
            assert constraint["maximum"] == MAX_TYPE_COMPONENTS
            return {"count": 2}

    assert _count(Builder(), "test_type_count", minimum=2,
                  maximum=MAX_TYPE_COMPONENTS) == 2



def test_nested_helper_chain_is_bounded_without_limiting_sibling_count():
    from minecraft_mod_ai.incremental_java_progress import (
        MAX_HELPER_DEPENDENCY_DEPTH, _helper_dependency_frame, finish_member,
    )

    class Builder:
        _active_member_depth = MAX_HELPER_DEPENDENCY_DEPTH

    builder = Builder()
    with pytest.raises(RuntimeError, match="JAVA_HELPER_DEPENDENCY_NESTING_EXHAUSTED"):
        finish_member(builder, {"id": "m999"})
    assert builder._active_member_depth == MAX_HELPER_DEPENDENCY_DEPTH

    # Sequential sibling helper completion is not part of the nesting budget.
    builder._active_member_depth = 0
    for _ in range(100):
        with _helper_dependency_frame(builder):
            assert builder._active_member_depth == 1
        assert builder._active_member_depth == 0



def test_callback_local_only_assignments_cannot_masquerade_as_gameplay_effects():
    from minecraft_mod_ai.incremental_java_author import _assert_live_behavior_entrypoints

    local_only = {"id": "m2", "event": {"event_id": "example.Events#ACTION"},
                  "metadata": {"parameters": [{"name": "observed", "type": "String"}]},
                  "body": [
                      {"id": "n1", "op": "local", "name": "mmmValue1",
                       "type": "int", "value": "0"},
                      {"id": "n2", "op": "assign", "operator": "=",
                       "target": "mmmValue1", "value": "7"},
                      {"id": "n3", "op": "assign", "operator": "=",
                       "target": "observed", "value": "null"},
                  ]}
    with pytest.raises(RuntimeError, match="INCREMENTAL_JAVA_NO_OBSERVABLE_ACTION"):
        _assert_live_behavior_entrypoints({"members": [local_only]})

    real_effect = {**local_only,
                   "body": [*local_only["body"],
                            {"id": "n4", "op": "assign", "operator": "+=",
                             "target": "this.mmmField1", "value": "1"}]}
    _assert_live_behavior_entrypoints({"members": [real_effect]})
