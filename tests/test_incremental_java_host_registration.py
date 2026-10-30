"""Regression: inspected source hooks are host work, not a 9B menu loop.

The production-log 13 failure performed 96 semantic nonprogress decisions,
including repeated event cursor/cancel navigation without any Java operation.
These tests do not hard-code a gameplay request or trigger vocabulary.
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
from minecraft_mod_ai.incremental_java_events import select_original_occurrence
from minecraft_mod_ai.incremental_java_operations import _register
from minecraft_mod_ai.source_occurrence_error import SourceHookUnavailable


def _event():
    return {
        "event_id": "inspected.events.Observed#EVENT",
        "event_owner": "inspected.events.Observed",
        "event_field": "EVENT",
        "parameters": ["java.lang.Object"],
        "return_type": "void",
    }


def test_explicit_original_hook_is_bound_without_model_inference():
    event = _event()

    class NoModel:
        def generate_structured_data(self, *args, **kwargs):
            pytest.fail("explicit, checked ABI binding must not consult the AI")

    builder = SimpleNamespace(
        member={"behavior": {
            "behavior_id": "b-original", "event_binding": event["event_id"],
            "trigger": "a custom runtime occurrence",
        }},
        api=SimpleNamespace(callbacks=lambda: [event]),
        router=NoModel(),
    )
    assert select_original_occurrence(builder)["event_id"] == event["event_id"]


def test_absent_observation_reports_producer_without_model_loop():
    class NoModel:
        def generate_structured_data(self, *args, **kwargs):
            pytest.fail("missing ABI is an upstream producer task, not a model menu")

    builder = SimpleNamespace(
        member={"behavior": {"behavior_id": "b-missing"}},
        api=SimpleNamespace(callbacks=lambda: []),
        router=NoModel(),
    )
    with pytest.raises(SourceHookUnavailable, match="no_inspected_callback_requires_source_producer"):
        select_original_occurrence(builder)


def test_ambiguous_hook_is_one_bounded_source_decision_not_event_paging():
    event = _event()

    class DecliningModel:
        calls = 0

        def generate_structured_data(self, role, messages, **kwargs):
            self.calls += 1
            assert role == "planner"
            assert "SOURCE_REQUIRED" in kwargs["parameters"]["properties"]["event_id"]["enum"]
            return {"event_id": "SOURCE_REQUIRED"}

    router = DecliningModel()
    builder = SimpleNamespace(
        member={"behavior": {"behavior_id": "b-unknown", "trigger": "new runtime interaction"}},
        api=SimpleNamespace(callbacks=lambda: [event] * 13),
        router=router,
    )
    with pytest.raises(SourceHookUnavailable, match="custom_runtime_hook_needed"):
        select_original_occurrence(builder)
    assert router.calls == 1


def _builder_without_model():
    builder = object.__new__(IncrementalJavaBuilder)
    builder.member = None
    builder.context = {"host_registration_enabled": True}
    builder.scope = {}
    builder.path = []
    builder.block = []
    builder.current_operation = ""
    builder.pending = deque()
    builder.history = {}
    builder._progress_epoch = {}
    builder._nonprogress_choices = {}
    builder._semantic_operation_keys = {}
    return builder


def test_world_registrar_is_rendered_once_by_host_without_kind_or_event_menu(monkeypatch):
    from minecraft_mod_ai import incremental_java_author as author
    from minecraft_mod_ai import incremental_java_progress as progress
    from minecraft_mod_ai import incremental_java_work as work

    builder = _builder_without_model()
    behavior = {"behavior_id": "b-runtime", "runtime_state_writes": ["world.currency"]}
    member = {"id": "m1", "kind": "method", "registrar": True,
              "behavior": behavior, "body": []}
    builder.allocate = lambda prefix: "n1"
    builder.api = SimpleNamespace(bindings={})

    calls = []
    def materialize(current, node_owner, operation):
        assert current is builder
        assert node_owner is member
        assert operation == "register"
        calls.append(operation)
        return {"id": "n1", "op": "register", "event_owner": "inspected.Event",
                "event_field": "EVENT", "handler": "handler1"}

    def finish(current, pending_before):
        assert current is builder
        assert pending_before == 0

    monkeypatch.setattr(author, "_safe_author_operation", materialize)
    monkeypatch.setattr(progress, "finish_new_members", finish)
    monkeypatch.setattr(work, "_world_bound", lambda b, m: True)

    def forbidden(*args, **kwargs):
        pytest.fail("registrar construction should not consume a model choice")

    builder.ask = forbidden
    assert builder.author_block(member, {}, [], body=member["body"]) is member["body"]
    assert len(calls) == 1
    assert len(member["body"]) == 1
    assert builder.author_block(member, {}, [], body=member["body"]) is member["body"]
    assert len(calls) == 1


def test_registrar_rejects_duplicate_source_hooks_before_ai_turn(monkeypatch):
    from minecraft_mod_ai import incremental_java_work as work
    builder = _builder_without_model()
    member = {"id": "m2", "kind": "method", "registrar": True, "behavior": {},
              "body": [{"id": "n1", "op": "register"},
                       {"id": "n2", "op": "register"}]}
    monkeypatch.setattr(work, "_world_bound", lambda b, m: True)
    with pytest.raises(RuntimeError, match="JAVA_DUPLICATE_ORIGINAL_OCCURRENCE_REGISTRATION"):
        builder.author_block(member, {}, [], body=member["body"])


def test_incremental_registration_uses_host_binding_instead_of_menu(monkeypatch):
    from minecraft_mod_ai import incremental_java_events as events

    chosen = _event()
    requests = []
    monkeypatch.setattr(events, "select_original_occurrence", lambda builder: (
        requests.append(builder.member["behavior"]["behavior_id"]) or chosen))
    class Builder:
        context = {"host_registration_enabled": True}
        member = {"owner": "Owner", "behavior": {"behavior_id": "b-1"}}
        def new_method(self, **kwargs):
            return {"name": "listener", "metadata": kwargs["metadata"]}
    node = {"id": "n1", "op": "register"}
    _register(Builder(), node)
    assert requests == ["b-1"]
    assert node["event_owner"] == chosen["event_owner"]
    assert node["handler"] == "listener"


def test_semantic_stall_switches_to_host_owned_compiler_once(monkeypatch):
    from minecraft_mod_ai import incremental_java_author as author
    from minecraft_mod_ai.incremental_java_pipeline import compose_source

    calls = []
    def stalled(*args, **kwargs):
        calls.append("incremental")
        raise RuntimeError(
            "INCREMENTAL_JAVA_NO_SEMANTIC_PROGRESS: member=m1; 96 choices")
    monkeypatch.setattr(author, "compose_incremental_feature", stalled)

    def host_first(router, spec, messages, *, source_context, construction_context, behaviors):
        calls.append("source_owned")
        assert construction_context["host_source_owned"] is True
        assert construction_context["host_direct_feature_source"] is False
        assert construction_context["incremental_state"]["strategy"] == "host_source_owned"
        return "package example; class HostComposed {}"

    context = {"host_direct_feature_source": True, "incremental_state": {}}
    result = compose_source(
        object(), {}, [], source_context=[], construction_context=context,
        behaviors=[{"behavior_id": "b1"}], legacy_author=host_first,
    )
    assert result == "package example; class HostComposed {}"
    assert calls == ["incremental", "source_owned"]
    assert "program" not in context["incremental_state"]


def test_missing_source_producer_uses_host_owned_compiler_once(monkeypatch):
    from minecraft_mod_ai import incremental_java_author as author
    from minecraft_mod_ai.incremental_java_pipeline import compose_source

    monkeypatch.setattr(author, "compose_incremental_feature", lambda *a, **k: (
        (_ for _ in ()).throw(SourceHookUnavailable(
            "b2", reason="no_matching_original_occurrence_callback",
            inspected=14,
        ))))
    invocations = []
    def source_owned(*args, **kwargs):
        invocations.append(kwargs["construction_context"]["host_source_owned"])
        return "host-output"
    state = {"host_direct_feature_source": True, "incremental_state": {}}
    assert compose_source(
        object(), {}, [], source_context=[], construction_context=state,
        behaviors=[], legacy_author=source_owned,
    ) == "host-output"
    assert invocations == [True]


def test_nonsemantic_failures_do_not_trigger_fake_success_fallback(monkeypatch):
    from minecraft_mod_ai import incremental_java_author as author
    from minecraft_mod_ai.incremental_java_pipeline import compose_source

    def failed(*args, **kwargs):
        raise RuntimeError("CONSTRUCTION_TARGET_CLASSPATH_REQUIRED")
    monkeypatch.setattr(author, "compose_incremental_feature", failed)

    def forbidden(*args, **kwargs):
        pytest.fail("a missing compiler classpath is not a semantic stall")

    with pytest.raises(RuntimeError, match="CONSTRUCTION_TARGET_CLASSPATH_REQUIRED"):
        compose_source(
            object(), {}, [], source_context=[],
            construction_context={"host_direct_feature_source": True,
                                  "incremental_state": {}},
            behaviors=[], legacy_author=forbidden,
        )


def test_host_fallback_preserves_every_runtime_source_and_write_obligation(monkeypatch):
    from minecraft_mod_ai.host_source_strategy import prepare_source_owned_strategy
    from minecraft_mod_ai import host_fabric_callback_abi

    monkeypatch.setattr(
        host_fabric_callback_abi, "callback_interfaces", lambda *a, **k: [_event()])
    behavior = {
        "behavior_id": "b-independent",
        "intent": "non-template gameplay",
        "trigger": "original occurrence",
        "effects": "original effect",
        "runtime_state_reads": ["world.sensor", "world.sensor"],
        "runtime_state_writes": ["world.currency", "world.inventory"],
    }
    result = prepare_source_owned_strategy(
        {"symbols": {}}, [behavior], side="COMMON")
    assert result["host_source_owned"] is True
    assert result["host_direct_feature_source"] is False
    graph = result["source_owned_work_graph"]
    nodes = graph["tasks"]
    reads = [n for n in nodes if n["kind"] == "runtime_producer"]
    writes = [n for n in nodes if n["kind"] == "runtime_mutation"]
    effects = [n for n in nodes if n["kind"] == "gameplay_effect"]
    assert len(reads) == 1 and reads[0]["name"] == "world.sensor"
    assert {n["name"] for n in writes} == {"world.currency", "world.inventory"}
    assert len(effects) == 1
    assert {n["task_id"] for n in (*reads, *writes)} <= set(effects[0]["depends_on"])
    assert graph["completion_verified"] is False


def test_source_owned_rejected_candidate_replays_its_native_write_proof():
    from minecraft_mod_ai.incremental_java_pipeline import candidate_data, load_candidate

    proof = [{"behavior_id": "b1", "state_name": "world.currency",
              "verified_signature": "world.Owner#addCurrency(int)",
              "statement": "owner.addCurrency(1);"}]
    candidate = candidate_data("source", "compile error", {
        "strategy": "host_source_owned", "native_mutation_receipts": proof,
    })
    restored = {}
    assert load_candidate(candidate, restored, True) == candidate
    assert restored["strategy"] == "host_source_owned"
    assert restored["native_mutation_receipts"] == proof
    assert "program" not in restored


def test_source_owned_resume_refuses_missing_or_corrupted_native_receipts():
    from minecraft_mod_ai.incremental_java_pipeline import load_candidate

    for record in (
        {"source": "code", "diagnostic": "compiler", "strategy": "host_source_owned"},
        {"source": "code", "diagnostic": "compiler", "strategy": "host_source_owned",
         "native_mutation_receipts_json": "{}"},
    ):
        with pytest.raises(ValueError, match="JAVA_HOST_SOURCE_NATIVE_PROOF"):
            load_candidate(record, {}, True)


def test_constrained_runtime_feature_enters_host_engine_before_first_model_choice(monkeypatch):
    from minecraft_mod_ai import incremental_java_author as author
    from minecraft_mod_ai import host_fabric_callback_abi
    from minecraft_mod_ai.incremental_java_pipeline import compose_source

    monkeypatch.setattr(host_fabric_callback_abi, "callback_interfaces",
                        lambda *a, **k: [_event()])
    monkeypatch.setattr(
        author, "compose_incremental_feature",
        lambda *a, **k: pytest.fail("runtime feature must not enter 96-choice AI loop"),
    )
    behavior = {"behavior_id": "general-world-state",
                "trigger": "original world occurrence", "effects": "mutate original state",
                "runtime_state_reads": ["world.resource"],
                "runtime_state_writes": ["world.stock"]}
    observations = []
    def render(router, spec, messages, *, source_context, construction_context, behaviors):
        observations.append(construction_context["source_owned_work_graph"])
        return "host-built-source"
    state = {"host_direct_feature_source": True,
             "prefer_host_source_owned": True,
             "symbols": {}, "incremental_state": {}}
    result = compose_source(
        object(), {"side": "COMMON"}, [], source_context=[],
        construction_context=state, behaviors=[behavior], legacy_author=render,
    )
    assert result == "host-built-source"
    assert len(observations) == 1
    assert state["incremental_state"]["strategy"] == "host_source_owned"
    assert {n["kind"] for n in observations[0]["tasks"]} >= {
        "occurrence_hook", "runtime_producer", "runtime_mutation",
        "gameplay_effect", "event_registration", "target_compile",
    }


def test_nonlocal_fixture_uses_original_author_unless_explicitly_requested():
    from minecraft_mod_ai.host_source_strategy import prefer_source_owned_strategy

    behavior = [{"runtime_state_writes": ["world.money"]}]
    assert prefer_source_owned_strategy(object(), {}, behavior) is False
    assert prefer_source_owned_strategy(
        object(), {"prefer_host_source_owned": True}, behavior) is True
    assert prefer_source_owned_strategy(
        object(), {"prefer_host_source_owned": False}, behavior) is False
    with pytest.raises(ValueError, match="HOST_SOURCE_PREFERENCE_INVALID"):
        prefer_source_owned_strategy(object(), {"prefer_host_source_owned": "true"}, behavior)


def test_no_inspected_event_creates_original_source_before_listener(monkeypatch):
    from minecraft_mod_ai.host_source_strategy import prepare_source_owned_strategy
    from minecraft_mod_ai import host_fabric_callback_abi

    monkeypatch.setattr(host_fabric_callback_abi, "callback_interfaces",
                        lambda *args, **kwargs: [])
    graph = prepare_source_owned_strategy(
        {"symbols": {}}, [{
            "behavior_id": "b-unimplemented",
            "trigger": "original custom world occurrence",
            "intent": "new mechanic", "effects": "change gameplay",
        }], side="COMMON")["source_owned_work_graph"]
    source = next(n for n in graph["tasks"] if n["kind"] == "custom_event_source")
    hook = next(n for n in graph["tasks"] if n["kind"] == "occurrence_hook")
    effect = next(n for n in graph["tasks"] if n["kind"] == "gameplay_effect")
    assert source["task_id"] in hook["depends_on"]
    assert hook["task_id"] in effect["depends_on"]
    assert graph["completion_verified"] is False


def test_host_registration_applies_to_stateless_features_too(monkeypatch):
    from minecraft_mod_ai import incremental_java_author as author
    from minecraft_mod_ai import incremental_java_progress as progress

    builder = _builder_without_model()
    member = {"id": "m-stateless", "kind": "method", "registrar": True,
              "behavior": {"behavior_id": "basic-event"}, "body": []}
    calls = []
    def emit(builder_arg, member_arg, operation):
        calls.append(operation)
        return {"id": "node", "op": "register", "event_owner": "real.Event",
                "event_field": "EVENT", "handler": "handle"}
    monkeypatch.setattr(author, "_safe_author_operation", emit)
    monkeypatch.setattr(progress, "finish_new_members", lambda *a, **k: None)
    builder.ask = lambda *a, **k: pytest.fail("no model action for a lifecycle registrar")
    builder.author_block(member, {}, [], body=member["body"])
    assert calls == ["register"]
    assert [n["op"] for n in member["body"]] == ["register"]


def test_live_llama_adapter_defaults_to_source_owned_graph_behind_router_proxy():
    from minecraft_mod_ai.host_source_strategy import prefer_source_owned_strategy
    from minecraft_mod_ai.incremental_java_registration_policy import host_registration_enabled

    class Registry:
        def role(self, profile, role):
            assert (profile, role) == ("local", "coder")
            return SimpleNamespace(adapter="llama_cpp")
    direct = SimpleNamespace(profile="local", registry=Registry())
    class Proxy:
        def __getattr__(self, name):
            return getattr(direct, name)
    proxy = Proxy()
    runtime = [{"behavior_id": "b1", "runtime_state_reads": ["world.location"]}]
    assert prefer_source_owned_strategy(proxy, {}, runtime) is True
    assert host_registration_enabled(SimpleNamespace(router=proxy, context={})) is True


def test_host_source_without_checked_fabric_index_is_never_marked_compiled():
    from minecraft_mod_ai.construction_file_author import (
        _verify_full_source, ConstructionFileError,
    )

    source = "package example; public class Feature { void implementation() {} }"
    spec = {"target_path": "src/main/java/example/Feature.java",
            "bindings": {"request_feature_entrypoint": True}}
    context = {"java_classpath": ["/fake"], "symbols": {},
               "host_source_owned": True,
               "host_direct_feature_source": False}
    with pytest.raises(ConstructionFileError, match="SOURCE_OWNED_INSPECTED_ABI_REQUIRED"):
        _verify_full_source(source, spec, context, package="example", name="Feature")


@pytest.mark.parametrize("strategy,valid", [
    ("host_source_owned", False),
    ("host_verified_source", False),
    ("host_rule_template", False),
    (None, False),
    ("incremental_java", True),
])
def test_only_real_ast_strategy_can_use_unbounded_incremental_repairs(strategy, valid):
    from minecraft_mod_ai.incremental_java_pipeline import has_incremental_ast
    state = {"strategy": strategy}
    if valid:
        state["program"] = {"schema": "mmm/incremental-java-v1", "members": []}
    assert has_incremental_ast(True, state) is valid


def test_ast_strategy_requires_a_saved_program_not_just_a_route_label():
    from minecraft_mod_ai.incremental_java_pipeline import has_incremental_ast
    with pytest.raises(ValueError, match="JAVA_HOST_INCREMENTAL_AST_MISSING"):
        has_incremental_ast(True, {"strategy": "incremental_java"})


@pytest.mark.parametrize("strategy", ["host_verified_source", "host_rule_template"])
def test_compiler_resume_preserves_host_source_strategy_without_ast(strategy):
    from minecraft_mod_ai.incremental_java_pipeline import candidate_data, load_candidate
    record = candidate_data(
        "package example; class Feature {}", "javac: expected symbol",
        {"strategy": strategy},
    )
    restored = {}
    assert load_candidate(record, restored, True) == record
    assert restored == {"strategy": strategy}
    assert "program_json" not in record


def test_compiler_resume_rejects_native_proof_on_non_ast_host_strategy():
    from minecraft_mod_ai.incremental_java_pipeline import load_candidate
    with pytest.raises(ValueError, match="JAVA_HOST_NON_AST_CANDIDATE_INVALID"):
        load_candidate({
            "source": "code", "diagnostic": "error",
            "strategy": "host_verified_source",
            "native_mutation_receipts_json": "[]",
        }, {}, True)


def test_existing_declared_state_read_is_reused_without_duplicate_runtime_producer(monkeypatch):
    from minecraft_mod_ai.host_source_strategy import prepare_source_owned_strategy
    from minecraft_mod_ai import host_fabric_callback_abi
    monkeypatch.setattr(host_fabric_callback_abi, "callback_interfaces",
                        lambda *a, **k: [_event()])
    context = {
        "symbols": {},
        "host_state_work_order_scope": {"declared_readable": ["credits"]},
        "host_state_access": [{"owner": "example.AuthoredStateModel",
            "methods": [{"name": "readState_credits",
                         "signature": "public static int readState_credits() { return 1; }"}]}],
    }
    behavior = {"behavior_id": "state-reuse", "intent": "apply source-backed behavior",
                "trigger": "checked callback", "effects": "update state",
                "state_reads": ["credits"],
                "runtime_state_reads": ["credits", "world.missing"]}
    result = prepare_source_owned_strategy(context, [behavior], side="COMMON")
    assert result["source_owned_reused_reads"]["state-reuse"]["credits"]["method"] == "readState_credits"
    producers = [r["name"] for r in result["source_owned_work_graph"]["tasks"]
                 if r["kind"] == "runtime_producer"]
    assert producers == ["world.missing"]


def test_unverified_state_label_is_not_a_callable_binding(monkeypatch):
    from minecraft_mod_ai.host_source_strategy import prepare_source_owned_strategy
    from minecraft_mod_ai import host_fabric_callback_abi
    monkeypatch.setattr(host_fabric_callback_abi, "callback_interfaces",
                        lambda *a, **k: [_event()])
    behavior = {"behavior_id": "unbound", "intent": "new state mechanic",
                "state_reads": ["credits"], "runtime_state_reads": ["credits"]}
    result = prepare_source_owned_strategy({
        "symbols": {},
        "host_state_work_order_scope": {"declared_readable": ["credits"]},
        "host_state_access": [],
    }, [behavior], side="COMMON")
    assert result["source_owned_reused_reads"] == {}
    assert any(row["kind"] == "runtime_producer" and row["name"] == "credits"
               for row in result["source_owned_work_graph"]["tasks"])


def test_identical_compiler_repair_is_rejected_before_another_model_turn():
    from minecraft_mod_ai.incremental_java_pipeline import require_compiler_repair_progress
    with pytest.raises(ValueError, match="JAVA_HOST_COMPILER_REPAIR_NO_PROGRESS"):
        require_compiler_repair_progress("class A {}", "class A {}", "cannot resolve symbol")
    with pytest.raises(ValueError, match="JAVA_HOST_COMPILER_REPAIR_EMPTY"):
        require_compiler_repair_progress("class A {}", "", "cannot resolve symbol")
    assert require_compiler_repair_progress("class A {}", "class A { int n; }", "error")


def test_source_owned_strategy_does_not_replay_conflicting_direct_author_prompt(monkeypatch):
    import json
    from minecraft_mod_ai.incremental_java_pipeline import compose_source
    from minecraft_mod_ai import host_fabric_callback_abi
    monkeypatch.setattr(host_fabric_callback_abi, "callback_interfaces",
                        lambda *a, **k: [_event()])
    behavior = {"behavior_id": "b1", "intent": "check original condition",
                "runtime_state_reads": ["world.condition"]}
    original = {"host_java_authoring": True,
                "assigned_behaviors": [behavior],
                "inspected_target_api": {"target_api": [{"signature": "verified"}]},
                "implementation_requirements": "preserve world state"}
    payload = [{"role": "user", "content": json.dumps(original)}]
    def capture(_router, _spec, messages, **kwargs):
        actual = json.loads(messages[0]["content"])
        assert actual["host_java_authoring"] is False
        assert actual["host_source_owned"] is True
        assert actual["assigned_behaviors"] == original["assigned_behaviors"]
        assert actual["inspected_target_api"] == original["inspected_target_api"]
        assert actual["implementation_requirements"] == original["implementation_requirements"]
        assert kwargs["construction_context"]["host_source_owned"] is True
        return "checked"
    assert compose_source(
        object(), {"side": "COMMON"}, payload, source_context=[],
        construction_context={"host_direct_feature_source": True,
                              "prefer_host_source_owned": True,
                              "incremental_state": {}, "symbols": {}},
        behaviors=[behavior], legacy_author=capture,
    ) == "checked"
    assert json.loads(payload[0]["content"]) == original


def test_custom_source_is_added_after_original_event_inventory_proved_irrelevant():
    from minecraft_mod_ai.feature_source_work_graph import (
        build_source_work_graph, bind_planned_custom_occurrence, bind_selected_hooks,
    )
    original = {"behavior_id": "b-transition",
                "intent": "record original dimension arrival",
                "trigger": "player enters a new dimension",
                "effects": "record arrival"}
    unrelated = {**_event(), "event_id": "example.PlayerDamage#AFTER_DAMAGE"}
    graph = build_source_work_graph(
        [original], producer_obligations=[], inspected_hooks=[unrelated])
    assert not any(r["kind"] == "custom_event_source" for r in graph["tasks"])
    planned = {"source_descriptor": "(Lnet/minecraft/server/level/ServerPlayer;)V",
               "files": {"src/main/java/example/Trigger.java": "source"},
               "callback": {**_event(), "event_id": "example.Trigger#EVENT"}}
    amended = bind_planned_custom_occurrence(
        graph, "b-transition", planned, "inspected:original-join")
    source = next(r for r in amended["tasks"] if r["kind"] == "custom_event_source")
    hook = next(r for r in amended["tasks"] if r["kind"] == "occurrence_hook")
    assert source["task_id"] in hook["depends_on"]
    assert source["status"] == "planned"
    assert source["evidence"]["runtime_verified"] is False
    bound = bind_selected_hooks(amended, {"b-transition": planned["callback"]})
    assert any(r["kind"] == "occurrence_hook" and r["status"] == "source_selected"
               for r in bound["tasks"])
    assert bound["completion_verified"] is False


def test_host_graph_flags_missing_semantic_occurrence_even_if_some_events_exist(monkeypatch):
    from minecraft_mod_ai.host_source_strategy import prepare_source_owned_strategy
    from minecraft_mod_ai import host_fabric_callback_abi
    monkeypatch.setattr(host_fabric_callback_abi, "callback_interfaces",
                        lambda *args, **kwargs: [_event()])
    behavior = {"behavior_id": "b-transition",
                "trigger": "player enters a new dimension",
                "intent": "handle transition"}
    graph = prepare_source_owned_strategy({"symbols": {}}, [behavior],
                                          side="COMMON")["source_owned_work_graph"]
    source = next(r for r in graph["tasks"] if r["kind"] == "custom_event_source")
    hook = next(r for r in graph["tasks"] if r["kind"] == "occurrence_hook")
    assert source["task_id"] in hook["depends_on"]


def test_state_reader_with_unbound_context_parameter_is_not_reused():
    from minecraft_mod_ai.host_source_strategy import _materialized_read_bindings
    context = {
        "host_state_work_order_scope": {"declared_readable": ["credits"]},
        "host_state_access": [{"owner": "example.AuthoredStateModel", "methods": [
            {"name": "readState_credits",
             "signature": "public static int readState_credits(String player) { return 1; }"},
        ]}],
    }
    assert _materialized_read_bindings(context) == {}


def test_compiler_no_progress_detects_comment_and_formatting_only_rewrites():
    from minecraft_mod_ai.incremental_java_pipeline import require_compiler_repair_progress
    before = "package example; public class A { void go() { int x = 1; } }"
    after = "package example;\npublic class A {\n // changed comment\nvoid go(){int x=1;}\n}"
    with pytest.raises(ValueError, match="JAVA_HOST_COMPILER_REPAIR_TOKEN_NO_PROGRESS"):
        require_compiler_repair_progress(before, after, "cannot find symbol")
    actual = "package example; public class A { void go() { int x = 2; } }"
    assert require_compiler_repair_progress(before, actual, "cannot find symbol") == actual


def test_local_member_prompt_keeps_host_verified_canonical_reader_contract():
    import json
    from minecraft_mod_ai.construction_java_context import scoped_java_member_messages
    frame = {"kind": "method", "prefix": "private boolean check(){", "suffix": "}"}
    reader = {"credits": {"owner": "example.AuthoredStateModel",
                         "method": "readState_credits",
                         "signature": "public static int readState_credits() { return 1; }"}}
    messages = [{"role": "user", "content": json.dumps({
        "response_source_frame": frame, "host_java_authoring": False,
        "member_role": "test_callback_condition",
        "verified_existing_runtime_reads": reader,
        "assigned_behavior": {"behavior_id": "claim"},
    })}]
    result = scoped_java_member_messages(messages, frame)
    parsed = json.loads(result[-1]["content"])
    assert parsed["verified_existing_runtime_reads"] == reader
    assert parsed["member_role"] == "test_callback_condition"


def test_all_live_local_features_use_host_owned_engine_even_without_state_labels():
    from minecraft_mod_ai.host_source_strategy import prefer_source_owned_strategy
    class Registry:
        def role(self, profile, role):
            assert role == "coder"
            return SimpleNamespace(adapter="llama_cpp")
    router = SimpleNamespace(registry=Registry(), profile="local")
    behavior = {"behavior_id": "stateless-feature", "intent": "real feature",
                "trigger": "real callback", "effects": "real operation"}
    assert prefer_source_owned_strategy(router, {}, [behavior]) is True
    assert prefer_source_owned_strategy(router, {}, []) is False
    assert prefer_source_owned_strategy(
        router, {"prefer_host_source_owned": False}, [behavior]) is False


def test_cycle_A_B_A_rejected_before_third_source_repair():
    from minecraft_mod_ai.incremental_java_pipeline import record_rejected_compiler_source
    checked = set()
    a = "package example; public class A { void test() {} }"
    b = "package example; public class A { int n; void test() {} }"
    assert record_rejected_compiler_source(a, checked)
    assert record_rejected_compiler_source(b, checked)
    with pytest.raises(ValueError, match="JAVA_HOST_COMPILER_SOURCE_CYCLE"):
        record_rejected_compiler_source(a, checked)
    assert len(checked) == 2


def test_incremental_compiler_repair_cap_is_finite_with_separate_legacy_budget():
    from minecraft_mod_ai.incremental_java_pipeline import (
        MAX_INCREMENTAL_COMPILER_ATTEMPTS, compile_attempts, exhausted,
    )
    steps = list(compile_attempts(True, 2))
    assert len(steps) == MAX_INCREMENTAL_COMPILER_ATTEMPTS == 64
    assert not exhausted(True, steps[-2], 2)
    assert exhausted(True, steps[-1], 2)
    assert list(compile_attempts(False, 2)) == [0, 1, 2]
    assert not exhausted(False, 1, 2)
    assert exhausted(False, 2, 2)


def test_missing_verifier_behavior_list_is_rebound_from_real_assigned_work():
    import json
    from minecraft_mod_ai.construction_host_source import _exact_behavior_verification_spec
    original = {"bindings": {"request_feature_entrypoint": True}, "requirement": "new"}
    expected = [{"behavior_id": "b1", "intent": "feature", "trigger": "observation"}]
    checked = _exact_behavior_verification_spec(original, expected)
    assert json.loads(checked["bindings"]["construction_behaviors_json"]) == expected
    assert "construction_behaviors_json" not in original["bindings"]


def test_stale_feature_verifier_scope_cannot_be_silently_reused():
    import json
    from minecraft_mod_ai.construction_host_source import (
        _exact_behavior_verification_spec, HostJavaConstructionError,
    )
    original = {"bindings": {
        "request_feature_entrypoint": True,
        "construction_behaviors_json": json.dumps([{"behavior_id": "old"}]),
    }}
    with pytest.raises(HostJavaConstructionError, match="SCOPE_MISMATCH"):
        _exact_behavior_verification_spec(original, [{"behavior_id": "new"}])
    with pytest.raises(HostJavaConstructionError, match="BEHAVIOR_ID_SET_INVALID"):
        _exact_behavior_verification_spec(
            {"bindings": {"request_feature_entrypoint": True}},
            [{"behavior_id": "repeat"}, {"behavior_id": "repeat"}])


def test_non_host_ast_fixture_can_author_state_before_original_event_registration():
    from minecraft_mod_ai.incremental_java_work import work_options
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    from test_incremental_java_author import _inputs, OperationCoder
    behavior, spec, context, messages = _inputs()
    builder = IncrementalJavaBuilder(
        OperationCoder(), spec, messages, context, [behavior])
    registrar = next(m for m in builder.program["members"] if m.get("registrar"))
    base = ["field", "register", "complete", "lookup"]
    assert work_options(builder, registrar, {}, [], [], base) == base
    context["host_registration_enabled"] = True
    assert work_options(builder, registrar, {}, [], [], base) == ["register"]
