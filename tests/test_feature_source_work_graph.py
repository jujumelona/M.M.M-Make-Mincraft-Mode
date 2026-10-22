"""The source route cannot call an event registration an implementation.

Every original behavior owns an occurrence, its unsatisfied runtime
observations, a gameplay effect, a registration, and verification. Unknown
state names never become fabricated Java getters merely by planning them.
"""
import pytest

from minecraft_mod_ai.feature_source_work_graph import (
    build_source_work_graph, member_source_work, bind_selected_hooks,
)


def test_producer_hook_effect_registration_compile_playtest_is_a_dag():
    graph = build_source_work_graph(
        [{"behavior_id": "b1", "intent": "Interact with the new world",
          "source_quote": "visit another world", "effects": "Travel"}],
        producer_obligations=[
            {"behavior_id": "b1", "name": "selected_section",
             "reason": "no_inspected_semantic_source"},
            {"behavior_id": "b1", "reason": "occurrence_hook_not_owned"},
        ],
        inspected_hooks=[{"event_id": "FabricOwner#END"}],
    )
    nodes = graph["tasks"]
    assert [row["kind"] for row in nodes] == [
        "occurrence_hook", "runtime_producer", "gameplay_effect",
        "event_registration", "target_compile", "runtime_playtest",
    ]
    assert nodes[0]["status"] == "pending"
    assert nodes[1]["depends_on"] == [nodes[0]["task_id"]]
    assert nodes[2]["depends_on"] == [nodes[0]["task_id"], nodes[1]["task_id"]]
    assert nodes[-1]["depends_on"] == [nodes[-2]["task_id"]]
    assert graph["completion_verified"] is False
    effect = member_source_work(graph, "b1", member_role="implement_callback_effects")
    assert {row["kind"] for row in effect["source_tasks"]} == {
        "occurrence_hook", "runtime_producer", "gameplay_effect",
    }
    registrar = member_source_work(graph, "b1", member_role="register_callback")
    assert {row["kind"] for row in registrar["source_tasks"]} == {
        "occurrence_hook", "event_registration",
    }


def test_verified_api_inventory_is_not_verified_occurrence():
    kwargs = {
        "producer_obligations": [],
        "inspected_hooks": [{"event_id": "FabricOwner#END"}],
    }
    plan = [{"behavior_id": "b1", "event_binding": "FabricOwner#END"}]
    pending = build_source_work_graph(plan, **kwargs)
    assert pending["tasks"][0]["status"] == "pending"
    checked = build_source_work_graph(
        plan, verified_behavior_ids=["b1"], **kwargs)
    assert checked["tasks"][0]["status"] == "verified"


def test_uninspected_event_is_rejected_before_codegen():
    with pytest.raises(ValueError, match="EXPLICIT_HOOK_NOT_INSPECTED"):
        build_source_work_graph(
            [{"behavior_id": "b1", "event_binding": "invented#EVENT"}],
            producer_obligations=[], inspected_hooks=[])


def test_dangling_runtime_obligation_is_rejected():
    with pytest.raises(ValueError, match="UNKNOWN_BEHAVIOR"):
        build_source_work_graph(
            [{"behavior_id": "b1"}], producer_obligations=[
                {"behavior_id": "other", "name": "unimplemented_observation"}],
            inspected_hooks=[])


def test_duplicate_behaviors_fail_instead_of_collapsing_coverage():
    with pytest.raises(ValueError, match="DUPLICATE"):
        build_source_work_graph(
            [{"behavior_id": "same"}, {"behavior_id": "same"}],
            producer_obligations=[], inspected_hooks=[])



def test_checked_selected_callback_does_not_fake_observer_completion():
    graph = build_source_work_graph(
        [{"behavior_id": "b1", "trigger": "Real runtime condition"}],
        producer_obligations=[{"behavior_id": "b1", "name": "selected_section",
                               "reason": "planning_only_world_state"}],
        inspected_hooks=[{"event_id": "actual#EVENT"}],
    )
    selected = bind_selected_hooks(graph, {"b1": {"event_id": "actual#EVENT"}})
    assert selected["tasks"][0]["status"] == "source_selected"
    assert selected["tasks"][0]["evidence"]["event_binding"] == "actual#EVENT"
    assert selected["tasks"][1]["status"] == "pending"
    assert selected["completion_verified"] is False
    assert graph["tasks"][0]["evidence"]["event_binding"] is None


def test_uninspected_source_hook_cannot_be_inserted_into_producer_graph():
    graph = build_source_work_graph(
        [{"behavior_id": "b1"}], producer_obligations=[],
        inspected_hooks=[{"event_id": "actual#EVENT"}],
    )
    with pytest.raises(ValueError, match="NOT_INSPECTED"):
        bind_selected_hooks(graph, {"b1": {"event_id": "invented#EVENT"}})
