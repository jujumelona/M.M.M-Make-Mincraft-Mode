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
        "custom_event_source", "occurrence_hook", "runtime_producer",
        "gameplay_effect", "event_registration", "target_compile", "runtime_playtest",
    ]
    assert nodes[0]["status"] == "pending"
    assert nodes[0]["evidence"]["reasons"] == ["occurrence_hook_not_owned"]
    assert nodes[1]["depends_on"] == [nodes[0]["task_id"]]
    assert nodes[2]["depends_on"] == [nodes[1]["task_id"]]
    assert nodes[2]["evidence"]["source_order"] == "after_occurrence_hook_before_effect"
    assert nodes[3]["depends_on"] == [nodes[1]["task_id"], nodes[2]["task_id"]]
    assert nodes[-1]["depends_on"] == [nodes[-2]["task_id"]]
    assert graph["completion_verified"] is False
    effect = member_source_work(graph, "b1", member_role="implement_callback_effects")
    assert {row["kind"] for row in effect["source_tasks"]} == {
        "custom_event_source", "occurrence_hook", "runtime_producer", "gameplay_effect",
    }
    registrar = member_source_work(graph, "b1", member_role="register_callback")
    assert {row["kind"] for row in registrar["source_tasks"]} == {
        "custom_event_source", "occurrence_hook", "event_registration",
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


def test_missing_inspected_callback_creates_source_owner_before_hook():
    graph = build_source_work_graph(
        [{"behavior_id": "new", "trigger": "User selects a custom build part"}],
        producer_obligations=[{"behavior_id": "new", "reason": "no_inspected_callback"}],
        inspected_hooks=[],
    )
    source, hook, effect, registration, compilation, playtest = graph["tasks"]
    assert source["kind"] == "custom_event_source"
    assert source["status"] == "pending"
    assert hook["depends_on"] == [source["task_id"]]
    assert effect["depends_on"] == [hook["task_id"]]
    assert registration["depends_on"] == [hook["task_id"], effect["task_id"]]
    assert compilation["depends_on"] == [registration["task_id"]]
    assert playtest["depends_on"] == [compilation["task_id"]]
    assert graph["completion_verified"] is False


def test_missing_runtime_observation_precedes_real_callback_even_if_fabric_has_events():
    graph = build_source_work_graph(
        [{"behavior_id": "trade", "trigger": "Player confirms a transaction",
          "runtime_state_reads": ["selected_offer"]}],
        producer_obligations=[{"behavior_id": "trade",
                               "name": "selected_offer",
                               "reason": "no_inspected_semantic_source"}],
        inspected_hooks=[{"event_id": "FabricOwner#END"}],
    )
    hook, producer, effect, registration, compilation, playtest = graph["tasks"]
    assert producer["kind"] == "runtime_producer"
    assert hook["depends_on"] == []
    assert producer["depends_on"] == [hook["task_id"]]
    assert producer["evidence"]["source_order"] == "after_occurrence_hook_before_effect"
    assert hook["status"] == "pending"
    assert effect["depends_on"] == [hook["task_id"], producer["task_id"]]
    assert all(task["status"] != "verified" for task in graph["tasks"])
    assert graph["completion_verified"] is False


def test_colonization_missing_observations_are_not_misclassified_as_missing_hook():
    from minecraft_mod_ai.feature_source_work_graph import requires_original_occurrence_site

    graph = build_source_work_graph(
        [{"behavior_id": "colonize", "trigger":
          "Player reaches a planet or space location and initiates colonization",
          "runtime_state_reads": [
              "Player location", "Available planets", "Colonization prerequisites"]}],
        producer_obligations=[
            {"behavior_id": "colonize", "name": "Available planets",
             "reason": "no_inspected_semantic_source"},
            {"behavior_id": "colonize", "name": "Colonization prerequisites",
             "reason": "no_inspected_semantic_source"},
        ],
        inspected_hooks=[{"event_id": "verified#CHANGE_LEVEL"}],
    )
    assert not requires_original_occurrence_site(graph, "colonize")
    assert graph["tasks"][0]["kind"] == "occurrence_hook"
    assert {row["kind"] for row in graph["tasks"][1:3]} == {"runtime_producer"}
    assert all(graph["tasks"][0]["task_id"] in row["depends_on"]
               for row in graph["tasks"][1:3])
    selected = bind_selected_hooks(
        graph, {"colonize": {"event_id": "verified#CHANGE_LEVEL"}})
    assert selected["tasks"][0]["status"] == "source_selected"
    assert all(row["status"] == "pending" for row in selected["tasks"][1:3])
    # A later compiler/runtime check must still demonstrate actual data.
    assert not selected["completion_verified"]


def test_real_missing_occurrence_requires_source_even_when_data_is_known():
    from minecraft_mod_ai.feature_source_work_graph import requires_original_occurrence_site
    graph = build_source_work_graph(
        [{"behavior_id": "build", "trigger": "Build spacecraft section"}],
        producer_obligations=[
            {"behavior_id": "build", "reason": "occurrence_hook_not_owned"}],
        inspected_hooks=[{"event_id": "verified#OTHER"}],
    )
    assert requires_original_occurrence_site(graph, "build")


def test_observer_slots_are_deterministic_distinct_and_keep_source_concepts():
    from minecraft_mod_ai.feature_source_work_graph import runtime_observer_methods
    behaviors = [{
        "behavior_id": "colonize",
        "trigger": "Player reaches a planet and initiates colonization",
    }]
    obligations = [
        {"behavior_id": "colonize", "name": "Available planets",
         "reason": "no_inspected_semantic_source"},
        {"behavior_id": "colonize", "name": "Colonization prerequisites",
         "reason": "no_inspected_semantic_source"},
    ]
    graph = build_source_work_graph(
        behaviors, producer_obligations=obligations,
        inspected_hooks=[{"event_id": "real#AFTER_PLAYER_CHANGE_LEVEL"}],
    )
    slots = runtime_observer_methods(graph, "colonize")
    assert {row["source_name"] for row in slots} == {
        "Available planets", "Colonization prerequisites"}
    assert len({row["name"] for row in slots}) == 2
    assert all(row["name"].startswith("observeRuntime") and
               row["name"].isidentifier() for row in slots)
    assert slots == runtime_observer_methods(graph, "colonize")
    assert runtime_observer_methods(graph, "other") == []
