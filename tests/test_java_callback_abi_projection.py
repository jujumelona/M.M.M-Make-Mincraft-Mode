"""Production replay: never send unrelated API owner inventories to a callback coder."""
import json

from minecraft_mod_ai.construction_java_context import scoped_java_member_messages


def test_runtime_state_observation_bindings_reach_the_current_callback():
    frame = {"kind": "method", "prefix": "private void apply() {\n", "suffix": "\n}"}
    observations = {"build_area_contents": {"binding": "world_state",
                    "observation": "Read and change actual blocks in the event world"}}
    messages = [{"role": "user", "content": json.dumps({
        "response_source_frame": frame, "runtime_state_observations": observations})}]
    projected = scoped_java_member_messages(messages, frame)
    assert json.loads(projected[-1]["content"])["runtime_state_observations"] == observations


def test_materialized_content_apis_survive_alongside_inspected_classpath_api():
    frame = {"kind": "method", "prefix": "private void apply(Target target) {\n", "suffix": "\n}"}
    prerequisite = [{"path": "ModItems.java", "types": [{"owner": "example.ModItems",
        "members": ["public static final example.Item ION_RELAY;"]}]}]
    payload = {"response_source_frame": frame,
               "inspected_target_api": {"target_api": [{"owner": "example.Target", "name": "equip"}]},
               "materialized_prerequisite_api": prerequisite}
    projected = scoped_java_member_messages([{"role": "user", "content": json.dumps(payload)}], frame)
    task = json.loads(projected[-1]["content"])
    assert task["materialized_prerequisite_api"] == prerequisite
    assert "equip" in json.dumps(task["inspected_target_api"])



def _record(name, count=16):
    return {
        "verified_class": name,
        "declaration": {"owner": name, "kind": "CLASS", "interfaces": []},
        "members": [
            {"kind": "METHOD", "name": "call" + str(i), "owner": name,
             "return_type": "void", "parameters": ["java.lang.String"],
             "access": 1, "generic_parameters": [],
             "generic_return_type": "", "throws": [],
             "documentation": "irrelevant verbose Java API metadata" * 20}
            for i in range(count)
        ],
        "member_count": count, "members_truncated": True,
    }


def test_host_projects_real_callback_types_and_rejects_location_search_noise():
    frame = {
        "kind": "method", "prefix": "public boolean check(Object x) {",
        "suffix": "}", "host_state_access": [{
            "owner": "example.AuthoredStateModel",
            "methods": [{"name": "readState_balance",
                         "signature": "public static int readState_balance();"}],
        }],
    }
    event = {
        "event_id": "fabric.event#EVENT", "event_owner": "fabric.event",
        "parameters": ["net.minecraft.world.entity.player.Player",
                       "net.minecraft.world.entity.Entity"],
        "return_type": "net.minecraft.world.InteractionResult",
    }
    payload = {
        "response_source_frame": frame,
        "verified_event_callback": event,
        "assigned_behavior": {
            "behavior_id": "b1", "intent": "Fight aliens",
            "trigger": "Alien encounter", "effects": "Combat aliens",
        },
        "callback_input_api": [
            _record("net.minecraft.world.entity.player.Player"),
            _record("net.minecraft.world.entity.Entity"),
            _record("net.minecraft.world.InteractionResult"),
            _record("net.minecraft.world.InteractionResult.Success"),
            _record("javax.tools.StandardLocation"),
        ],
        "declared_member_api": [
            {"signature": "private boolean check(Object x)"}],
        "implementation_requirements": "Preserve all behavior",
    }
    context = {"inspected_target_api": {
        "target_api": [
            {"owner": "javax.tools.StandardLocation",
             "kind": "METHOD", "name": "locationFor"},
            {"owner": "net.minecraft.world.entity.Entity",
             "kind": "METHOD", "name": "hurt"},
        ],
        "class_members": [_record("javax.tools.StandardLocation"),
                          _record("net.minecraft.world.entity.Entity")],
    }}
    messages = [
        {"role": "user", "content": json.dumps(context)},
        {"role": "user", "content": json.dumps(payload)},
    ]
    result = scoped_java_member_messages(messages, frame)
    text = "\n".join(msg["content"] for msg in result)
    task = json.loads(result[-1]["content"])
    assert len(text) < 15000
    assert "javax.tools.StandardLocation" not in text
    assert "locationFor" not in text
    assert "net.minecraft.world.InteractionResult.Success" not in text
    assert {row["verified_class"] for row in task["callback_input_api"]} == {
        "net.minecraft.world.entity.player.Player",
        "net.minecraft.world.entity.Entity",
        "net.minecraft.world.InteractionResult",
    }
    assert task["inspected_target_api"]["target_api"][0]["name"] == "hurt"
    assert task["host_state_access"] == frame["host_state_access"]
    assert task["assigned_behavior"]["behavior_id"] == "b1"
    assert "import statements" in result[0]["content"]


def test_non_event_tasks_keep_full_inspected_api_ownership():
    from minecraft_mod_ai.construction_java_context import _relevant_current_api
    evidence = {
        "target_api": [{"owner": "my.Verified", "name": "call", "kind": "METHOD"}],
        "class_members": [{"verified_class": "my.Verified", "members": [
            {"owner": "my.Verified", "name": "call", "kind": "METHOD"}]}],
    }
    projected = _relevant_current_api(evidence, set(), {})
    assert projected["target_api"] == evidence["target_api"]
    assert projected["class_members"][0]["verified_class"] == "my.Verified"


def test_state_work_order_exposes_missing_canonical_values_without_fabricating_accessors():
    from minecraft_mod_ai.host_typed_state_access import state_work_order_scope
    order = {
        "state_reads": ["selected_section", "spacecraft_sections",
                        "Player health", "player_credits"],
        "state_writes": ["spacecraft_sections", "build_area_contents"],
    }
    contracts = [{"owner": "proof.AuthoredStateModel", "methods": [
        {"name": "readState_player_credits"},
        {"name": "writeState_player_credits"},
    ]}]
    scope = state_work_order_scope(order, contracts)
    assert scope["declared_readable"] == ["player_credits"]
    assert scope["undeclared_plan_reads"] == [
        "selected_section", "spacecraft_sections",
    ]
    assert scope["undeclared_plan_writes"] == [
        "build_area_contents", "spacecraft_sections",
    ]
    assert "unrelated" in scope["contract"]
    assert "readState_spacecraft_sections" not in str(scope)


def test_scoped_member_keeps_plan_gap_provenance_for_correction():
    frame = {"kind": "method", "prefix": "public void f() {",
             "suffix": "}"}
    gap = {"declared_readable": ["credits"],
           "undeclared_plan_reads": ["selected_section"]}
    task = {"response_source_frame": frame,
            "host_state_work_order_scope": gap,
            "assigned_behavior": {"behavior_id": "b", "effects": "Set section"}}
    messages = [{"role": "user", "content": json.dumps(task)}]
    scoped = scoped_java_member_messages(messages, frame)
    assert json.loads(scoped[-1]["content"])["host_state_work_order_scope"] == gap
