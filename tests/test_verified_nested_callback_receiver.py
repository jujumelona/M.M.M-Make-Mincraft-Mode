"""Real inspected receiver-edge lowering for event-driven gameplay."""
import json

from minecraft_mod_ai.construction_file_api_evidence import FileAPIEvidence
from minecraft_mod_ai.construction_java_context import scoped_java_member_messages
from minecraft_mod_ai.host_verified_callback_ops import verified_callback_choices


def test_actual_player_to_world_to_gameplay_mutator_is_host_lowered():
    player, world = "example.Player", "example.World"
    symbols = {
        player: {"kind": "CLASS", "owner": player, "name": "Player", "access": 1},
        world: {"kind": "CLASS", "owner": world, "name": "World", "access": 1},
        "getter": {"kind": "METHOD", "owner": player, "name": "getWorld",
                   "parameters": [], "return_type": world,
                   "access": 1, "static": False},
        "mutator": {"kind": "METHOD", "owner": world, "name": "colonizePlanet",
                    "parameters": [], "return_type": "void",
                    "access": 1, "static": False},
    }
    api = FileAPIEvidence(
        symbols, target_path="src/main/java/example/Feature.java",
        owner="example.Feature",
    ).callback_member_records([player], intent="colonize planet in world")
    bridge = next(row for row in api
                  if row.get("verified_class") == world)
    assert bridge["verified_receiver_path"]["source_method"] == "getWorld"
    assert any(row["name"] == "colonizePlanet" for row in bridge["members"])
    event = {"event_id": "example.Event#EVENT", "parameters": [player],
             "return_type": "void"}
    task = {
        "verified_event_callback": event, "callback_input_api": api,
        "response_source_frame": {"kind": "method",
            "prefix": "private void execute(example.Player player) {\n",
            "suffix": "\n}"},
    }
    frame = task["response_source_frame"]
    scoped = scoped_java_member_messages(
        [{"role": "user", "content": json.dumps(task)}], frame)
    projected = json.loads(scoped[-1]["content"])
    assert any("verified_receiver_path" in row for row
               in projected["callback_input_api"])
    calls = verified_callback_choices(scoped, frame)
    assert "player.getWorld().colonizePlanet()" in {
        row["expression"] for row in calls}
    # Removing the verified root getter must remove the derived call.
    altered = json.loads(json.dumps(projected))
    root = next(row for row in altered["callback_input_api"]
                if row["verified_class"] == player)
    root["members"] = []
    none = verified_callback_choices(
        [{"role": "user", "content": json.dumps(altered)}], frame)
    assert "player.getWorld().colonizePlanet()" not in {
        row["expression"] for row in none}
