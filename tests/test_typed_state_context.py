from copy import deepcopy
import shutil
import subprocess

import pytest

from minecraft_mod_ai.production_state_compiler import render_production_state_java
from minecraft_mod_ai.typed_plan_authoring import lower_semantic_game_dispatch_to_ir
from minecraft_mod_ai.typed_plan_java import render_typed_plan_java
from minecraft_mod_ai.typed_state_context import bind_player_state_context
from test_production_state_compiler import _credits_section


def _legacy_plan():
    return {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": "a" * 64,
        "initialize": [],
        "event_bindings": [],
        "functions": [
            {
                "id": "logic_dispatch",
                "parameters": [
                    {"name": "event", "type": "string"},
                    {"name": "primary", "type": "object"},
                    {"name": "secondary", "type": "object"},
                    {"name": "flag", "type": "boolean"},
                ],
                "return_type": "int",
                "covers": [],
                "body": lower_semantic_game_dispatch_to_ir(
                    [
                        {
                            "trigger_event": "player_join",
                            "action_kind": "increment_state",
                            "state_key": "credits",
                            "value": 7,
                        }
                    ],
                    {"credits": "long"},
                    {},
                ),
            }
        ],
    }


def test_saved_semantic_dispatch_uses_its_actual_primary_player(tmp_path):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK required")
    original = _legacy_plan()
    before = deepcopy(original)
    plan, caps = bind_player_state_context(original, _credits_section(), {})
    assert original == before
    sources = {
        "AuthoredProgram.java": render_typed_plan_java(
            plan, package="proof", capabilities=caps
        ),
        "AuthoredStateModel.java": render_production_state_java(
            _credits_section(), package_name="proof"
        ),
        "AuthoredHostCapabilities.java": "package proof; public class AuthoredHostCapabilities { public static String playerUuid(Object actor) { return actor.toString(); } }",
        "Probe.java": """package proof;
public class Probe {
 public static void main(String[] args) {
  for (String actor : new String[]{"a", "b", "a"})
   AuthoredProgram.invoke("logic_dispatch", java.util.Map.of("event", "player_join", "primary", actor, "secondary", actor, "flag", false));
  if (AuthoredStateModel.readState_credits(java.util.Map.of("player", "a")) != 14L
    || AuthoredStateModel.readState_credits(java.util.Map.of("player", "b")) != 7L) throw new AssertionError();
 }
}""",
    }
    for name, source in sources.items():
        (tmp_path / name).write_text(source)
    compiled = subprocess.run(
        [javac, "-d", str(tmp_path), *map(str, tmp_path.glob("*.java"))],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(
        [java, "-cp", str(tmp_path), "proof.Probe"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert executed.returncode == 0, executed.stderr


def test_player_context_follows_host_bound_event_arguments_through_helpers():
    plan = _legacy_plan()
    helper = plan["functions"][0]
    helper["id"] = "nested"
    helper["parameters"][1]["name"] = "actor"
    for node in helper["body"]:
        # The semantic trigger uses event; the host context is added to actor.
        assert node["op"] in {"if", "return"}
    params = [{"name": "caller", "type": "object"}]
    plan["functions"].append(
        {
            "id": "command_root",
            "parameters": params,
            "return_type": "int",
            "covers": [],
            "body": [
                {
                    "op": "return",
                    "value": {
                        "op": "call",
                        "function": "nested",
                        "args": [
                            {"op": "literal", "type": "string", "value": "player_join"},
                            {"op": "ref", "name": "caller"},
                            {"op": "literal", "type": "object", "value": None},
                            {"op": "literal", "type": "boolean", "value": False},
                        ],
                    },
                }
            ],
        }
    )
    plan["event_bindings"] = [
        {
            "event": "command",
            "function": "command_root",
            "entry_point_index": 0,
            "config": {"literal": "legacy", "permission_level": 0},
        }
    ]
    bound, _ = bind_player_state_context(plan, _credits_section(), {})
    value = bound["functions"][0]["body"][0]["then"][0]["context"]["entries"][0][
        "value"
    ]
    assert value["args"] == [{"op": "ref", "name": "actor"}]


def test_player_state_without_an_actor_fails_before_java_generation():
    plan = _legacy_plan()
    plan["functions"][0]["id"] = "unbound"
    with pytest.raises(ValueError, match="TYPED_PLAYER_STATE_ACTOR_REQUIRED"):
        bind_player_state_context(plan, _credits_section(), {})


@pytest.mark.parametrize("second_call", ["other_position", "server"])
def test_shared_helper_rejects_ambiguous_or_nonplayer_incoming_arguments(second_call):
    plan = _legacy_plan()
    helper = plan["functions"][0]
    helper["id"] = "helper"
    helper["parameters"] = [
        {"name": "left", "type": "object"},
        {"name": "right", "type": "object"},
    ]
    helper["body"] = [helper["body"][0]["then"][0], helper["body"][1]]
    for i in range(2):
        player = {"op": "ref", "name": "source"}
        null = {"op": "literal", "type": "object", "value": None}
        args = [player, null] if i == 0 else [null, player]
        event = "server_started" if i and second_call == "server" else "command"
        plan["functions"].append(
            {
                "id": f"root_{i}",
                "parameters": [{"name": "source", "type": "object"}],
                "return_type": "int" if event == "command" else "void",
                "covers": [],
                "body": [
                    {
                        "op": "return",
                        "value": {"op": "call", "function": "helper", "args": args},
                    }
                ]
                if event == "command"
                else [
                    {
                        "op": "expr",
                        "value": {"op": "call", "function": "helper", "args": args},
                    }
                ],
            }
        )
        plan["event_bindings"].append(
            {
                "event": event,
                "function": f"root_{i}",
                "entry_point_index": i,
                "config": {"literal": f"root_{i}", "permission_level": 0}
                if event == "command"
                else {},
            }
        )
    with pytest.raises(ValueError, match="TYPED_PLAYER_STATE_ACTOR"):
        bind_player_state_context(plan, _credits_section(), {})


def test_semantic_dispatch_does_not_treat_server_event_as_a_player():
    plan = _legacy_plan()
    plan["functions"][0]["body"][0]["condition"]["right"]["value"] = "server_tick"
    with pytest.raises(ValueError, match="TYPED_PLAYER_STATE_ACTOR"):
        bind_player_state_context(plan, _credits_section(), {})


def test_same_function_bound_to_player_and_server_events_has_no_unique_actor():
    plan = _legacy_plan()
    fn = plan["functions"][0]
    fn.update(
        id="shared",
        parameters=[{"name": "actor", "type": "object"}],
        return_type="void",
        body=[fn["body"][0]["then"][0]],
    )
    plan["event_bindings"] = [
        {"event": event, "function": "shared", "entry_point_index": i, "config": {}}
        for i, event in enumerate(("player_join", "server_started"))
    ]
    with pytest.raises(ValueError, match="TYPED_PLAYER_STATE_ACTOR"):
        bind_player_state_context(plan, _credits_section(), {})


@pytest.mark.parametrize("shared_root", [True, False])
def test_semantic_dispatch_keeps_player_and_server_root_provenance(shared_root):
    plan = _legacy_plan()
    for index, event in enumerate(("player_join", "server_started")):
        name = "shared_root" if shared_root else f"root_{index}"
        if index == 0 or not shared_root:
            plan["functions"].append(
                {
                    "id": name,
                    "parameters": [{"name": "actor", "type": "object"}],
                    "return_type": "void",
                    "covers": [],
                    "body": [
                        {
                            "op": "expr",
                            "value": {
                                "op": "call",
                                "function": "logic_dispatch",
                                "args": [
                                    {
                                        "op": "literal",
                                        "type": "string",
                                        "value": "player_join"
                                        if shared_root
                                        else event,
                                    },
                                    {"op": "ref", "name": "actor"},
                                    {"op": "literal", "type": "object", "value": None},
                                    {
                                        "op": "literal",
                                        "type": "boolean",
                                        "value": False,
                                    },
                                ],
                            },
                        }
                    ],
                }
            )
        plan["event_bindings"].append(
            {"event": event, "function": name, "entry_point_index": index, "config": {}}
        )
    if shared_root:
        with pytest.raises(ValueError, match="TYPED_PLAYER_STATE_ACTOR"):
            bind_player_state_context(plan, _credits_section(), {})
    else:
        bound, _ = bind_player_state_context(plan, _credits_section(), {})
        assert bound["functions"][0]["body"][0]["then"][0]["context"]["entries"]
