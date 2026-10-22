"""A diagnostic call is not a gameplay effect."""
import pytest

from minecraft_mod_ai.host_callback_body import _effect_operation


@pytest.mark.parametrize("body", [
    "System.out.println(\"done\");",
    "java.lang.System.err.println(\"error\");",
    'logger.info("requested effect");',
    "obj.getHealth();",
    "world.isDay();",
    "player.hasPermission();",
])
def test_diagnostic_and_read_only_calls_are_not_gameplay_mutations(body):
    assert _effect_operation("call", body) is False


@pytest.mark.parametrize("body,action", [
    ("world.setBlock(pos, block);", "call"),
    ("target.effects += 1;", "assign"),
    ("items[0] = value;", "assign"),
    ("world.teleportEntity(entity);", "call"),
])
def test_candidate_mutations_still_need_actual_compilation_and_playtest(body, action):
    assert _effect_operation(action, body) is True


def test_local_value_assignment_never_satisfies_world_effect():
    assert not _effect_operation("assign", "count += 1;")
