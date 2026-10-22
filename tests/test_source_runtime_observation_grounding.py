"""A fake true predicate cannot complete a missing world-state producer."""
import json

import pytest

from minecraft_mod_ai.host_callback_body import validate_runtime_observation_predicate


def _scope(required=True):
    payload = {
        "declaration_metadata": {"parameters": [
            {"type": "example.World", "name": "world"}]},
        "source_owned_work": {"source_tasks": [
            {"kind": "runtime_producer", "name": "selected_section",
             "status": "pending" if required else "verified"}]},
    }
    return [{"role": "user", "content": json.dumps(payload)}]


@pytest.mark.parametrize("expression", [
    "true", "false", "1 == 1", "java.lang.Boolean.TRUE",
])
def test_pending_producer_rejects_constant_gameplay_observation(expression):
    with pytest.raises(ValueError, match="SOURCE_RUNTIME_OBSERVATION_NOT_GROUNDED"):
        validate_runtime_observation_predicate(_scope(), expression)


def test_pending_observation_must_at_least_depend_on_real_callback_input():
    validate_runtime_observation_predicate(_scope(), "world.isReady()")
    validate_runtime_observation_predicate(_scope(), "world != null")


def test_already_verified_observation_does_not_claim_a_new_receiver():
    validate_runtime_observation_predicate(_scope(required=False), "true")


def test_unrelated_variable_cannot_stand_in_for_real_world_receiver():
    with pytest.raises(ValueError, match="SOURCE_RUNTIME_OBSERVATION_NOT_GROUNDED"):
        validate_runtime_observation_predicate(_scope(), "inventedFlag")
