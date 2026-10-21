"""A domain transition is lowered through a host observation operator."""
import importlib
import json

import pytest

from minecraft_mod_ai.structured_output import StructuredOutputValidationError


TICKS = "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents"


def event(owner, field, input_type="net.minecraft.server.level.ServerLevel"):
    return {"event_id": owner + "#" + field, "event_owner": owner,
            "event_field": field, "event_interface": owner + ".Listener",
            "parameters": [input_type], "return_type": "void"}


def bind(*args):
    return importlib.import_module(
        "minecraft_mod_ai.host_rule_observation").bind_rule_observation(*args)


@pytest.mark.parametrize("trigger", [
    "Game time reaches the start of a season",
    "Season changes or player enters a season",
    "A reactor's pressure crosses the configured limit",
])
def test_domain_observation_uses_checked_tick_without_api_matching(trigger):
    tick = event(TICKS, "END_LEVEL_TICK")

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            assert data_name == "rule_observation"
            assert task["behavior"]["trigger"] == trigger
            assert len(json.dumps(task)) < 1600
            assert "verified_hook_index" not in task
            assert parameters["properties"]["observation"]["enum"] == [
                "world_state", "native_event"]
            return {"observation": "world_state"}

    result = bind(Router(), {"behavior_id": "observed-change", "trigger": trigger}, [
        tick, *[event("example.Event" + str(i), "EVENT") for i in range(80)]])
    assert result == {"mode": "world_state", "event": tick}


def test_native_occurrence_is_never_replaced_by_polling_after_rejection():
    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            return ({"observation": "native_event"} if data_name == "rule_observation"
                    else {"event_id": "next_page"})

    with pytest.raises(ValueError, match="RULE_NATIVE_HOOK_REQUIRED"):
        bind(Router(), {"behavior_id": "before-damage", "trigger": "before damage"}, [
            event(TICKS, "END_LEVEL_TICK"), event("example.Other", "EVENT")])


def test_native_catalogue_is_paged_without_losing_later_candidates():
    candidates = [event("example.Event" + str(i).zfill(3), "EVENT") for i in range(39)]
    selected = candidates[-1]
    sizes = []

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name == "rule_observation":
                return {"observation": "native_event"}
            payload = json.loads(messages[-1]["content"])
            sizes.append(len(payload["events"]))
            options = parameters["properties"]["event_id"]["enum"]
            return {"event_id": selected["event_id"] if selected["event_id"] in options
                    else "next_page"}

    assert bind(Router(), {"behavior_id": "native"}, candidates)["event"] is selected
    assert max(sizes) <= 12 and sum(sizes) == len(candidates)


def test_router_cannot_inject_an_uninspected_hook():
    class Router:
        def generate_structured_data(self, *args, **kwargs):
            return {"observation": "invented.Event#EVENT"}

    with pytest.raises(StructuredOutputValidationError):
        bind(Router(), {"behavior_id": "invalid"}, [event(TICKS, "END_LEVEL_TICK")])
