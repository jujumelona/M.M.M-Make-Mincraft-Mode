"""A failed event-owner group must not erase a valid inspected hook."""

import json

import pytest

from minecraft_mod_ai.host_behavior_event_binding import (
    HostEventBindingError, select_behavior_callback,
)


class EventSelector:
    def __init__(self):
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        task = json.loads(messages[-1]["content"])
        self.calls.append((data_name, task, parameters))
        if data_name == "select_verified_fabric_event_index":
            return {"event_binding": "UNBOUND"}
        if data_name == "select_verified_fabric_event_owner":
            return {"event_owner": "test.LifecycleEvents"}
        if data_name == "select_verified_fabric_event":
            return {"event_binding": "UNBOUND"}
        if data_name == "select_verified_fabric_event_owner_alternative_1":
            assert "test.LifecycleEvents" not in [
                row["event_owner"] for row in task["verified_event_groups"]]
            assert task["rejected_event_owners"] == [{
                "event_owner": "test.LifecycleEvents",
                "reason": "No event within the selected owner can observe this trigger.",
            }]
            return {"event_owner": "test.TickEvents"}
        if data_name == "select_verified_fabric_event_alternative_1":
            return {"event_binding": "test.TickEvents#END_LEVEL_TICK"}
        raise AssertionError("unexpected model request " + data_name)


def _event(owner, name):
    return {"event_id": owner + "#" + name, "event_owner": owner,
            "event_field": name, "event_interface": owner + ".Listener",
            "parameters": ["net.minecraft.server.MinecraftServer"],
            "return_type": "void"}


def test_event_owner_search_recovers_from_first_group_unbound():
    router = EventSelector()
    behavior = {"behavior_id": "season-calendar", "trigger": "A repeating world-time condition",
                "intent": "Check a periodic world-state transition"}
    candidate = _event("test.TickEvents", "END_LEVEL_TICK")
    result = select_behavior_callback(router, behavior, [
        _event("test.LifecycleEvents", "AFTER_SAVE"), candidate,
        _event("test.ItemEvents", "USE"),
    ])
    assert result is candidate
    assert [name for name, _, _ in router.calls] == [
        "select_verified_fabric_event_index",
        "select_verified_fabric_event_owner", "select_verified_fabric_event",
        "select_verified_fabric_event_owner_alternative_1",
        "select_verified_fabric_event_alternative_1",
    ]


def test_event_binding_refuses_to_invent_any_hook_after_exhaustion():
    class NoVerifiedHook:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            options = parameters["properties"]
            key = next(iter(options))
            if key == "event_binding":
                return {"event_binding": "UNBOUND"}
            return {"event_owner": "test.LifecycleEvents"}

    with pytest.raises(HostEventBindingError, match="UNBOUND"):
        select_behavior_callback(NoVerifiedHook(),
            {"behavior_id": "unresolvable", "trigger": "No observable callback"},
            [_event("test.LifecycleEvents", "AFTER_SAVE")])



def test_global_index_recovers_time_tick_without_wrong_owner_selection():
    class GlobalSelector:
        def __init__(self):
            self.calls = []

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert data_name == "select_verified_fabric_event_index"
            payload = json.loads(messages[-1]["content"])
            self.calls.append(payload)
            offered = {row["event_id"] for row in payload["verified_hook_index"]}
            assert "test.TickEvents#END_LEVEL_TICK" in offered
            assert "test.LifecycleEvents#AFTER_SAVE" in offered
            return {"event_binding": "test.TickEvents#END_LEVEL_TICK"}

    router = GlobalSelector()
    tick = _event("test.TickEvents", "END_LEVEL_TICK")
    result = select_behavior_callback(router, {
        "behavior_id": "world-period", "trigger": "Game time reaches an interval boundary",
    }, [_event("test.LifecycleEvents", "AFTER_SAVE"), tick])
    assert result is tick
    assert len(router.calls) == 1


def test_unbound_owner_is_search_uncertainty_not_absence_of_all_hooks():
    class DecliningOwner:
        def __init__(self):
            self.calls = []

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            self.calls.append(data_name)
            if data_name == "select_verified_fabric_event_index":
                return {"event_binding": "UNBOUND"}
            if data_name == "select_verified_fabric_event_owner":
                return {"event_owner": "UNBOUND"}
            if data_name == "select_verified_fabric_event":
                # A real owner was chosen from inspected callback inputs.
                return {"event_binding": "test.TickEvents#END_LEVEL_TICK"}
            raise AssertionError(data_name)

    router = DecliningOwner()
    usable = _event("test.TickEvents", "END_LEVEL_TICK")
    lifecycle = _event("test.LifecycleEvents", "AFTER_SAVE")
    lifecycle["parameters"] = []
    result = select_behavior_callback(
        router, {"behavior_id": "arbitrary-world-trigger",
                 "trigger": "World condition reaches threshold"},
        [lifecycle, usable],
    )
    assert result is usable
    assert router.calls == [
        "select_verified_fabric_event_index",
        "select_verified_fabric_event_owner",
        "select_verified_fabric_event",
    ]
