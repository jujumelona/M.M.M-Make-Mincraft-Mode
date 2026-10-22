"""Small-model source hook selection: never invent event receiver or key press."""
import pytest

from minecraft_mod_ai.source_owned_hook import choose_source_hook, SourceHookUnavailable


class HookPicker:
    def __init__(self, choice):
        self.choice = choice
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append((role, data_name, parameters, messages))
        return {"event_id": self.choice}


def _hook(event_id):
    return {
        "event_id": event_id,
        "event_owner": "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents",
        "event_field": "END_SERVER_TICK",
        "event_interface": "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents.EndTick",
        "method": "onEndTick",
        "parameters": ["net.minecraft.server.MinecraftServer"],
        "return_type": "void",
    }


def _behavior():
    return {
        "behavior_id": "space-world-1",
        "source_quote": "Explore the planets and build a spacecraft",
        "intent": "Travel between worlds",
        "trigger": "Space travel condition becomes true",
        "effects": "Transition to another planet",
    }


def test_source_hook_choice_is_one_bounded_real_callback():
    picker = HookPicker("real#END")
    found = choose_source_hook(picker, _behavior(), [_hook("real#END"), _hook("other#START")])
    assert found["event_id"] == "real#END"
    assert len(picker.calls) == 1
    role, data_name, schema, messages = picker.calls[0]
    assert role == "planner"
    assert data_name == "select_source_owned_observation_hook"
    assert sorted(schema["properties"]["event_id"]["enum"]) == [
        "SOURCE_REQUIRED", "other#START", "real#END",
    ]
    assert "minecraft.server" in messages[-1]["content"]


def test_explicit_inspected_event_does_not_ask_model_again():
    picker = HookPicker("SOURCE_REQUIRED")
    b = {**_behavior(), "event_binding": "real#END"}
    assert choose_source_hook(picker, b, [_hook("real#END")])["event_id"] == "real#END"
    assert not picker.calls


def test_missing_actual_hook_is_owned_work_not_random_fallback():
    picker = HookPicker("SOURCE_REQUIRED")
    with pytest.raises(SourceHookUnavailable, match="SOURCE_HOOK_PRODUCER_REQUIRED"):
        choose_source_hook(picker, _behavior(), [_hook("real#END")])
    assert len(picker.calls) == 1


def test_no_inspected_hook_requires_new_source_before_codegen():
    with pytest.raises(SourceHookUnavailable, match="no_inspected_callback"):
        choose_source_hook(HookPicker("real#END"), _behavior(), [])


def test_unknown_explicit_hook_is_rejected():
    with pytest.raises(ValueError, match="SOURCE_HOOK_NOT_IN_INSPECTED_ABI"):
        choose_source_hook(HookPicker("other#EVENT"),
                           {**_behavior(), "event_binding": "fabricated#EVENT"},
                           [_hook("real#END")])
