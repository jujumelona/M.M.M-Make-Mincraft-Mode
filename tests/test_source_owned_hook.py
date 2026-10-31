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


def test_colonization_world_transition_cannot_use_damage_event():
    from minecraft_mod_ai.structured_output import StructuredOutputValidationError

    behavior = {
        "behavior_id": "colonize",
        "trigger": ("Player reaches a planet or space location and "
                    "initiates colonization"),
        "intent": "Colonize available planets",
        "runtime_state_reads": ["Player location", "Available planets",
                                "Colonization prerequisites"],
    }
    damaged = {
        "event_id": "net.fabricmc.fabric.api.entity.event.v1.ServerLivingEntityEvents#AFTER_DAMAGE",
        "parameters": ["net.minecraft.world.entity.LivingEntity",
                       "net.minecraft.world.damagesource.DamageSource"],
    }
    changed = {
        "event_id": (
            "net.fabricmc.fabric.api.entity.event.v1."
            "ServerEntityLevelChangeEvents#AFTER_PLAYER_CHANGE_LEVEL"
        ),
        "parameters": ["net.minecraft.server.level.ServerPlayer",
                       "net.minecraft.server.level.ServerLevel",
                       "net.minecraft.server.level.ServerLevel"],
    }
    picker = HookPicker(changed["event_id"])
    chosen = choose_source_hook(picker, behavior, [damaged, changed])
    assert chosen == changed
    offered = picker.calls[0][2]["properties"]["event_id"]["enum"]
    assert changed["event_id"] in offered
    assert damaged["event_id"] not in offered
    with pytest.raises(StructuredOutputValidationError):
        choose_source_hook(
            HookPicker(damaged["event_id"]), behavior, [damaged, changed])


def test_unobservable_world_change_requires_real_source_not_arbitrary_event():
    behavior = {
        "behavior_id": "change",
        "trigger": "Player enters another world",
    }
    with pytest.raises(SourceHookUnavailable, match="no_matching_original_occurrence_callback"):
        choose_source_hook(HookPicker("bad#DAMAGE"), behavior, [{
            "event_id": "bad#DAMAGE", "parameters": [
                "net.minecraft.world.entity.LivingEntity"],
        }])


def test_explicit_world_transition_cannot_bind_unrelated_damage_abi():
    behavior = {
        "behavior_id": "colonize",
        "trigger": "Player reaches another planet and initiates colonization",
        "event_binding": "real#AFTER_DAMAGE",
    }
    bad = {
        "event_id": "real#AFTER_DAMAGE",
        "parameters": ["net.minecraft.world.entity.LivingEntity"],
    }
    with pytest.raises(SourceHookUnavailable, match="explicit_hook_cannot_observe_original_occurrence"):
        choose_source_hook(HookPicker("real#AFTER_DAMAGE"), behavior, [bad])



def _real_hook(owner, field, *, params=()):
    return {
        "event_id": owner + "#" + field,
        "event_owner": owner, "event_field": field,
        "event_interface": owner + ".Listener", "method": "invoke",
        "parameters": list(params), "return_type": "void",
    }


def test_source_owned_production_never_chooses_post_kill_for_ongoing_combat():
    killed = _real_hook(
        "net.fabricmc.fabric.api.entity.event.v1.ServerEntityCombatEvents",
        "AFTER_KILLED_OTHER_ENTITY")
    attacked = _real_hook(
        "net.fabricmc.fabric.api.event.player.AttackEntityCallback",
        "EVENT", params=["net.minecraft.world.entity.player.Player"])
    behavior = {
        "behavior_id": "combat",
        "source_quote": "외계인과 싸움",
        "intent": "Fight aliens in space",
        "trigger": "Alien encounter in space",
        "effects": "Combat with aliens",
        "runtime_state_writes": ["Combat damage", "Player health", "Weapon durability"],
    }
    picker = HookPicker(attacked["event_id"])
    assert choose_source_hook(picker, behavior, [killed, attacked]) == attacked
    offered = picker.calls[0][2]["properties"]["event_id"]["enum"]
    assert killed["event_id"] not in offered
    assert attacked["event_id"] in offered
    with pytest.raises(Exception):
        choose_source_hook(HookPicker(killed["event_id"]), behavior, [killed, attacked])


def test_source_owned_mining_requires_real_block_break_not_middle_click():
    mined = _real_hook(
        "net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents", "AFTER")
    picked = _real_hook(
        "net.fabricmc.fabric.api.event.player.PlayerPickItemEvents", "BLOCK")
    b = {
        "behavior_id": "mining",
        "trigger": "Player mines ore to gather materials",
        "intent": "Gather resources",
        "effects": "Add collected minerals",
    }
    picker = HookPicker(mined["event_id"])
    assert choose_source_hook(picker, b, [picked, mined]) == mined
    assert not picker.calls  # exact inspected AFTER-break is host bound


def test_source_owned_explicit_death_occurrence_preserved():
    killed = _real_hook(
        "net.fabricmc.fabric.api.entity.event.v1.ServerEntityCombatEvents",
        "AFTER_KILLED_OTHER_ENTITY")
    b = {"behavior_id": "kill_reward",
         "trigger": "After an alien is killed",
         "effects": "Give a reward"}
    assert choose_source_hook(
        HookPicker(killed["event_id"]), b, [killed]) == killed
