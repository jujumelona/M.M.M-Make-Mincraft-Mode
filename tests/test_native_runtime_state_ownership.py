"""Native Minecraft state must never become disconnected persisted scalars."""

import pytest

from minecraft_mod_ai.construction_behavior_state import (
    _runtime_owned_state_reference,
    _validate_replayed_binding,
    reconcile_behavior_state,
)
from test_construction_behavior_state import BindingRouter, plan, variable


@pytest.mark.parametrize("name", [
    "Player health", "player health", "alien.health", "Alien entity health",
    "player hunger", "Weapon durability", "player armor durability",
    "Combat damage", "player.weapon_damage", "player position",
    "game mode", "alien.alive",
])
def test_native_state_is_classified_as_live_runtime(name):
    assert _runtime_owned_state_reference(name)
    router = BindingRouter()
    resolved = reconcile_behavior_state(router, plan(reads=[name], writes=[name]))
    behavior = resolved.request_scope["behavior_requirements"][0]
    assert behavior["state_reads"] == []
    assert behavior["state_writes"] == []
    assert behavior["runtime_state_reads"] == [name]
    assert behavior["runtime_state_writes"] == [name]
    assert resolved.request_scope["behavior_state_bindings"][name]["semantic_status"] == "host_native_runtime_owner"
    assert router.calls == []
    assert resolved.structured_sections["state_model"]["specification"]["variables"] == [variable()]


@pytest.mark.parametrize("name", [
    "ship_performance_level", "crew_level", "credits", "custom_ship_power",
    "spacecraft_construction_progress", "weapon_level",
    "player.inventory.currency", "world.resource_nodes.depleted",
    "entity.custom_energy", "world.planets.colonized",
    "current world", "Combat particles and effects", "available planets",
])
def test_custom_state_is_not_reclassified_as_vanilla(name):
    assert not _runtime_owned_state_reference(name)


def test_stale_native_scalar_binding_is_rejected_even_if_previously_saved():
    with pytest.raises(ValueError, match="BEHAVIOR_STATE_NATIVE_RUNTIME_OWNER_NOT_SCALAR"):
        _validate_replayed_binding("Player health", {
            "original_name": "Player health", "binding": "new_scalar_state",
            "state_name": "state_unsafe",
            "variable": {"name": "state_unsafe", "owner": "player"},
        }, [variable()])



def test_saved_legacy_native_scalar_is_migrated_without_reusing_fake_health():
    import hashlib

    original_name = "Player health"
    shadow = "state_" + hashlib.sha256(original_name.encode()).hexdigest()[:20]
    record = {"name": shadow, "owner": "player", "type": "int", "unit": "health",
              "default": "20", "domain": "nonnegative"}
    original = plan(reads=[shadow], writes=[shadow],
                    variables=[variable(), record])
    original.request_scope["behavior_state_bindings"] = {original_name: {
        "original_name": original_name, "binding": "new_scalar_state",
        "state_name": shadow, "variable": record,
        "semantic_status": "model_authored_unverified",
    }}
    before = original.to_dict()
    resolved = reconcile_behavior_state(BindingRouter(), original)
    assert original.to_dict() == before
    assert resolved.structured_sections["state_model"]["specification"]["variables"] == [variable()]
    behavior = resolved.request_scope["behavior_requirements"][0]
    assert behavior["state_reads"] == behavior["state_writes"] == []
    assert behavior["runtime_state_reads"] == behavior["runtime_state_writes"] == [original_name]
    assert resolved.request_scope["behavior_state_bindings"][original_name]["binding"] == "world_state"
    assert reconcile_behavior_state(BindingRouter(), resolved).to_dict() == resolved.to_dict()


def test_saved_native_scalar_migration_rejects_unrelated_declaration():
    import hashlib

    original_name = "Player health"
    shadow = "state_" + hashlib.sha256(original_name.encode()).hexdigest()[:20]
    original = plan(writes=[shadow], variables=[
        variable(), variable(shadow, "int", "9"),
    ])
    original.request_scope["behavior_state_bindings"] = {original_name: {
        "original_name": original_name, "binding": "new_scalar_state",
        "state_name": shadow, "variable": variable(shadow, "int", "20"),
    }}
    with pytest.raises(ValueError, match="BEHAVIOR_STATE_NATIVE_SCALAR_MIGRATION_UNSAFE"):
        reconcile_behavior_state(BindingRouter(), original)


def test_saved_heuristic_world_owner_reopened_and_canonicalized():
    # A saved run previously classified every player.inventory.* descendant
    # as vanilla even if the leaf is a custom mod-owned field.
    from test_construction_behavior_state import new_state_decisions

    reference = "player.inventory.currency"
    old = plan()
    behavior = old.request_scope["behavior_requirements"][0]
    behavior["state_reads"] = []
    behavior["state_writes"] = []
    behavior["runtime_state_reads"] = [reference]
    behavior["runtime_state_writes"] = [reference]
    old.request_scope["behavior_state_bindings"] = {reference: {
        "original_name": reference,
        "binding": "world_state",
        "observation": reference,
        "semantic_status": "host_native_runtime_owner",
    }}
    before = old.to_dict()
    decisions = [row for row in new_state_decisions() if "owner" not in row]
    router = BindingRouter(decisions)
    recovered = reconcile_behavior_state(router, old)
    assert all(call[1]["properties"].keys() != {"owner"} for call in router.calls)
    assert recovered.structured_sections["state_model"]["specification"]["variables"][-1]["owner"] == "player"
    assert old.to_dict() == before
    fixed = recovered.request_scope["behavior_requirements"][0]
    assert fixed.get("runtime_state_reads") == []
    assert fixed.get("runtime_state_writes") == []
    assert fixed["state_reads"] == fixed["state_writes"]
    assert len(fixed["state_writes"]) == 1
    assert recovered.request_scope["behavior_state_bindings"][reference]["binding"] == "new_scalar_state"
    assert reconcile_behavior_state(BindingRouter(), recovered).to_dict() == recovered.to_dict()
