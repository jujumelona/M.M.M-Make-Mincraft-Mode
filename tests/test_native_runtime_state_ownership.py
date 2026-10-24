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
    "player.inventory.currency", "current world", "game mode", "alien.alive",
    "Combat particles and effects", "available planets",
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
