"""No arbitrary game occurrence is invented when Fabric lacks a callback."""
import pytest

from minecraft_mod_ai.source_owned_hook import SourceHookUnavailable
from minecraft_mod_ai.structured_output import StructuredOutputValidationError
from minecraft_mod_ai.source_owned_site_choice import (
    shortlist_verified_sites, choose_verified_source_site,
)


SYMBOLS = {
    "local.class": {"kind": "CLASS", "owner": "example.TradingStation"},
    "local.purchase": {
        "kind": "METHOD", "owner": "example.TradingStation",
        "name": "purchaseModule", "parameters": ["int"], "return_type": "boolean",
        "access": 1, "source_path": "src/main/java/example/TradingStation.java",
    },
    "other.class": {"kind": "CLASS", "owner": "example.PlayerLifecycle"},
    "other.login": {
        "kind": "METHOD", "owner": "example.PlayerLifecycle",
        "name": "onPlayerLogin", "parameters": [], "return_type": "void",
        "access": 1,
    },
    "other.tick": {
        "kind": "METHOD", "owner": "example.TradingStation",
        "name": "onServerTick", "parameters": [], "return_type": "void",
        "access": 1,
    },
    "static.buy": {
        "kind": "METHOD", "owner": "example.TradingStation",
        "name": "purchase", "parameters": [], "return_type": "void",
        "access": 9,
    },
}


class Picker:
    def __init__(self, choice):
        self.choice = choice
        self.calls = []
    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append((role, data_name, parameters))
        return {"source_site_ref": self.choice}


def test_source_site_inventory_excludes_arbitrary_tick_join_and_static():
    behavior = {"behavior_id": "trade", "trigger": "Purchase a station module"}
    result = shortlist_verified_sites(behavior, SYMBOLS)
    assert [v["ref"] for v in result] == ["local.purchase"]
    assert result[0]["parameter_types"] == ["int"]
    assert result[0]["source_path"] == "src/main/java/example/TradingStation.java"


def test_source_site_is_a_bounded_model_choice_over_actual_java_symbols():
    behavior = {"behavior_id": "trade", "trigger": "Purchase a station module"}
    router = Picker("local.purchase")
    assert choose_verified_source_site(router, behavior, SYMBOLS) == "local.purchase"
    assert router.calls[0][0] == "planner"
    assert router.calls[0][1] == "select_inspected_source_producer_site"
    assert set(router.calls[0][2]["properties"]["source_site_ref"]["enum"]) == {
        "local.purchase", "SOURCE_REQUIRED",
    }


def test_unsupported_site_remains_explicit_unimplemented_producer():
    behavior = {"behavior_id": "warp", "trigger": "Activate hyperspace and travel"}
    router = Picker("other.login")
    with pytest.raises(SourceHookUnavailable, match="no_relevant_inspected_source_site"):
        choose_verified_source_site(router, behavior, SYMBOLS)
    assert not router.calls


def test_model_cannot_add_an_uninspected_site_to_the_candidate_set():
    behavior = {"behavior_id": "trade", "trigger": "Purchase a station module"}
    with pytest.raises(StructuredOutputValidationError):
        choose_verified_source_site(Picker("invented#EVENT"), behavior, SYMBOLS)
    with pytest.raises(SourceHookUnavailable, match="custom_source_site_not_resolved"):
        choose_verified_source_site(Picker("SOURCE_REQUIRED"), behavior, SYMBOLS)


def test_short_java_occurrence_verbs_remain_searchable():
    # Attack/use/trading hooks are frequently three-letter methods.
    # Excluding them meant the planner never even saw the real source site.
    symbols = {
        "c": {"kind": "CLASS", "owner": "example.PlayerInteractions"},
        "use": {"kind": "METHOD", "owner": "example.PlayerInteractions",
                "name": "useItem", "parameters": ["example.Item"], "return_type": "void",
                "access": 1},
        "hit": {"kind": "METHOD", "owner": "example.PlayerInteractions",
                "name": "hitTarget", "parameters": ["example.Target"], "return_type": "void",
                "access": 1},
        "buy": {"kind": "METHOD", "owner": "example.PlayerInteractions",
                "name": "buy", "parameters": ["int"], "return_type": "void",
                "access": 1},
    }
    for verb, target in (("use", "use"), ("hit", "hit"), ("buy", "buy")):
        sites = shortlist_verified_sites(
            {"behavior_id": verb, "trigger": f"Player chooses to {verb} the target"},
            symbols,
        )
        assert target in {row["ref"] for row in sites}
    # Retaining a short word is candidate retrieval, not permission to invent a
    # hook: source selection still requires the exact inspected method.
    with pytest.raises(SourceHookUnavailable, match="custom_source_site_not_resolved"):
        choose_verified_source_site(
            Picker("SOURCE_REQUIRED"),
            {"behavior_id": "buy", "trigger": "Player buys a module"}, symbols,
        )


def test_planet_colonization_does_not_offer_lwjgl_or_space_getters_as_event_sites():
    behavior = {"behavior_id": "colonize", "trigger":
                "Player reaches planet location and starts colonization",
                "intent": "Colonize available planets"}
    jar = "/gradle/caches/org.lwjgl/lwjgl-vulkan.jar"
    mc = "/gradle/loom-cache/minecraft-common-26.2.jar"
    rows = {
        "lwjgl": {"kind": "CLASS",
                  "owner": "org.lwjgl.vulkan.VkPipelineSampleLocationsStateCreateInfoEXT"},
        "lwjgl_method": {
            "kind": "METHOD",
            "owner": "org.lwjgl.vulkan.VkPipelineSampleLocationsStateCreateInfoEXT",
            "name": "sampleLocationsEnable", "parameters": [], "access": 1,
            "source_path": jar,
        },
        "minecraft": {"kind": "CLASS",
                      "owner": "net.minecraft.world.entity.Entity"},
        "minecraft_getter": {
            "kind": "METHOD", "owner": "net.minecraft.world.entity.Entity",
            "name": "getAvailableSpaceBelow", "parameters": ["double"],
            "access": 1, "source_path": mc,
        },
        "gameplay": {"kind": "CLASS", "owner": "example.PlanetController"},
        "gameplay_action": {
            "kind": "METHOD", "owner": "example.PlanetController",
            "name": "colonizePlanet", "parameters": [], "access": 1,
            "source_path": "src/main/java/example/PlanetController.java",
        },
    }
    assert [row["ref"] for row in shortlist_verified_sites(
        behavior, rows)] == ["gameplay_action"]
    assert shortlist_verified_sites(
        behavior, {k: v for k, v in rows.items() if k != "gameplay_action"}) == []
