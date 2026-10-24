"""A newly produced GUI/controller source is a real hook site, not a guessed Fabric API.

Log 34: a model chose SOURCE_REQUIRED because construction consulted only the
original Fabric classpath, excluding source built by an earlier content shard.
"""
from copy import deepcopy

from minecraft_mod_ai.construction_source_symbols import merge_materialized_source_sites
from minecraft_mod_ai.source_owned_custom_event import plan_verified_custom_event
from minecraft_mod_ai.source_owned_site_choice import shortlist_verified_sites


def _unit(kind, **changes):
    row = {
        "kind": kind, "owner": "example.ShipyardController",
        "package": "example", "path": "src/main/java/example/ShipyardController.java",
        "source_sha256": "3" * 64, "side": "COMMON",
        "static": False, "modifiers": ["public"], "imports": [],
        "return_type": "void", "parameters": [], "throws": [],
    }
    row.update(changes)
    return row


def test_real_new_gui_method_is_available_to_source_site_adapter():
    original = {"java.lang.Object": {
        "kind": "CLASS", "owner": "java.lang.Object", "name": "Object",
        "access": 1,
    }}
    source_pages = {"shipyard": [
        _unit("class", name="ShipyardController"),
        _unit("method", name="placeSection", has_body=True),
    ]}
    saved = deepcopy(original)
    symbols = merge_materialized_source_sites(source_pages, original)
    assert original == saved
    ref = "example.ShipyardController#placeSection()"
    assert symbols[ref]["provenance"] == "materialized_source"
    assert symbols[ref]["source_hash"] == "3" * 64
    candidates = shortlist_verified_sites({
        "behavior_id": "build_section",
        "trigger": "player places spacecraft section",
        "intent": "placeSection",
        "effects": "places the spacecraft section",
    }, symbols)
    assert [row["ref"] for row in candidates] == [ref]
    custom = plan_verified_custom_event(
        symbols=symbols, source_site_ref=ref,
        behavior_id="build_section", package="example.generated",
        mod_id="shipyard",
    )
    assert custom["callback"]["provenance"] == "verified_source_site_mixin"
    assert "placeSection()V" in custom["source_descriptor"]


def test_materialized_source_cannot_replace_existing_verified_class():
    classpath = {
        "example.ShipyardController": {
            "kind": "CLASS", "name": "ShipyardController",
            "owner": "example.ShipyardController", "access": 1,
            "provenance": "verified_classpath",
        }
    }
    data = {"shipyard": [
        _unit("class", name="ShipyardController"),
        _unit("method", name="fictitiousMethod"),
    ]}
    assert merge_materialized_source_sites(data, classpath) == classpath


def test_unresolved_source_signature_cannot_become_mixin_target():
    data = {"shipyard": [
        _unit("class", name="ShipyardController"),
        _unit("method", name="placeSection", parameters=[
            {"name": "invented", "type": "example.NotYetImplemented"},
        ]),
    ]}
    result = merge_materialized_source_sites(data, {})
    assert not any(
        row.get("kind") == "METHOD" and row.get("name") == "placeSection"
        for row in result.values()
    )
