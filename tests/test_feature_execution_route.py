"""Capability routing must never confuse planned state names with live producers."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.feature_execution_route import select_feature_execution_route


def _callbacks():
    return [{"event_id": "test:event", "parameters": ["example.World"],
             "event_owner": "example.Events", "event_field": "ON_TICK",
             "return_type": "void"}]


def test_missing_live_producer_is_source_implementation_obligation():
    result = select_feature_execution_route(
        [{"behavior_id": "feature-a", "runtime_state_reads": ["custom_selection"]}],
        {"custom_selection": {"binding": "world_state",
                              "semantic_status": "host_classified_unbound_read"}},
        {"example.World": {"kind": "CLASS", "owner": "example.World", "access": 1}},
        _callbacks(),
    )
    assert result["route"] == "source_owned_java"
    assert result["producer_obligations"][0]["name"] == "custom_selection"
    assert result["verified_behavior_ids"] == []


def test_proven_getter_with_explicit_real_hook_retains_typed_rule():
    symbols = {
        "example.World": {"kind": "CLASS", "owner": "example.World", "access": 1},
        "readyGetter": {
            "kind": "METHOD", "owner": "example.World", "access": 1,
            "name": "isReady", "parameters": [], "return_type": "boolean",
        },
    }
    result = select_feature_execution_route(
        [{"behavior_id": "feature-b", "event_binding": "test:event",
          "runtime_state_reads": ["ready"]}],
        {"ready": {"binding": "world_state", "verified_symbol_ref": "readyGetter"}},
        symbols, _callbacks(),
    )
    assert result["route"] == "typed_rule"
    assert result["producer_obligations"] == []
    assert result["verified_behavior_ids"] == ["feature-b"]


def test_unknown_explicit_hook_not_rewritten_as_tick():
    with pytest.raises(ValueError, match="FEATURE_EXECUTION_EXPLICIT_HOOK_NOT_INSPECTED"):
        select_feature_execution_route(
            [{"behavior_id": "feature-c", "event_binding": "missing:event"}],
            {}, {}, _callbacks(),
        )


def test_unnamed_occurrence_requires_source_ownership():
    result = select_feature_execution_route(
        [{"behavior_id": "feature-d"}], {},
        {"example.World": {"kind": "CLASS", "owner": "example.World", "access": 1}},
        _callbacks(),
    )
    assert result["route"] == "source_owned_java"
    assert result["producer_obligations"][0]["reason"] == "occurrence_hook_not_owned"



def test_native_observable_trigger_uses_checked_rule_authoring_not_unbounded_java():
    symbols = {"example.World": {"kind": "CLASS", "owner": "example.World",
                                  "access": 1}}
    result = select_feature_execution_route(
        [{"behavior_id": "time-threshold", "trigger": "World time reaches boundary",
          "intent": "Run a time-bound effect", "effects": "Update world state",
          "runtime_state_reads": []}],
        {}, symbols, _callbacks())
    assert result["route"] == "typed_rule"
    assert result["producer_obligations"] == []
    # The real rule author must still prove occurrence and effects.
    assert result["verified_behavior_ids"] == ["time-threshold"]


def test_unknown_runtime_source_cannot_use_rule_even_if_trigger_is_present():
    symbols = {"example.World": {"kind": "CLASS", "owner": "example.World",
                                  "access": 1}}
    result = select_feature_execution_route(
        [{"behavior_id": "missing-observer", "trigger": "When user selects a section",
          "runtime_state_reads": ["selected_section"]}],
        {"selected_section": {"binding": "world_state",
                               "semantic_status": "host_classified_unbound_read"}},
        symbols, _callbacks())
    assert result["route"] == "source_owned_java"
    assert any(row.get("name") == "selected_section"
               for row in result["producer_obligations"])
