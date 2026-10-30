"""Event lookup must not consume an unrelated API/state pagination cursor."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.incremental_java_events import (
    _resolve_event_query, select_event,
)


def _events(count=12):
    return [
        {"event_id": f"example.Event#{i:04d}",
         "event_owner": "example.Event", "event_field": f"EVENT_{i}",
         "parameters": ["example.Player"], "return_type": "void"}
        for i in range(count)
    ]


@pytest.mark.parametrize("query,offset,issue", [
    ("@state_access:selection_d4", 0, "event_query_not_in_catalog"),
    ("@work_order:effects", 0, "event_query_not_in_catalog"),
    ("no-such-event", 0, "event_query_not_in_catalog"),
    ("*", 999, "event_offset_out_of_range"),
])
def test_foreign_or_empty_lookup_cannot_create_an_unselectable_event_page(query, offset, issue):
    events = _events()
    resolved, index, rows, diagnostic = _resolve_event_query(events, query, offset)
    assert rows and len(rows) == len(events)
    assert diagnostic == issue
    assert index == 0
    assert resolved == "*"


def test_valid_exact_event_search_and_offset_remain_usable():
    events = _events(25)
    query, offset, matches, issue = _resolve_event_query(events, "example.Event#0021", 0)
    assert not issue
    assert query == "example.Event#0021" and offset == 0
    assert len(matches) == 1 and matches[0]["event_id"].endswith("0021")


def test_actual_register_page_recovers_state_access_cursor_from_production_log():
    events = _events()

    class Api:
        def callbacks(self):
            return events

    class Builder:
        api = Api()
        member = {"id": "m1", "behavior": {"behavior_id": "b1"}}
        rejection = {}
        queries = []
        register_pages = []

        def ask(self, schema, suffix, extra=None):
            if suffix == "register_lookup":
                self.queries.append(extra)
                return {"query": "@state_access:selection_72e1124164963c8262832094", "offset": 0}
            assert suffix == "register"
            self.register_pages.append(extra)
            assert extra["event_total"] == 12
            assert extra["event_query"] == "*"
            assert extra["event_index"]
            assert extra["event_search_diagnostic"] == "event_query_not_in_catalog"
            return {"event_id": extra["event_index"][0]["event_id"]}

    builder = Builder()
    chosen = select_event(builder)
    assert chosen["event_id"] == events[0]["event_id"]
    assert len(builder.queries) == 1
    assert len(builder.register_pages) == 1


def test_world_registrar_does_not_do_unrelated_java_api_lookup():
    from minecraft_mod_ai.incremental_java_work import work_options
    from test_incremental_java_work_ownership import _world_builder

    builder = _world_builder()
    actions = work_options(builder, builder.member, builder.scope, [], [],
                           ["register", "lookup", "method", "call", "complete"])
    assert actions == ["register"]
