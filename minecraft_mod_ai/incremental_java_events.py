"""Choose framework observation points from finite, queryable API pages."""
from __future__ import annotations

import re

from .incremental_java_navigation import record_navigation_transition
from .incremental_java_schema import SCHEMAS, choice

# Applies to choosing one framework observation point, not to the number of
# events or statements a generated Minecraft feature can implement.
_MAX_EVENT_SELECTION_NAVIGATION = 128

PAGE = 10


def select_event(builder):
    events = builder.api.callbacks()
    if not events:
        builder.rejection = {"operand_diagnostic": "No indexed Event fields; author the interaction with call/lambda or new helpers."}
        return None
    query, offset = "*", 0
    if len(events) > PAGE:
        page = builder.ask(SCHEMAS["lookup"], "register_lookup", {"lookup_instruction": (
            "Query a framework observation API name or input type, not the name of a completed gameplay feature. "
            "Use * to browse all indexed events. The callback body will implement the domain condition.")})
        query, offset = page["query"], page["offset"]
    transitions = {}
    navigation_steps = 0
    while True:
        navigation_steps += 1
        if navigation_steps > _MAX_EVENT_SELECTION_NAVIGATION:
            raise RuntimeError("JAVA_EVENT_SELECTION_NAVIGATION_EXHAUSTED")
        matching = _matching(events, query)
        available = matching[offset:offset + PAGE]
        indexed = {event["event_id"]: event for event in available}
        options = [*indexed, "@search", "@cancel"]
        if offset + PAGE < len(matching):
            options.append("@next")
        selected = builder.ask(choice("event_id", options), "register", {"event_index": [
            {"event_id": event["event_id"], "inputs": event["parameters"]} for event in available],
            "event_query": query, "event_offset": offset, "event_total": len(matching)})["event_id"]
        if selected in indexed:
            return indexed[selected]
        if selected == "@cancel":
            return None
        record_navigation_transition(
            transitions, (query, offset), selected,
            error="JAVA_EVENT_SELECTION_NAVIGATION_CYCLE",
        )
        if selected == "@next":
            offset += PAGE
        else:
            page = builder.ask(SCHEMAS["lookup"], "register_lookup")
            query, offset = page["query"], page["offset"]


def _matching(events, query):
    terms = re.findall(r"[a-z0-9_]+", str(query).lower())
    if not terms:
        return sorted(events, key=lambda event: event["event_id"])
    return sorted((event for event in events if all(term in (
        event["event_id"] + " " + " ".join(event["parameters"])).lower() for term in terms)),
        key=lambda event: event["event_id"])
