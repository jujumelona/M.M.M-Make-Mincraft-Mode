"""Choose framework observation points from finite, queryable API pages."""
from __future__ import annotations

import re

from .incremental_java_navigation import record_navigation_transition
from .incremental_java_schema import SCHEMAS, choice

# Applies to choosing one framework observation point, not to the number of
# events or statements a generated Minecraft feature can implement.
_MAX_EVENT_SELECTION_NAVIGATION = 128

PAGE = 10


def _event_navigation_step(count):
    count += 1
    if count > _MAX_EVENT_SELECTION_NAVIGATION:
        raise RuntimeError("JAVA_EVENT_SELECTION_NAVIGATION_EXHAUSTED")
    return count


def _resolve_event_query(events, query, offset):
    """Never interpret a state/work-order page cursor as an event search."""
    query = str(query)
    matching = _matching(events, query)
    if query.startswith("@") or not matching:
        # A different lookup namespace or zero-result search cannot become
        # an empty event-choice page. Show the REAL inspected event inventory.
        return "*", 0, _matching(events, "*"), "event_query_not_in_catalog"
    if offset >= len(matching):
        return query, 0, matching, "event_offset_out_of_range"
    return query, offset, matching, ""


def select_event(builder):
    events = builder.api.callbacks()
    if not events:
        builder.rejection = {"operand_diagnostic": "No inspected callback exists for this source unit; materialize an occurrence producer first."}
        return None
    query, offset = "*", 0
    if len(events) > PAGE:
        page = builder.ask(SCHEMAS["lookup"], "register_lookup", {"lookup_instruction": (
            "Search ONLY the Fabric EVENT index. Never submit an @state_access, "
            "@work_order, or other API cursor as an event query. Use * to browse "
            "all inspected events; the listener must still implement the original trigger.")})
        query, offset = page["query"], page["offset"]
    transitions = {}
    navigation_steps = 0
    invalid_searches = 0
    while True:
        navigation_steps = _event_navigation_step(navigation_steps)
        query, offset, matching, issue = _resolve_event_query(events, query, offset)
        if issue:
            invalid_searches += 1
            if invalid_searches > 4:
                from .source_occurrence_error import SourceHookUnavailable
                raise SourceHookUnavailable(
                    str((builder.member.get("behavior") or {}).get("behavior_id") or builder.member["id"]),
                    reason="event_query_namespace_stalled", inspected=len(events))
        available = matching[offset:offset + PAGE]
        indexed = {event["event_id"]: event for event in available}
        options = [*indexed, "@search", "@cancel"]
        if offset + PAGE < len(matching):
            options.append("@next")
        selected = builder.ask(choice("event_id", options), "register", {"event_index": [
            {"event_id": event["event_id"], "inputs": event["parameters"]} for event in available],
            "event_query": query, "event_offset": offset, "event_total": len(matching),
            "event_search_diagnostic": issue})["event_id"]
        if selected in indexed:
            return indexed[selected]
        if selected == "@cancel":
            return None
        try:
            record_navigation_transition(
                transitions, (query, offset), selected,
                error="JAVA_EVENT_SELECTION_NAVIGATION_CYCLE",
            )
        except RuntimeError as exc:
            if not str(exc).startswith("JAVA_EVENT_SELECTION_NAVIGATION_CYCLE:"):
                raise
            # A cycling menu is a declined attempt, not a compiler/runtime
            # failure. The registrar's alternate producer strategy owns it.
            builder.rejection = {"operand_diagnostic": str(exc)}
            return None
        if selected == "@next":
            offset += PAGE
        else:
            page = builder.ask(SCHEMAS["lookup"], "register_lookup")
            query, offset = page["query"], page["offset"]



def select_original_occurrence(builder):
    """Bind the original runtime occurrence in one checked host decision.

    A registrar owns *wiring*, not event-index navigation. Asking the small
    coder to page through all callbacks left production-logs (13) cycling
    around an unrelated state cursor for 96 actions. An explicit ABI binding
    needs zero inference; otherwise a single semantic planner selection
    either supplies an inspected event or reports the missing source producer.
    Never silently replace the requested occurrence with an arbitrary hook.
    """
    from .source_occurrence_error import SourceHookUnavailable
    from .source_owned_hook import choose_source_hook

    behavior = (builder.member or {}).get("behavior") or {}
    behavior_id = str(behavior.get("behavior_id") or "")
    if not behavior_id:
        raise ValueError("SOURCE_HOOK_BEHAVIOR_REQUIRED")
    inspected = builder.api.callbacks()
    if not inspected:
        raise SourceHookUnavailable(
            behavior_id, reason="no_inspected_callback_requires_source_producer",
            inspected=0,
        )
    return choose_source_hook(builder.router, behavior, inspected)


def _matching(events, query):
    terms = re.findall(r"[a-z0-9_]+", str(query).lower())
    if not terms:
        return sorted(events, key=lambda event: event["event_id"])
    return sorted((event for event in events if all(term in (
        event["event_id"] + " " + " ".join(event["parameters"])).lower() for term in terms)),
        key=lambda event: event["event_id"])
