"""Require the generated feature registration to bind an inspected callback.

Compilation proves Java types, not activation. A no-argument registrar which
merely calls a handler directly from onInitialize can compile yet never respond
to any gameplay event. Check the already-indexed target ABI before accepting
such generated source as a feature implementation.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence


def _listener_invokes_handler(compact_body: str, marker: str,
                              handler: str) -> bool:
    """Inspect register(...)'s callback argument, not unrelated method calls."""
    start = 0
    while True:
        position = compact_body.find(marker, start)
        if position < 0:
            return False
        cursor, depth = position + len(marker), 1
        begin = cursor
        while cursor < len(compact_body) and depth:
            char = compact_body[cursor]
            depth += (char == "(") - (char == ")")
            cursor += 1
        if depth == 0:
            listener = compact_body[begin:cursor - 1]
            if (listener == "this::" + handler or (
                "->" in listener and
                re.search(r"\bthis\." + re.escape(handler) + r"\(", listener)
            )):
                return True
        start = position + len(marker)


def validate_registered_event_sources(
    source: str,
    behaviors: Sequence[Mapping],
    inspected_events: Sequence[Mapping],
    *, materialized_events: Sequence[Mapping] = (),
) -> None:
    """Admit only registration of real, inspected Fabric Event fields.

    This is a necessary wiring check, NOT a playtest or proof of semantic
    correctness. Custom Mixin-created hooks require a separate materialized
    source owner and cannot be passed off as one of the inspected Fabric hooks.
    """
    from .construction_file_author import _lexical_code, _method_body

    masked = _lexical_code(source)
    compact = re.sub(r"\s+", "", masked)
    available = {str(row.get("event_id") or ""): row for row in inspected_events}
    for row in materialized_events:
        if (row.get("provenance") != "verified_source_site_mixin"
                or not row.get("event_id") or not row.get("event_owner")
                or row.get("event_field") != "EVENT"):
            raise ValueError("FEATURE_PROOF_CUSTOM_EVENT_RECEIPT_INVALID")
        if row["event_id"] in available:
            raise ValueError("FEATURE_PROOF_DUPLICATE_EVENT_ID")
        available[row["event_id"]] = row
    for behavior in behaviors:
        identity = str(behavior.get("behavior_id") or "")
        if not identity:
            raise ValueError("FEATURE_PROOF_BEHAVIOR_ID_REQUIRED")
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        registrar = "registerBehavior" + digest
        handler = "handleBehavior" + digest
        body = _method_body(masked, registrar)
        if body is None:
            raise ValueError("FEATURE_PROOF_REGISTRAR_MISSING: " + identity)
        if not re.search(r"\b" + re.escape(handler) + r"\b", body):
            raise ValueError("FEATURE_PROOF_HANDLER_NOT_CONNECTED: " + identity)
        explicit = str(behavior.get("event_binding") or "")
        eligible = ([available[explicit]] if explicit in available else
                    [] if explicit else list(available.values()))
        found = False
        compact_body = re.sub(r"\s+", "", body)
        for event in eligible:
            owner = str(event.get("event_owner") or "")
            field = str(event.get("event_field") or "")
            if not owner or not field:
                continue
            reference = owner + "." + field + ".register("
            simple = owner.rsplit(".", 1)[-1] + "." + field + ".register("
            markers = [reference]
            if "import" + owner + ";" in compact:
                markers.append(simple)
            if "importstatic" + owner + "." + field + ";" in compact:
                markers.append(field + ".register(")
            if any(_listener_invokes_handler(compact_body, current, handler)
                   for current in markers):
                found = True
                break
        if not found:
            raise ValueError("FEATURE_PROOF_INSPECTED_HOOK_NOT_REGISTERED: " + identity)
