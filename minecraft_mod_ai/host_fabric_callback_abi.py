"""Verified Fabric event ABI: no model-created listener signatures."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .java_operation_ir import parameter_types

EVENT = "net.fabricmc.fabric.api.event.Event"
PREFIX = EVENT.rsplit(".", 1)[0] + "."


def callback_interfaces(symbols: Mapping) -> list[dict]:
    by_owner = defaultdict(list)
    registered = False
    for row in symbols.values():
        if not isinstance(row, Mapping) or row.get("provenance") == "construction_plan":
            continue
        owner = row.get("owner")
        if owner == EVENT and row.get("kind") == "METHOD" and row.get("name") == "register":
            if len(parameter_types(row)) == 1 and int(row.get("access") or 0) & 1:
                registered = True
        if isinstance(owner, str) and owner.startswith(PREFIX):
            by_owner[owner].append(row)
    if not registered:
        return []
    result = []
    for owner, rows in sorted(by_owner.items()):
        interface = next((r for r in rows if r.get("kind") == "CLASS"
                          and int(r.get("access") or 0) & 512), None)
        event = next((r for r in rows if r.get("kind") == "FIELD"
                      and r.get("name") == "EVENT"
                      and int(r.get("access") or 0) & 9 == 9
                      and str(r.get("return_type") or "").startswith(EVENT)), None)
        abstract = [r for r in rows if r.get("kind") == "METHOD"
                    and int(r.get("access") or 0) & 1024
                    and not int(r.get("access") or 0) & 8]
        if interface is None or event is None or len(abstract) != 1:
            continue
        method = abstract[0]
        result.append({"event_interface": owner, "event_field": "EVENT",
                       "method": method["name"],
                       "parameters": parameter_types(method),
                       "return_type": method["return_type"]})
    return result
