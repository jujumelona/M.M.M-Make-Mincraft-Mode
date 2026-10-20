"""Verified Fabric event ABI: no model-created listener signatures."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import re

from .java_operation_ir import parameter_types
from .construction_type_contracts import _source_set_owner_accessible

EVENT = "net.fabricmc.fabric.api.event.Event"
PREFIX = "net.fabricmc.fabric.api."


def callback_interfaces(symbols: Mapping, *, side: str = "COMMON") -> list[dict]:
    """Join public Event<Listener> fields to inspected functional interfaces.

    Fabric also declares events on holder classes (ServerTickEvents,
    PlayerBlockBreakEvents, entity/command/network packages). The field owner,
    field name and listener are separate identities. No class-name catalogue
    or assumed EVENT field can recover this relationship from erased types.
    """
    by_owner = defaultdict(list)
    classes = {}
    target_path = "src/client/java/" if str(side).upper() == "CLIENT" else "src/main/java/"
    registered = False
    for row in symbols.values():
        if not isinstance(row, Mapping) or row.get("provenance") == "construction_plan":
            continue
        row_side = str(row.get("side") or "COMMON").upper()
        if row_side != "COMMON" and row_side != str(side).upper():
            continue
        owner = row.get("owner")
        if not isinstance(owner, str) or not _source_set_owner_accessible(owner, target_path, row):
            continue
        if owner == EVENT and row.get("kind") == "METHOD" and row.get("name") == "register":
            if len(parameter_types(row)) == 1 and int(row.get("access") or 0) & 1:
                registered = True
        if isinstance(owner, str) and owner.startswith(PREFIX):
            by_owner[owner].append(row)
            if row.get("kind") == "CLASS" and int(row.get("access") or 0) & 1:
                classes[owner] = row
    if not registered:
        return []
    result = []
    for owner, rows in sorted(by_owner.items()):
        if owner not in classes:
            continue
        for event in sorted(rows, key=lambda row: str(row.get("name") or "")):
            if (event.get("kind") != "FIELD"
                    or int(event.get("access") or 0) & 9 != 9
                    or event.get("return_type") != EVENT):
                continue
            match = re.fullmatch(re.escape(EVENT) + r"<([\w.$]+)>",
                                 str(event.get("generic_return_type") or ""))
            if not match:
                continue
            listener = match[1].replace("$", ".")
            interface = classes.get(listener)
            if interface is None or not int(interface.get("access") or 0) & 512:
                continue
            # Reuse the existing bytecode-derived SAM contract, including
            # inherited interfaces. Hand-assembled indexes may instead expose
            # the one direct abstract public method without that annotation.
            sources = interface.get("functional_contract_sources")
            if sources:
                abstract = [symbols[key] for key in sources if key in symbols]
            else:
                abstract = [r for r in by_owner[listener] if r.get("kind") == "METHOD"
                            and int(r.get("access") or 0) & 1025 == 1025
                            and not int(r.get("access") or 0) & (8 | 64 | 4096)]
            if not abstract or (not sources and len(abstract) != 1):
                continue
            method = abstract[0]
            parameters = interface.get("functional_parameters", parameter_types(method))
            returns = interface.get("functional_return_type", method["return_type"])
            if not all(_source_set_owner_accessible(type_, target_path, symbols.get(type_))
                       for type_ in [*parameters, returns]):
                continue
            result.append({"event_id": owner + "#" + event["name"],
                           "event_owner": owner, "event_interface": listener,
                           "event_field": event["name"], "method": method["name"],
                           "parameters": parameters, "return_type": returns,
                           "throws": interface.get("functional_throws", method.get("throws", []))})
    return result
