"""Select portable whole-method evidence and produce target-typed helper owners.

Only pinned reference code is eligible. Old methods are *never* copied into
Fabric as source; MiMo authors a new typed operation graph on the target API.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

from .construction_task import ConstructionTask
from .construction_author import ask, obj, ID, TEXT
from .java_operation_ir import identifier


_PORT_CHOICE = obj({"port": {"type": "boolean"}, "reason": TEXT})
_PORT_SIGNATURE = obj({
    "return_type": ID,
    "static": {"type": "boolean"},
})
_PORT_PARAMETER = obj({
    "complete": {"type": "boolean"}, "name": TEXT, "type_ref": TEXT,
})


def select_port_tasks(router, parent_task, *, owner, references, symbols, exact_type, state=None):
    """One explicit decision per old method; install only new method declarations."""
    owners = []
    for reference in references:
        ref = reference["source_ref"]
        if state: state.enter('port_selection')
        selected = ask(router, "construction_port_reference_select", {
            "current_behavior": parent_task.intent,
            "source_ref": ref,
            "original_signature": reference.get("signature", ""),
            "migration_diagnostics": reference.get("migration_diagnostics", ""),
            **(state.payload(['port', 'skip']) if state else {}),
        }, _PORT_CHOICE,
        "Select port=true only if this exact pinned method is relevant to the "
        "requested behavior and can be reimplemented using inspected target API. "
        "Otherwise port=false; an unselected reference is never installed.")
        if not selected["port"]:
            continue
        member = "portedLogic" + hashlib.sha256(ref.encode()).hexdigest()[:20]
        if state: state.enter('port_signature')
        signature = ask(router, "construction_port_signature", {
            "source_ref": ref, "owner": owner,
            "original_signature": reference.get("signature", ""),
            "purpose": parent_task.intent,
            **(state.payload(['signature']) if state else {}),
        }, _PORT_SIGNATURE,
        "Choose one target Java method signature for porting the old method. "
        "Use only inspected types. Do not write Java code.")
        result_type = exact_type(signature["return_type"], symbols)
        if state:
            state.advance()
            state.enter('port_parameters')
        parameters = []
        while True:
            row = ask(router, "construction_port_parameter", {
                "source_ref": ref, "member": member,
                "signature": signature,
                "parameter_index": len(parameters), "parameters": parameters,
                **(state.payload(['parameter', 'complete']) if state else {}),
            }, _PORT_PARAMETER,
            "Provide exactly one parameter of the new target API helper at a time. "
            "Use complete=true after all needed parameters. No Java source.")
            if row["complete"]:
                break
            identifier(row["name"])
            resolved = exact_type(row["type_ref"], symbols, allow_void=False)
            if any(x["name"] == row["name"] for x in parameters):
                raise ValueError("CONSTRUCTION_PORT_PARAMETER_DUPLICATE")
            parameters.append({"name": row["name"], "type": resolved})
            if state: state.advance()
        symbol_id = owner + "#" + member + "(" + ",".join(p["type"] for p in parameters) + ")"
        if symbol_id in symbols:
            raise ValueError("CONSTRUCTION_PORT_SYMBOL_COLLISION")
        symbols[symbol_id] = {
            "kind": "METHOD", "owner": owner, "name": member,
            "parameters": parameters, "return_type": result_type,
            "visibility": "private", "access": 2,
            "static": signature["static"], "side": "COMMON",
            "provenance": "construction_plan", "body_owner": symbol_id,
            "purpose": parent_task.intent,
            "class_superclass": symbols[owner].get("superclass", ""),
            "class_interfaces": symbols[owner].get("interfaces", []),
        }
        if state: state.add_declaration(symbol_id)
        task = replace(
            parent_task, task_id="port_" + hashlib.sha256(
                (parent_task.task_id + ref).encode()).hexdigest()[:20],
            kind="port", owner_symbol=symbol_id,
            source_refs=({"source_ref": ref},),
            output_symbols=(symbol_id,),
            intent="Reimplement pinned source method " +
                   reference.get("signature", "") + " for " + parent_task.intent,
            verification_ids=(symbol_id + ".compile", symbol_id + ".behavior"),
        )
        owners.append((task, ref))
    return tuple(owners)
