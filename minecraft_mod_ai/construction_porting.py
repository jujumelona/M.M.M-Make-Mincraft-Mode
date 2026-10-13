"""Select portable whole-method evidence and produce target-typed helper owners.

Only pinned reference code is eligible. Old methods are *never* copied into
Fabric as source; MiMo authors a new typed operation graph on the target API.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

from .construction_task import ConstructionTask
from .construction_author import ask, obj, TEXT
from .construction_declaration_choices import (choose_type, parameter_schema, with_binding_lookups,
                                             choice_actions, consume_lookup_choice)
from .construction_state import STATE_INSTRUCTION
from .java_operation_ir import identifier


_PORT_CHOICE = obj({"port": {"type": "boolean"}, "reason": TEXT})
_PORT_SIGNATURE = obj({"static": {"type": "boolean"}})


def select_port_tasks(router, parent_task, *, owner, references, symbols, exact_type, state):
    """One explicit decision per old method; install only new method declarations."""
    owners = []
    for reference in references:
        ref = reference["source_ref"]
        state.enter('port_selection')
        while True:
            schema = with_binding_lookups(state, _PORT_CHOICE)
            selected = ask(router, "construction_port_reference_select", {
                "current_behavior": parent_task.intent,
                "source_ref": ref,
                "original_signature": reference.get("signature", ""),
                "migration_diagnostics": reference.get("migration_diagnostics", ""),
                **state.payload(choice_actions(schema, default_action='port_decision')),
            }, schema,
            "Select port=true only if this exact pinned method is relevant to the "
            "requested behavior and can be reimplemented using inspected target API. "
            "Otherwise port=false; an unselected reference is never installed. " + STATE_INSTRUCTION)
            if consume_lookup_choice(state, selected): continue
            break
        state.advance()
        if not selected["port"]:
            continue
        member = "portedLogic" + hashlib.sha256(ref.encode()).hexdigest()[:20]
        current = {
            "source_ref": ref, "owner": owner,
            "original_signature": reference.get("signature", ""),
            "purpose": parent_task.intent,
        }
        result_type = choose_type(router, state, current, phase='port_return_type', allow_void=True)
        state.enter('port_signature')
        while True:
            schema = with_binding_lookups(state,
                obj({**_PORT_SIGNATURE['properties'], 'return_type': {'const': result_type}}))
            signature = ask(router, "construction_port_signature", {
                **current, 'return_type': result_type,
                **state.payload(choice_actions(schema, default_action='signature')),
            }, schema,
            "Choose one target Java method signature for porting the old method. "
            "Use only inspected types. Do not write Java code. " + STATE_INSTRUCTION)
            if consume_lookup_choice(state, signature): continue
            break
        exact_type(result_type, symbols)
        state.advance()
        state.enter('port_parameters')
        parameters = []
        while True:
            schema = parameter_schema(parameters, state=state)
            row = ask(router, "construction_port_parameter", {
                "source_ref": ref, "member": member,
                "signature": signature,
                "parameter_index": len(parameters), "parameters": parameters,
                **state.payload(choice_actions(schema)),
            }, schema,
            "Provide exactly one parameter of the new target API helper at a time. "
            "Use complete=true after all needed parameters. No Java source. " + STATE_INSTRUCTION)
            if consume_lookup_choice(state, row): continue
            if row["complete"]:
                break
            identifier(row["name"])
            resolved = choose_type(router, state, {
                **current, 'signature': signature, 'parameters': parameters, 'parameter_name': row['name'],
            }, phase='port_parameter_type')
            exact_type(resolved, symbols, allow_void=False)
            if any(x["name"] == row["name"] for x in parameters):
                raise ValueError("CONSTRUCTION_PORT_PARAMETER_DUPLICATE")
            parameters.append({"name": row["name"], "type": resolved})
            state.advance()
            state.enter('port_parameters')
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
        state.add_declaration(symbol_id)
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
