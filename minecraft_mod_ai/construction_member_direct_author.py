"""Complete Java method author for known, host-owned declarations.

This replaces the one-operation-per-model-call loop for independent canonical
Java methods. Stateful/donor/port jobs retain their stricter legacy author
until a grounded state scaffold makes them eligible.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .construction_direct_writer import (
    _claim_checkpoint, _digest, _evidence, _read_checkpoint, _save_checkpoint,
)
from .first_pass_java_declarations import declaration_frame, render_declaration


def eligible_method(member_task, symbols, context, *, owned_records=(), port_source_ref="") -> bool:
    member = symbols.get(member_task.owner_symbol, {})
    if member.get("kind") != "METHOD":
        return False
    if (owned_records or port_source_ref or context.get("donor_callable_symbols")
            or context.get("direct_method_forbidden")):
        return False
    if context.get("reference_port_obligations"):
        return False
    if member_task.state_reads or member_task.state_writes:
        return False
    if member.get("body_owner") == "host_lifecycle":
        return False
    # The host must already own a complete declaration signature. No guessed
    # Java types, no model-selected overload lists.
    if (not member.get("name") or not member.get("return_type")
            or not isinstance(member.get("parameters"), (list, tuple))):
        return False
    return True


def compose_method(router, member_task, context, *, owner_name: str) -> str:
    member = context["symbols"][member_task.owner_symbol]
    metadata = {
        "visibility": member.get("visibility", "private"),
        "static": bool(member.get("static", False)),
        "final": bool(member.get("final", False)),
        "return_type": member["return_type"],
        "parameters": list(member["parameters"]),
        "throws": list(member.get("throws", ())),
        "annotations": list(member.get("annotations", ())),
        "type_parameters": member.get("type_parameters", ""),
    }
    frame = declaration_frame("method", member["name"], metadata)
    spec = {
        "task": member_task.to_dict(),
        "owner": member["owner"],
        "declaration": metadata,
        "target": context["target"],
        "prerequisites": context["prerequisite_hashes"],
        "authoring_contract": context.get("authoring_contract", {}),
        "feature_work_order_sha256": context.get("feature_work_order_sha256", ""),
        "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    identity = _digest(spec)
    root = Path(context["project_root"]).resolve()
    checkpoint = root / ".mmm" / "construction_direct_methods" / (identity + ".json")
    with _claim_checkpoint(checkpoint):
        saved = _read_checkpoint(checkpoint, identity)
        if saved:
            if set(saved) != {member_task.owner_symbol}:
                raise ValueError("CONSTRUCTION_METHOD_CHECKPOINT_FOREIGN_DECLARATION")
            return saved[member_task.owner_symbol]
        index = context["evidence_index"]
        snippets = _evidence(index, member_task.intent, context.get("source_pages", {}))
        payload = {
            "implementation_task": member_task.intent,
            "required_effects": member_task.effects,
            "host_declaration": {
                "symbol_ref": member_task.owner_symbol,
                "signature": frame["prefix"],
            },
            "target": context["target"],
            "bounded_inspected_apis": snippets["target_api"],
            "installed_source": snippets["source"],
            "feature_work_order": context.get("feature_work_order"),
            "prerequisite_sha256": context["prerequisite_hashes"],
            "previous_sibling_sources": [
                row.get("code", "")[:4000] for ref, units in
                list(context.get("source_pages", {}).items())[-2:]
                for row in units if row.get("kind") == "accepted_java_member"
            ][:2],
        }
        messages = [
            {"role": "system", "content": (
                "Implement exactly ONE complete Java method for the real target. "
                "The host owns the class, signature, target version and import scope. "
                "Output the WHOLE fixed-frame method, no class/package or extra declarations. "
                "Implement the required gameplay or registration effect. Use ONLY actual "
                "inspected compatible target APIs and materialized prerequisite symbols. "
                "Do not invent a magic method, stub, TODO, empty action, null success "
                "or unrelated registration. Do not search all API symbols. "
                "If a specific value is needed, construct it using verified target "
                "types and real inputs, not a guessed provider.")
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]
        # One authored method; at most one tightly scoped syntax correction.
        for attempt in range(2):
            candidate = router.generate_java_source(
                "coder", messages,
                data_name="construction_unit_" + member_task.task_id[:64],
                source_frame=frame,
            )
            try:
                if (not isinstance(candidate, str)
                        or not candidate.startswith(frame["prefix"])
                        or not candidate.endswith(frame["suffix"])):
                    raise ValueError("CONSTRUCTION_UNIT_SIGNATURE_INVALID")
                body = candidate[len(frame["prefix"]):-len(frame["suffix"])].strip()
                if (not body or body in {"return;", ";"}
                        or re.search(r"\b(?:TODO|FIXME|UnsupportedOperationException)\b", body)):
                    raise ValueError("CONSTRUCTION_UNIT_BODY_INCOMPLETE")
                source = render_declaration(
                    "method", member["name"], {**metadata, "source": candidate},
                    owner=owner_name,
                )
            except ValueError as error:
                if attempt:
                    raise
                messages.append({"role": "user", "content": json.dumps({
                    "repair_only_this_method": True,
                    "diagnostic": str(error)[:1000],
                    "previous_source": str(candidate)[:7000],
                    "required_prefix": frame["prefix"],
                }, ensure_ascii=False)})
                continue
            _save_checkpoint(checkpoint, identity, {member_task.owner_symbol: source})
            print("CONSTRUCTION DIRECT METHOD: " + json.dumps({
                "owner_symbol": member_task.owner_symbol,
                "model_attempts": attempt + 1, "source_bytes": len(source.encode()),
                "checkpointed": True, "target_compiled": False,
            }, ensure_ascii=False), flush=True)
            return source
        raise ValueError("CONSTRUCTION_UNIT_NOT_AUTHORED")
