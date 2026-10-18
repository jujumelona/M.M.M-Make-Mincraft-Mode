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



def _verify_direct_member(source: str, member: dict, context: dict) -> bool:
    """Prove a complete direct member against the selected target classpath.

    A syntactically valid Java method can reference nonexistent classes, the
    wrong Minecraft mappings, or client-only types. Do not commit its checkpoint
    before javac has resolved the generated owner compilation unit.
    """
    classpath = context.get("java_classpath")
    if not classpath:
        # A syntactically valid member with unknown imports or API owners is
        # NOT an accepted artifact. Do not persist unverifiable source from
        # an alternate caller that bypassed construction preflight.
        raise ValueError("CONSTRUCTION_DIRECT_TARGET_CLASSPATH_REQUIRED")
    from .javac_bridge import analyze_java
    owner = str(member["owner"])
    package, sep, class_name = owner.rpartition(".")
    if not sep or not package:
        raise ValueError("CONSTRUCTION_DIRECT_OWNER_INVALID")
    superclass = str(member.get("class_superclass") or "").strip()
    interfaces = tuple(member.get("class_interfaces") or ())
    header = "public class " + class_name
    if superclass:
        header += " extends " + superclass
    if interfaces:
        header += " implements " + ", ".join(interfaces)
    unit = "package " + package + ";\n" + header + " {\n" + source + "\n}\n"
    try:
        analyze_java(
            unit, classpath=classpath,
            java_version=str(context["target"]["java_version"]),
            filename=class_name + ".java",
        )
    except (ValueError, OSError, RuntimeError) as exc:
        # The complete diagnostic is passed to the *same* method correction
        # request; it is not converted into an alleged valid API match.
        raise ValueError("CONSTRUCTION_DIRECT_TARGET_SYMBOL_INVALID: " + str(exc)[:2400]) from exc
    return True


def eligible_method(member_task, symbols, context, *, owned_records=(), port_source_ref="") -> bool:
    member = symbols.get(member_task.owner_symbol, {})
    if member.get("kind") not in {"METHOD", "FIELD"}:
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
            or (member.get("kind") == "METHOD" and
                not isinstance(member.get("parameters"), (list, tuple)))):
        return False
    return True


def compose_method(router, member_task, context, *, owner_name: str) -> str:
    member = context["symbols"][member_task.owner_symbol]
    metadata = {
        "visibility": member.get("visibility", "private"),
        "static": bool(member.get("static", False)),
        "final": bool(member.get("final", False)),
        "return_type": member["return_type"],
        "parameters": list(member.get("parameters", ())),
        "throws": list(member.get("throws", ())),
        "annotations": list(member.get("annotations", ())),
        "type_parameters": member.get("type_parameters", ""),
    }
    kind = "field" if member["kind"] == "FIELD" else "method"
    if kind == "field":
        metadata["type"] = member["return_type"]
    frame = declaration_frame(kind, member["name"], metadata,
                              allow_uninitialized=False)
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
            _verify_direct_member(saved[member_task.owner_symbol], member, context)
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
            "typed_input_producers": snippets.get("typed_input_producers", []),
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
                "Implement exactly ONE complete Java method or initialized field "
                "for the real target. The host owns the class, declaration type, "
                "signature, target version and import scope. "
                "Output the WHOLE fixed-frame declaration, no class/package or extras. "
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
            candidate = None
            try:
                candidate = router.generate_java_source(
                    "coder", messages,
                    data_name="construction_unit_" + member_task.task_id[:64],
                    source_frame=frame,
                )
                if (not isinstance(candidate, str)
                        or not candidate.startswith(frame["prefix"])
                        or not candidate.endswith(frame["suffix"])):
                    raise ValueError("CONSTRUCTION_UNIT_SIGNATURE_INVALID")
                body = candidate[len(frame["prefix"]):-len(frame["suffix"])].strip()
                if (not body or body in {"return;", ";"}
                        or re.search(r"\b(?:TODO|FIXME|UnsupportedOperationException)\b", body)):
                    raise ValueError("CONSTRUCTION_UNIT_BODY_INCOMPLETE")
                source = render_declaration(
                    kind, member["name"], {**metadata, "source": candidate},
                    owner=owner_name,
                )
                target_symbols_verified = _verify_direct_member(source, member, context)
            except ValueError as error:
                from .construction_source_repair import preserve_failed_java, java_correction_message
                candidate = preserve_failed_java(error, candidate, kind=kind, name=member["name"],
                                                 owner=owner_name, phase="direct_author")
                if attempt:
                    raise
                messages.append(java_correction_message(error, candidate, frame))
                continue
            _save_checkpoint(checkpoint, identity, {member_task.owner_symbol: source})
            print("CONSTRUCTION DIRECT MEMBER: " + json.dumps({
                "owner_symbol": member_task.owner_symbol,
                "model_attempts": attempt + 1, "source_bytes": len(source.encode()),
                "checkpointed": True, "target_symbols_verified": target_symbols_verified,
                "target_compiled": False,
            }, ensure_ascii=False), flush=True)
            return source
        raise ValueError("CONSTRUCTION_UNIT_NOT_AUTHORED")
