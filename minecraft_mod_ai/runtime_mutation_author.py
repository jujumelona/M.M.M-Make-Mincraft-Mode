"""Host-owned runtime mutation admission.

An inspected callable is only evidence that an invocation exists, not that
a Minecraft mechanic works.  Each planned runtime_state_write must select a
concrete, bytecode-indexed invocation on a live callback receiver, and that
invocation must appear in the effect body.  Runtime playtests remain required.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence


def mutation_targets(behavior: Mapping | None) -> tuple[str, ...]:
    if not isinstance(behavior, Mapping):
        return ()
    raw = behavior.get("runtime_state_writes") or ()
    if not isinstance(raw, (tuple, list)):
        raise ValueError("CALLBACK_RUNTIME_MUTATIONS_INVALID")
    targets = []
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("CALLBACK_RUNTIME_MUTATION_NAME_INVALID")
        if value not in targets:
            targets.append(value)
    return tuple(targets)


def author_runtime_mutation_calls(
    router, scoped, *, data_name: str, frame: Mapping,
    verified_calls: Sequence[Mapping], typed_candidates: Sequence[Mapping],
    targets: Sequence[str], statements: list[str],
    semantic_statements: list[str], effect_receipt: bool, receipt_name: str,
) -> None:
    """Lower one distinct checked native effect per runtime-state obligation.

    Directly bound calls use the inspected callback ABI. Methods requiring a
    scalar argument use that same checked receiver and signature, but obtain
    each missing scalar from a separate bounded typed-value response. No
    model-authored method name, import, receiver, overload or source wrapper.
    """
    from .host_callback_body import (
        _ask, _effect_operation, lower_operation_from_host_verified_call,
        parse_expression,
    )

    known_pure = {
        "equals", "hashCode", "toString", "compareTo", "contains",
        "containsKey", "containsValue", "size", "length", "iterator",
        "stream", "spliterator", "clone", "copy", "matches",
        "startsWith", "endsWith", "indexOf", "lastIndexOf",
    }
    bound = {
        str(row["id"]): {"kind": "bound", **row}
        for row in verified_calls
        if isinstance(row, Mapping)
        and row.get("id") and row.get("expression") and row.get("signature")
        and "#" in str(row["signature"])
        and str(row["signature"]).split("#", 1)[1].split("(", 1)[0] not in known_pure
        and not str(row["signature"]).startswith(("java.lang.Object#", "java.lang.String#"))
        and _effect_operation("call", str(row["expression"]) + ";")
    }
    # Bound candidates above already cover methods whose every parameter is
    # a callback input. Include typed candidates only when at least one
    # argument truly needs an independently validated scalar value.
    typed = {
        str(row["id"]): {"kind": "typed", **row}
        for row in typed_candidates
        if isinstance(row, Mapping)
        and row.get("id") and row.get("signature")
        and any(arg.get("expression") is None for arg in row.get("arguments", ()))
    }
    available = {**bound, **typed}
    if not available:
        raise ValueError(
            "CALLBACK_RUNTIME_MUTATION_VERIFIED_API_UNAVAILABLE: "
            + ", ".join(targets))

    used_signatures: set[str] = set()
    for index, name in enumerate(targets):
        options = {
            identifier: row for identifier, row in available.items()
            if row["signature"] not in used_signatures
        }
        if not options:
            raise ValueError("CALLBACK_RUNTIME_MUTATION_DISTINCT_API_REQUIRED: " + name)

        def lower(row):
            choice = row["verified_call_id"]
            if choice == "unavailable":
                raise ValueError("CALLBACK_RUNTIME_MUTATION_TARGET_UNRESOLVED: " + name)
            selected = options[choice]
            if selected["kind"] == "bound":
                expression = str(selected["expression"])
            else:
                args = []
                for offset, argument in enumerate(selected["arguments"]):
                    resolved = argument.get("expression")
                    if resolved is None:
                        java_type = argument["type"]
                        resolved = _ask(
                            router, scoped,
                            name=(data_name + "_native_value_" +
                                  hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
                                  + "_" + str(offset)),
                            schema={
                                "type": "object",
                                "properties": {"expression": {
                                    "type": "string", "minLength": 1, "maxLength": 2048}},
                                "required": ["expression"],
                                "additionalProperties": False,
                            },
                            context={
                                "instruction": (
                                    "Supply ONLY a Java value expression of the exact "
                                    "declared type for the checked native mutator. "
                                    "Use verified callback/world inputs. Do not "
                                    "author a method call statement, class, import "
                                    "or different target state. The host owns "
                                    "the receiver and method overload."
                                ),
                                "required_native_state": name,
                                "verified_method": selected["signature"],
                                "argument_index": offset,
                                "required_java_type": java_type,
                                "assigned_behavior": frame.get("host_behavior_contract"),
                            },
                            lower=lambda value: parse_expression(value["expression"]),
                        )
                    args.append(resolved)
                expression = (
                    selected["receiver"] + "." + selected["name"] +
                    "(" + ", ".join(args) + ")"
                )
            code = lower_operation_from_host_verified_call(
                expression, frame, statements, [])
            return {"statement": code, "signature": selected["signature"]}

        selected_result = _ask(
            router, scoped,
            name=(data_name + "_runtime_mutation_" +
                  hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]),
            schema={
                "type": "object",
                "properties": {"verified_call_id": {
                    "type": "string",
                    "enum": [*options, "unavailable"],
                }},
                "required": ["verified_call_id"],
                "additionalProperties": False,
            },
            context={
                "instruction": (
                    "Select the real inspected mutator for the EXACT original "
                    "Minecraft state named below. Select 'unavailable' if "
                    "none changes its original world/player/entity/item owner. "
                    "A log, getter, callback registration, unrelated setter "
                    "or detached scalar cannot satisfy the obligation. "
                    "Actual semantic success still requires gameplay tests."
                ),
                "required_native_state": name,
                "behavior": frame.get("host_behavior_contract"),
                "inspected_calls": [
                    {"verified_call_id": identifier, "signature": item["signature"],
                     "expression": item.get("expression"),
                     "receiver": item.get("receiver"),
                     "typed_arguments": item.get("arguments")}
                    for identifier, item in options.items()
                ],
                "prior_verified_signatures": sorted(used_signatures),
            }, lower=lower,
        )
        # Signature is carried directly from the selected checked ID, not
        # rediscovered with a text-prefix search (which conflates overloads).
        used_signatures.add(selected_result["signature"])
        statement = selected_result["statement"]
        statements.append(statement)
        semantic_statements.append(statement)
        if effect_receipt:
            statements.append(receipt_name + " = true;")
        collector = frame.get("host_runtime_mutation_receipts")
        if collector is not None:
            collector.append({
                "behavior_id": str(
                    (frame.get("host_behavior_contract") or {}).get("behavior_id") or ""),
                "state_name": name,
                "verified_signature": selected_result["signature"],
                "statement": statement,
            })



def verify_native_runtime_mutation_coverage(source, behaviors, receipts):
    """Recheck immutable native effect identity AFTER compiler repair and replay.

    The first-pass generator selected each call from inspected Java symbols,
    but compiler repair can replace a whole member. Compilation alone must
    not certify a repaired handler after it dropped a mandatory mutation.
    This proves call preservation/owner placement, not in-game semantics.
    """
    import re

    from .java_region_parser import _parse, _text

    required = {
        (str(row.get("behavior_id") or ""), str(name))
        for row in behaviors
        for name in (row.get("runtime_state_writes") or ())
    }
    if not required:
        return
    if not isinstance(receipts, (list, tuple)):
        raise ValueError("NATIVE_MUTATION_PROOF_RECEIPTS_REQUIRED")
    by_key = {}
    for item in receipts:
        if not isinstance(item, dict):
            raise ValueError("NATIVE_MUTATION_PROOF_RECEIPT_INVALID")
        key = (str(item.get("behavior_id") or ""), str(item.get("state_name") or ""))
        if key in by_key or key not in required:
            raise ValueError("NATIVE_MUTATION_PROOF_OWNER_INVALID: " + repr(key))
        signature = str(item.get("verified_signature") or "")
        statement = str(item.get("statement") or "")
        if "#" not in signature or not statement.endswith(";"):
            raise ValueError("NATIVE_MUTATION_PROOF_ABI_INVALID: " + repr(key))
        by_key[key] = item
    if set(by_key) != required:
        raise ValueError("NATIVE_MUTATION_PROOF_COVERAGE_MISSING: " +
                         repr(sorted(required - set(by_key))))
    if len({item["verified_signature"] for item in by_key.values()}) != len(required):
        raise ValueError("NATIVE_MUTATION_PROOF_SIGNATURE_REUSED")

    raw, tree = _parse(source)
    methods = {}
    pending = [tree]
    while pending:
        node = pending.pop()
        if node.type == "method_declaration":
            name = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name is not None and body is not None:
                methods[_text(raw, name)] = body
        pending.extend(node.named_children)
    for (behavior_id, state_name), item in by_key.items():
        handler = "applyBehavior" + hashlib.sha256(
            behavior_id.encode("utf-8")).hexdigest()[:20]
        method = methods.get(handler)
        if method is None:
            raise ValueError("NATIVE_MUTATION_PROOF_HANDLER_MISSING: " + behavior_id)
        expected = re.sub(r"\s+", "", item["statement"].rstrip(";"))
        found = False
        pending = [method]
        while pending:
            node = pending.pop()
            if node.type == "method_invocation":
                candidate = re.sub(r"\s+", "", _text(raw, node))
                if candidate == expected:
                    found = True
                    break
            pending.extend(node.named_children)
        if not found:
            raise ValueError(
                "NATIVE_MUTATION_PROOF_EFFECT_DROPPED: " +
                behavior_id + "/" + state_name)
