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
    verified_calls: Sequence[Mapping], targets: Sequence[str],
    statements: list[str], semantic_statements: list[str],
    effect_receipt: bool, receipt_name: str,
) -> None:
    """Write exactly one distinct inspected effect for each native state target.

    The model selects semantic identity from a bounded, verified API menu. It
    never authors Java receiver, method name, arguments, or arbitrary source.
    No match => explicit non-success, not a synthetic stored value. This
    compiler does not pretend that type-checking certifies gameplay behavior.
    """
    from .host_callback_body import (
        _ask, _effect_operation, lower_operation_from_host_verified_call,
    )

    # Exclude Java Object and collection queries which may be non-getter
    # methods but still have no externally observable mutation.
    known_pure = {
        "equals", "hashCode", "toString", "compareTo", "contains",
        "containsKey", "containsValue", "size", "length", "iterator",
        "stream", "spliterator", "clone", "copy", "matches",
        "startsWith", "endsWith", "indexOf", "lastIndexOf",
    }
    available = {
        str(row["id"]): row
        for row in verified_calls
        if isinstance(row, Mapping)
        and row.get("id") and row.get("expression") and row.get("signature")
        and "#" in str(row["signature"])
        and str(row["signature"]).split("#", 1)[1].split("(", 1)[0] not in known_pure
        and not str(row["signature"]).startswith(("java.lang.Object#", "java.lang.String#"))
        and _effect_operation("call", str(row["expression"]) + ";")
    }
    if not available:
        raise ValueError(
            "CALLBACK_RUNTIME_MUTATION_VERIFIED_API_UNAVAILABLE: "
            + ", ".join(targets))
    used: set[str] = set()
    for name in targets:
        options = {
            identifier: item
            for identifier, item in available.items()
            if item["expression"] not in used
        }
        if not options:
            raise ValueError("CALLBACK_RUNTIME_MUTATION_DISTINCT_API_REQUIRED: " + name)

        def lower(row):
            chosen = row["verified_call_id"]
            if chosen == "unavailable":
                raise ValueError("CALLBACK_RUNTIME_MUTATION_TARGET_UNRESOLVED: " + name)
            expression = str(options[chosen]["expression"])
            if expression in used:
                raise ValueError("CALLBACK_RUNTIME_MUTATION_CALL_REUSED: " + name)
            code = lower_operation_from_host_verified_call(
                expression, frame, statements, [])
            return code

        choice = _ask(
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
                    "Select a REAL checked Minecraft runtime mutation for the "
                    "EXACT state named below. The call must mutate its original "
                    "world/player/entity/item owner, not log, register a new "
                    "listener, query a getter, or update an unrelated value. "
                    "If no offered method implements this effect, select "
                    "'unavailable'.  Compiler validity alone does not prove "
                    "semantic equivalence. Never select another state as a proxy."
                ),
                "required_native_state": name,
                "behavior": frame.get("host_behavior_contract"),
                "inspected_calls": [
                    {"verified_call_id": identifier,
                     "signature": entry["signature"],
                     "expression": entry["expression"]}
                    for identifier, entry in options.items()
                ],
                "previous_native_mutations": sorted(used),
            }, lower=lower,
        )
        statements.append(choice)
        semantic_statements.append(choice)
        used.add(choice[:-1])
        if effect_receipt:
            statements.append(receipt_name + " = true;")
