"""Production-side recovery for small-model Java type authority failures."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .atomic_region_correction import RegionCorrection
_TYPE_FAILURE_PREFIX = (
    "ATOMIC_CONCERN_RESPONSE_INVALID: ungrounded simple Java type name(s): "
)
_TYPE_FAILURE_SUFFIX = ". Use an authoritative sibling/dependency type"


def type_authority_unknown_names(diagnostic: str) -> tuple[str, ...]:
    text = str(diagnostic or "").strip()
    if not text.startswith(_TYPE_FAILURE_PREFIX):
        return ()
    body = text[len(_TYPE_FAILURE_PREFIX):]
    body = body.split(_TYPE_FAILURE_SUFFIX, 1)[0]
    return tuple(item.strip() for item in body.split(",") if item.strip())


def type_authority_repair_contract(diagnostic: str) -> dict[str, Any] | None:
    names = [
        name
        for name in type_authority_unknown_names(diagnostic)
        if "." not in name and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name)
    ]
    if not names:
        return None
    return {
        "unknown_simple_types": names,
        "rules": (
            "Resolve every unknown_simple_type before returning Java. Prefer an exact "
            "available_sibling_api or dependency_api type when one owns the concept. "
            "If the authored source requirement itself names a concern-local runtime "
            "domain object and no external authority owns it, declare the smallest "
            "private static nested class/record with that exact name in this selected "
            "concern region, then use it consistently. Never leave an undeclared simple "
            "type, never invent an external package, and never add a nested type for a "
            "misspelled JDK/dependency/platform symbol."
        ),
    }


def dependency_repair_contract(raw: str, diagnostic: str) -> dict[str, Any] | None:
    """Project a dependency diagnostic without bloating the atomic executor."""
    match = re.search(
        r"dependency API\s+([A-Za-z_$][A-Za-z0-9_$.]*)\."
        r"([A-Za-z_$][A-Za-z0-9_$]*)\s+called with\s+(\d+)\s+argument",
        str(diagnostic or ""),
    )
    if match is None:
        return None
    owner, method, arity_text = match.groups()
    from .atomic_concern_source import _dependency_call_contracts

    owner_rows = [
        row for row in _dependency_call_contracts(raw)
        if row.get("owner") == owner
    ]
    if not owner_rows:
        return None
    return {
        "owner": owner,
        "rejected_method": method,
        "rejected_arity": int(arity_text),
        "authoritative_calls": owner_rows,
        "rules": (
            "Use only one authoritative call listed here. parameter_names are semantic "
            "roles, not decoration. A trigger_dispatch/event_dispatch call consumes a "
            "trigger/event plus context and must never be used as a key/value setter. "
            "For a direct state assignment choose a direct_state_write signature. "
            "Do not add, remove, reorder, or invent dependency arguments outside an "
            "authoritative invocation_shape."
        ),
    }


def decorate_type_authority_retry(
    messages: list[dict[str, str]],
    payload: dict[str, Any],
    repair: Mapping[str, Any] | None,
) -> None:
    if repair is None:
        return
    payload["type_authority_repair_contract"] = dict(repair)
    scope = payload.get("scope")
    if isinstance(scope, dict):
        scope["repair_structure_rule"] = (
            "This is a pre-compilation type-authority correction. Regenerate the "
            "selected concern and add the smallest private concern-local nested "
            "class/record named by type_authority_repair_contract when the authored "
            "requirement owns that runtime object. Do not add unrelated nested types."
        )
    messages[0]["content"] += (
        "\nTYPE AUTHORITY CORRECTION TURN: resolve every listed unknown simple type. "
        "A requirement-owned local runtime domain type may be declared as a private "
        "nested class/record in this selected region; do not fabricate an external package."
    )


def region_correction_for_rejection(
    source: str,
    diagnostic: str,
    *,
    allow_private_restructure: bool,
    type_authority_repair: Mapping[str, Any] | None,
) -> RegionCorrection | None:
    # A type-authority failure can require adding a declaration. RegionCorrection
    # intentionally freezes declaration shape, so use the concern boundary instead.
    if allow_private_restructure and type_authority_repair is not None:
        return None
    return RegionCorrection.for_diagnostic(
        source,
        diagnostic,
        allow_private_restructure=allow_private_restructure,
    )


def type_authority_validation_rule(
    repair: Mapping[str, Any] | None,
    *,
    repair_mode: bool,
    response_region: str,
) -> str:
    if repair is not None and not repair_mode and response_region == "members":
        return (
            " This failure is an ungrounded simple-type error. For an exact type "
            "listed by type_authority_repair_contract that is owned by this authored "
            "concern rather than a sibling/dependency/JDK/platform API, regenerate "
            "the concern with the smallest private nested class/record needed to make "
            "that runtime type real. Do not add unrelated nested types."
        )
    return (
        " Do not introduce, rename, or change the kind of nested types during "
        "bounded regeneration."
    )
