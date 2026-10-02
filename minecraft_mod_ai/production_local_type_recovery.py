"""Production-side recovery for small-model Java type authority failures."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .atomic_region_correction import RegionCorrection
from .java_region_parser import (
    JavaRegionParseError,
    class_body_member_contracts,
    class_body_object_creations,
)

_TYPE_FAILURE_PREFIX = (
    "ATOMIC_CONCERN_RESPONSE_INVALID: ungrounded simple Java type name(s): "
)
_TYPE_FAILURE_SUFFIX = ". Use an authoritative sibling/dependency type"
_COPY_EDIT_RULE = (
    "This is a copy-edit task, not a design task. Copy "
    "current_selected_region_source, insert every exact_declaration exactly once "
    "as a sibling member, preserve every listed existing declaration, and return "
    "the whole repaired concern. Do not rename methods, invent packages, or "
    "reinterpret the requirement."
)
_TYPE_RESOLUTION_RULE = (
    "Resolve every unknown_simple_type before returning Java. Prefer an exact "
    "available_sibling_api or dependency_api type when one owns the concept. "
    "If the authored source requirement itself names a concern-local runtime "
    "domain object and no external authority owns it, declare the smallest "
    "private static nested class/record with that exact name in this selected "
    "concern region, then use it consistently. Never leave an undeclared simple "
    "type, never invent an external package, and never add a nested type for a "
    "misspelled JDK/dependency/platform symbol."
)
_DEFAULT_VALIDATION_RULE = (
    " Do not introduce, rename, or change the kind of nested types during "
    "bounded regeneration."
)
_MECHANICAL_VALIDATION_RULE = (
    " This is a mechanical copy-edit. Insert every exact_declaration from "
    "type_authority_repair_contract.mechanical_edits exactly once, preserve "
    "the supplied existing declarations, and return the whole concern."
)
_BOUNDED_VALIDATION_RULE = (
    " This failure is an ungrounded simple-type error. For an exact type "
    "listed by type_authority_repair_contract that is owned by this authored "
    "concern rather than a sibling/dependency/JDK/platform API, regenerate "
    "the concern with the smallest private nested class/record needed to make "
    "that runtime type real. Do not add unrelated nested types."
)
_SCOPE_RULES = {
    True: (
        "Mechanical copy-edit only: copy current_selected_region_source, insert "
        "each mechanical_edits[].exact_declaration exactly once as a sibling "
        "member, preserve listed declarations, and return the whole concern."
    ),
    False: (
        "This is a pre-compilation type-authority correction. Regenerate the "
        "selected concern and add the smallest private concern-local nested "
        "class/record named by type_authority_repair_contract when the authored "
        "requirement owns that runtime object. Do not add unrelated nested types."
    ),
}
_SYSTEM_SUFFIXES = {
    True: (
        "\nMECHANICAL TYPE REPAIR TURN: do not redesign anything. Copy the supplied "
        "current_selected_region_source, insert the exact Java declaration(s) listed "
        "in type_authority_repair_contract.mechanical_edits exactly once, preserve "
        "the existing declarations, and output the complete repaired Java region only."
    ),
    False: (
        "\nTYPE AUTHORITY CORRECTION TURN: resolve every listed unknown simple type. "
        "A requirement-owned local runtime domain type may be declared as a private "
        "nested class/record in this selected region; do not fabricate an external package."
    ),
}


def type_authority_unknown_names(diagnostic: str) -> tuple[str, ...]:
    text = str(diagnostic or "").strip()
    if not text.startswith(_TYPE_FAILURE_PREFIX):
        return ()
    body = text[len(_TYPE_FAILURE_PREFIX):]
    body = body.split(_TYPE_FAILURE_SUFFIX, 1)[0]
    return tuple(item.strip() for item in body.split(",") if item.strip())


def _simple_unknown_names(diagnostic: str) -> tuple[str, ...]:
    return tuple(
        name
        for name in type_authority_unknown_names(diagnostic)
        if "." not in name and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name)
    )


def _requirement_text(authority: Mapping[str, Any] | None) -> str:
    raw = authority.get("source_requirements") if isinstance(authority, Mapping) else {}
    values = raw.values() if isinstance(raw, Mapping) else ()
    return "\n".join(str(value or "") for value in values)


def _parse_candidate(
    source: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]] | None:
    try:
        return (
            class_body_member_contracts(source),
            class_body_object_creations(source),
        )
    except JavaRegionParseError:
        return None


def _requirement_owns_type(
    name: str,
    authority: Mapping[str, Any] | None,
) -> bool:
    return re.search(
        rf"\b{re.escape(name)}\b", _requirement_text(authority)
    ) is not None


def _declares_type(
    contracts: Sequence[Mapping[str, Any]],
    name: str,
) -> bool:
    return any(
        row.get("kind") == "type" and row.get("symbol") == name
        for row in contracts
    )


def _return_users(
    contracts: Sequence[Mapping[str, Any]],
    name: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        row
        for row in contracts
        if row.get("kind") == "method"
        and str(row.get("return_type") or "").strip() == name
    )


def _matching_creations(
    creations: Sequence[Mapping[str, Any]],
    name: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        row
        for row in creations
        if str(row.get("type") or "").strip() == name
    )


def _zero_arg_only(creations: Sequence[Mapping[str, Any]]) -> bool:
    return bool(creations) and all(
        int(row.get("argument_count") or 0) == 0
        for row in creations
    )


def _preserved_declarations(
    contracts: Sequence[Mapping[str, Any]],
) -> list[str]:
    return [
        declaration
        for row in contracts
        if row.get("kind") != "type"
        and (declaration := str(row.get("declaration") or "").strip())
    ]


def _mechanical_local_carrier_edit(
    name: str,
    rejected_source: str,
    concern_authority: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    source = str(rejected_source or "").strip()
    parsed = _parse_candidate(source)
    if not source or parsed is None or not _requirement_owns_type(name, concern_authority):
        return None
    contracts, creations = parsed
    if _declares_type(contracts, name):
        return None
    users = _return_users(contracts, name)
    created = _matching_creations(creations, name)
    if len(users) != 1 or not _zero_arg_only(created):
        return None
    return {
        "operation": "insert_exact_sibling_declaration",
        "type_name": name,
        "exact_declaration": f"private static final class {name} {{}}",
        "preserve_existing_declarations": _preserved_declarations(contracts),
        "edit_budget": "one exact sibling declaration insertion; no semantic rewrites",
    }


def _mechanical_edits(
    names: Sequence[str],
    rejected_source: str,
    concern_authority: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    return [
        edit
        for name in names
        if (
            edit := _mechanical_local_carrier_edit(
                name, rejected_source, concern_authority
            )
        ) is not None
    ]


def type_authority_repair_contract(
    diagnostic: str,
    rejected_source: str = "",
    concern_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    names = _simple_unknown_names(diagnostic)
    if not names:
        return None
    edits = _mechanical_edits(names, rejected_source, concern_authority)
    mechanical = len(edits) == len(names)
    mode = "mechanical_copy_edit" if mechanical else "bounded_type_resolution"
    return {
        "mode": mode,
        "unknown_simple_types": list(names),
        "mechanical_edits": edits,
        "rules": _COPY_EDIT_RULE if mechanical else _TYPE_RESOLUTION_RULE,
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
    mechanical = repair.get("mode") == "mechanical_copy_edit"
    scope = payload.get("scope")
    if isinstance(scope, dict):
        scope["repair_structure_rule"] = _SCOPE_RULES[mechanical]
    messages[0]["content"] += _SYSTEM_SUFFIXES[mechanical]


def region_correction_for_rejection(
    source: str,
    diagnostic: str,
    *,
    allow_private_restructure: bool,
    type_authority_repair: Mapping[str, Any] | None,
) -> RegionCorrection | None:
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
    active = repair is not None and not repair_mode and response_region == "members"
    if not active:
        return _DEFAULT_VALIDATION_RULE
    rules = {
        "mechanical_copy_edit": _MECHANICAL_VALIDATION_RULE,
        "bounded_type_resolution": _BOUNDED_VALIDATION_RULE,
    }
    return rules.get(str(repair.get("mode") or ""), _BOUNDED_VALIDATION_RULE)
