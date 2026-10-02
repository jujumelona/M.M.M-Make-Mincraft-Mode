"""Production-side recovery for small-model Java type authority failures."""
from __future__ import annotations

import re
from collections.abc import Mapping
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


def type_authority_unknown_names(diagnostic: str) -> tuple[str, ...]:
    text = str(diagnostic or "").strip()
    if not text.startswith(_TYPE_FAILURE_PREFIX):
        return ()
    body = text[len(_TYPE_FAILURE_PREFIX):]
    body = body.split(_TYPE_FAILURE_SUFFIX, 1)[0]
    return tuple(item.strip() for item in body.split(",") if item.strip())


def _requirement_text(authority: Mapping[str, Any] | None) -> str:
    if not isinstance(authority, Mapping):
        return ""
    raw = authority.get("source_requirements")
    if not isinstance(raw, Mapping):
        return ""
    return "\n".join(str(value or "") for value in raw.values())


def _mechanical_local_carrier_edit(
    name: str,
    *,
    rejected_source: str,
    concern_authority: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return an exact copy-edit when the missing type needs no design inference.

    The host does not apply this declaration. It proves that one exact declaration is
    sufficient for the currently rejected source shape, then tells the small coder to
    emit the complete repaired concern itself.
    """

    source = str(rejected_source or "").strip()
    if (
        not source
        or re.search(rf"\b{re.escape(name)}\b", _requirement_text(concern_authority))
        is None
    ):
        return None

    try:
        contracts = class_body_member_contracts(source)
        creations = class_body_object_creations(source)
    except JavaRegionParseError:
        return None

    # Do not turn a misspelling of an external/JDK API into a fake local type.
    if any(
        row.get("kind") == "type" and row.get("symbol") == name
        for row in contracts
    ):
        return None

    return_users = [
        row
        for row in contracts
        if row.get("kind") == "method"
        and str(row.get("return_type") or "").strip() == name
    ]
    matching_creations = [
        row
        for row in creations
        if str(row.get("type") or "").strip() == name
    ]
    if (
        len(return_users) != 1
        or not matching_creations
        or any(int(row.get("argument_count") or 0) != 0 for row in matching_creations)
    ):
        return None

    preserved = [
        str(row.get("declaration") or "").strip()
        for row in contracts
        if row.get("kind") != "type"
        and str(row.get("declaration") or "").strip()
    ]
    return {
        "operation": "insert_exact_sibling_declaration",
        "type_name": name,
        "exact_declaration": f"private static final class {name} {{}}",
        "preserve_existing_declarations": preserved,
        "edit_budget": "one exact sibling declaration insertion; no semantic rewrites",
    }


def type_authority_repair_contract(
    diagnostic: str,
    *,
    rejected_source: str = "",
    concern_authority: Mapping[str, Any] | None = None,
    retry_level: int = 1,
) -> dict[str, Any] | None:
    names = [
        name
        for name in type_authority_unknown_names(diagnostic)
        if "." not in name and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name)
    ]
    if not names:
        return None

    edits = [
        edit
        for name in names
        if (
            edit := _mechanical_local_carrier_edit(
                name,
                rejected_source=rejected_source,
                concern_authority=concern_authority,
            )
        ) is not None
    ]
    mechanical = len(edits) == len(names)
    return {
        "mode": (
            "mechanical_copy_edit"
            if mechanical
            else "bounded_type_resolution"
        ),
        "retry_level": max(1, int(retry_level)),
        "unknown_simple_types": names,
        "mechanical_edits": edits,
        "rules": (
            (
                "This is a copy-edit task, not a design task. Copy "
                "current_selected_region_source, insert every exact_declaration exactly "
                "once as a sibling member, preserve every listed existing declaration, "
                "and return the whole repaired concern. Do not rename methods, invent "
                "packages, or reinterpret the requirement."
            )
            if mechanical
            else (
                "Resolve every unknown_simple_type before returning Java. Prefer an exact "
                "available_sibling_api or dependency_api type when one owns the concept. "
                "If the authored source requirement itself names a concern-local runtime "
                "domain object and no external authority owns it, declare the smallest "
                "private static nested class/record with that exact name in this selected "
                "concern region, then use it consistently. Never leave an undeclared simple "
                "type, never invent an external package, and never add a nested type for a "
                "misspelled JDK/dependency/platform symbol."
            )
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
    mechanical = repair.get("mode") == "mechanical_copy_edit"
    if isinstance(scope, dict):
        scope["repair_structure_rule"] = (
            (
                "Mechanical copy-edit only: copy current_selected_region_source, insert "
                "each mechanical_edits[].exact_declaration exactly once as a sibling "
                "member, preserve listed declarations, and return the whole concern."
            )
            if mechanical
            else (
                "This is a pre-compilation type-authority correction. Regenerate the "
                "selected concern and add the smallest private concern-local nested "
                "class/record named by type_authority_repair_contract when the authored "
                "requirement owns that runtime object. Do not add unrelated nested types."
            )
        )
    messages[0]["content"] += (
        (
            "\nMECHANICAL TYPE REPAIR TURN: do not redesign anything. Copy the supplied "
            "current_selected_region_source, insert the exact Java declaration(s) listed "
            "in type_authority_repair_contract.mechanical_edits exactly once, preserve "
            "the existing declarations, and output the complete repaired Java region only."
        )
        if mechanical
        else (
            "\nTYPE AUTHORITY CORRECTION TURN: resolve every listed unknown simple type. "
            "A requirement-owned local runtime domain type may be declared as a private "
            "nested class/record in this selected region; do not fabricate an external package."
        )
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
        if repair.get("mode") == "mechanical_copy_edit":
            return (
                " This is a mechanical copy-edit. Insert every exact_declaration from "
                "type_authority_repair_contract.mechanical_edits exactly once, preserve "
                "the supplied existing declarations, and return the whole concern."
            )
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
