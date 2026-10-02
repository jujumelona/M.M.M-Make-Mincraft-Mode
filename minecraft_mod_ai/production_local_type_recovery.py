"""Production-side recovery for small-model Java type authority failures."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .atomic_region_correction import RegionCorrection
from .custom_module_errors import CustomModuleGenerationError
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


def _requirement_text(authority: Mapping[str, Any]) -> str:
    sources = authority.get("source_requirements")
    if not isinstance(sources, Mapping):
        return ""
    return "\n".join(str(value or "") for value in sources.values())


def _contract_type_values(contract: Mapping[str, Any]) -> tuple[Any, ...]:
    kind = contract.get("kind")
    if kind == "field":
        return (contract.get("declared_type"),)
    if kind == "method":
        params = tuple(
            item.get("type")
            for item in contract.get("parameters") or ()
            if isinstance(item, Mapping)
        )
        return (contract.get("return_type"), *params)
    if kind == "constructor":
        return tuple(
            item.get("type")
            for item in contract.get("parameters") or ()
            if isinstance(item, Mapping)
        )
    return ()


def _private_declaration_uses_type(
    contracts: Sequence[Mapping[str, Any]],
    name: str,
    type_leaf_names: Callable[[Any], tuple[str, ...]],
) -> bool:
    uses = [
        contract
        for contract in contracts
        if any(name in type_leaf_names(value) for value in _contract_type_values(contract))
    ]
    return bool(uses) and all(
        str(contract.get("visibility") or "").strip() == "private"
        for contract in uses
    )


def _direct_zero_arg_only(
    creations: Sequence[Mapping[str, Any]],
    name: str,
    type_leaf_names: Callable[[Any], tuple[str, ...]],
) -> bool:
    matches = [
        row for row in creations
        if name in type_leaf_names(row.get("type"))
    ]
    return bool(matches) and all(int(row.get("argument_count") or 0) == 0 for row in matches)


def _eligible_local_carrier(
    name: str,
    *,
    requirement_text: str,
    source: str,
    contracts: Sequence[Mapping[str, Any]],
    creations: Sequence[Mapping[str, Any]],
    type_leaf_names: Callable[[Any], tuple[str, ...]],
    structure_scan: Callable[[str], str],
) -> bool:
    named = (
        "." not in name
        and re.fullmatch(r"[A-Z][A-Za-z0-9_$]*", name) is not None
        and re.search(rf"\b{re.escape(name)}\b", requirement_text) is not None
    )
    private_use = _private_declaration_uses_type(contracts, name, type_leaf_names)
    zero_arg = _direct_zero_arg_only(creations, name, type_leaf_names)
    static_receiver = re.search(
        rf"\b{re.escape(name)}\s*\.", structure_scan(source)
    ) is not None
    return named and private_use and zero_arg and not static_receiver


def canonicalize_plan_local_zero_arg_domain_types(
    value: str,
    *,
    concern_authority: Mapping[str, Any],
    dependency_source: str,
    sibling_api: Sequence[Mapping[str, Any]],
    validate_declared_type_authority: Callable[..., None],
    type_leaf_names: Callable[[Any], tuple[str, ...]],
    structure_scan: Callable[[str], str],
) -> tuple[str, tuple[str, ...]]:
    source = str(value or "").strip()
    if not source:
        return source, ()
    try:
        contracts = class_body_member_contracts(source)
    except JavaRegionParseError:
        return source, ()
    try:
        validate_declared_type_authority(
            contracts,
            dependency_source=dependency_source,
            sibling_api=sibling_api,
        )
        return source, ()
    except CustomModuleGenerationError as exc:
        unknown = type_authority_unknown_names(str(exc))

    requirement_text = _requirement_text(concern_authority)
    creations = tuple(class_body_object_creations(source))
    eligible = [
        name
        for name in unknown
        if _eligible_local_carrier(
            name,
            requirement_text=requirement_text,
            source=source,
            contracts=contracts,
            creations=creations,
            type_leaf_names=type_leaf_names,
            structure_scan=structure_scan,
        )
    ]
    additions = [f"private static final class {name} {{}}" for name in eligible]
    repaired = source.rstrip() + (("\n\n" + "\n\n".join(additions)) if additions else "")
    changes = tuple(f"{name}:materialized_private_zero_arg_domain_type" for name in eligible)
    return repaired, changes


def prepare_and_validate_plan_local_types(
    value: str,
    *,
    concern: str,
    concern_authority: Mapping[str, Any],
    dependency_source: str,
    sibling_api: Sequence[Mapping[str, Any]],
    validate_declared_type_authority: Callable[..., None],
    validate_first_pass: Callable[..., None],
    type_leaf_names: Callable[[Any], tuple[str, ...]],
    structure_scan: Callable[[str], str],
) -> str:
    repaired, changes = canonicalize_plan_local_zero_arg_domain_types(
        value,
        concern_authority=concern_authority,
        dependency_source=dependency_source,
        sibling_api=sibling_api,
        validate_declared_type_authority=validate_declared_type_authority,
        type_leaf_names=type_leaf_names,
        structure_scan=structure_scan,
    )
    if changes:
        from .root_cause_trace import emit_root_cause

        emit_root_cause(
            "atomic_concern_plan_local_type_materialized",
            stage="production",
            operation="atomic_concern_region",
            gate="first_pass_semantic_canonicalization",
            result="PASS",
            details={"concern": concern, "changes": list(changes)},
        )
    validate_first_pass(
        repaired,
        dependency_source=dependency_source,
        sibling_api=sibling_api,
    )
    return repaired
