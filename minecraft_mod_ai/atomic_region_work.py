"""Bind a production page to a declaration, not just a free-form suggestion.

Source and compiler admission remain in the concern executor. This module owns
only scheduling identity, immutable progress receipts and unfinished parent work.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from copy import deepcopy

from .custom_module_errors import CustomModuleGenerationError
from .java_region_parser import (
    JavaRegionParseError,
    class_body_member_contracts,
    class_body_token_identity,
    strict_member_chunks,
)


def _type(value: str) -> str:
    from .atomic_concern_source import _canonicalize_jdk_type_expression

    raw = re.sub(r"\s*(?:\[\s*\]|\.\.\.)\s*", "[]", value.strip())
    suffix = ""
    while raw.endswith("[]"):
        raw, suffix = raw[:-2], suffix + "[]"
    return re.sub(r"\s+", "", _canonicalize_jdk_type_expression(raw)) + suffix


def _key(row: Mapping) -> tuple:
    from .atomic_concern_source import _erase_generic_arguments

    return (
        row.get("kind"), row.get("symbol"),
        tuple(_erase_generic_arguments(_type(p["type"])) for p in row.get("parameters", ())),
    )


def _contract(row: Mapping) -> dict:
    result = {k: v for k, v in row.items() if k not in {"declaration", "initialized"}}
    for name in ("declared_type", "return_type"):
        if name in result:
            result[name] = _type(result[name])
    if "parameters" in result:
        # Parameter names are local implementation details, types are not.
        result["parameters"] = [_type(p["type"]) for p in result["parameters"]]
    if row.get("kind") == "type":
        # Includes record components, extends/implements and generic bounds.
        result["header"] = class_body_token_identity(row["declaration"].rsplit("{", 1)[0] + " {}")
    return result


def target_contract(target: str) -> dict:
    """Parse one header with the same Java grammar used to admit the output."""
    if not target or any(token in target for token in ("{", "}", ";", "=", "//", "/*")):
        raise CustomModuleGenerationError(
            "ATOMIC_REGION_TARGET_INVALID: target must be one Java declaration header "
            "without body, initializer, comment or semicolon"
        )
    for suffix in (";", " {}"):
        try:
            source = target + suffix
            chunks = strict_member_chunks(source)
            rows = class_body_member_contracts(source)
            if len(chunks) == len(rows) == 1 and rows[0]["kind"] in {"field", "method", "type"}:
                return rows[0]
        except JavaRegionParseError:
            continue
    raise CustomModuleGenerationError("ATOMIC_REGION_TARGET_INVALID: expected one field, method or nested type header")


def validate_schedule(decision: dict, context: Mapping) -> dict:
    """Run inside native decision feedback, before invoking the source coder."""
    if "target" not in decision:
        return decision  # Explicit legacy callbacks; production requires target.
    if decision["done"]:
        if decision["target"]:
            raise CustomModuleGenerationError("ATOMIC_REGION_TARGET_INVALID: completion requires empty target")
        if context.get("deferred_work"):
            raise CustomModuleGenerationError("ATOMIC_REGION_TARGET_INVALID: unfinished parent work remains")
        return decision
    if context.get("response_region") != "members":
        if decision["target"]:
            raise CustomModuleGenerationError("ATOMIC_REGION_TARGET_INVALID: initialize uses ordered statements and empty target")
        return decision
    row = target_contract(decision["target"])
    completed = list(context.get("accepted_api", ()))
    completed.extend(class_body_member_contracts(context.get("current_page_source", "")))
    if any(_key(item) == _key(row) for item in completed):
        raise CustomModuleGenerationError(
            "ATOMIC_REGION_TARGET_ALREADY_ACCEPTED: choose unfinished work; " + decision["target"]
        )
    if context.get("completion_phase") == "refine_next_unit":
        for work in context.get("deferred_work", ()):
            if work.get("target") and _key(target_contract(work["target"])) == _key(row):
                raise CustomModuleGenerationError(
                    "ATOMIC_REGION_TARGET_NOT_REFINED: choose a smaller prerequisite, not an exhausted parent"
                )
    return decision


def implements_target(target: str, source: str) -> bool:
    if not target:
        return True
    expected = target_contract(target)
    return any(_contract(row) == _contract(expected) for row in class_body_member_contracts(source))


def conflicts_with_target(target: str, source: str) -> bool:
    expected = target_contract(target)
    return any(
        _key(row) == _key(expected) and _contract(row) != _contract(expected)
        for row in class_body_member_contracts(source)
    )


def work_receipt(work: Mapping, source: str, member_keys: set[str]) -> dict:
    return {
        "target": work.get("target", ""), "purpose": work.get("next_work", ""),
        "member_keys": sorted(member_keys),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def defer_work(deferred: list[dict], work: Mapping) -> None:
    if work.get("target") and not any(item["target"] == work["target"] for item in deferred):
        deferred.append(deepcopy(dict(work)))
