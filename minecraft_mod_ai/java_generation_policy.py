from __future__ import annotations

"""Compatibility facade over the canonical execution contract.

This module retains Java-specific helper functions used by parsers and older callers.
Hard policy data lives exclusively in execution_contract_policy.
"""

import re

from .execution_contract_policy import (
    DEFAULT_COMPILE_REPAIR_LIMIT,
    DEFAULT_REGION_ATTEMPT_LIMIT,
    JAVA_EXPLICIT_JDK_IMPORT_PATTERN,
    JAVA_FENCE_LANGUAGES,
    JAVA_INITIALIZE_LOCALIZABLE_FIELD_MODIFIERS,
    JAVA_INITIALIZE_PRESERVED_LOCAL_MODIFIERS,
    JAVA_INITIALIZE_WRAPPER_NAMES,
    JAVA_QUALIFIABLE_JDK_IMPORT_USE_ROLES,
    JAVA_REGION_RECOVERY_SHAPES,
    MAX_COMPILE_REPAIR_LIMIT,
    MAX_REGION_ATTEMPT_LIMIT,
    PRODUCTION_CANONICALIZE_LOCAL_FINAL_REBINDINGS,
    PRODUCTION_COMPILE_REPAIR_LIMIT,
    PRODUCTION_REGION_ATTEMPT_LIMIT,
    PRODUCTION_RETRY_STRUCTURAL_REJECTIONS,
    RECOVERABLE_ATOMIC_ERROR_PREFIXES,
    TERMINAL_AFTER_NORMALIZATION_PREFIXES,
    atomic_error_recoverable,
    atomic_error_terminal_after_normalization,
    authorized_concern_nested_type_symbols,
    java_generation_shared_recipe_policy,
)

EXPLICIT_JDK_IMPORT_PATTERN = re.compile(JAVA_EXPLICIT_JDK_IMPORT_PATTERN)
REGION_RECOVERY_SHAPES = JAVA_REGION_RECOVERY_SHAPES


def region_recovery_shapes(region: str) -> tuple[str, ...]:
    return tuple(JAVA_REGION_RECOVERY_SHAPES.get(str(region or "").strip(), ()))


def initialize_wrapper_allowed(
    name: str,
    return_type: str,
    parameters: str,
) -> bool:
    return (
        str(name or "").strip() in JAVA_INITIALIZE_WRAPPER_NAMES
        and str(return_type or "").strip() == "void"
        and str(parameters or "").strip() == "()"
    )


def localize_initialize_field_modifiers(
    modifiers: str,
) -> tuple[str, ...] | None:
    text = str(modifiers or "").strip()
    if not text:
        return ()
    if "@" in text:
        return None
    tokens = tuple(item for item in text.split() if item)
    if any(item not in JAVA_INITIALIZE_LOCALIZABLE_FIELD_MODIFIERS for item in tokens):
        return None
    return tuple(
        item for item in tokens if item in JAVA_INITIALIZE_PRESERVED_LOCAL_MODIFIERS
    )


def member_jdk_import_allowed(fqcn: str) -> bool:
    value = str(fqcn or "").strip()
    return bool(value.startswith("java.") and "*" not in value)


def imported_jdk_use_can_be_qualified(role: str) -> bool:
    return str(role or "").strip() in JAVA_QUALIFIABLE_JDK_IMPORT_USE_ROLES


def production_java_generation_recipe_policy() -> dict:
    return java_generation_shared_recipe_policy()


__all__ = [
    "DEFAULT_COMPILE_REPAIR_LIMIT",
    "DEFAULT_REGION_ATTEMPT_LIMIT",
    "EXPLICIT_JDK_IMPORT_PATTERN",
    "JAVA_FENCE_LANGUAGES",
    "MAX_COMPILE_REPAIR_LIMIT",
    "MAX_REGION_ATTEMPT_LIMIT",
    "RECOVERABLE_ATOMIC_ERROR_PREFIXES",
    "REGION_RECOVERY_SHAPES",
    "TERMINAL_AFTER_NORMALIZATION_PREFIXES",
    "PRODUCTION_CANONICALIZE_LOCAL_FINAL_REBINDINGS",
    "PRODUCTION_COMPILE_REPAIR_LIMIT",
    "PRODUCTION_REGION_ATTEMPT_LIMIT",
    "PRODUCTION_RETRY_STRUCTURAL_REJECTIONS",
    "atomic_error_recoverable",
    "authorized_concern_nested_type_symbols",
    "atomic_error_terminal_after_normalization",
    "imported_jdk_use_can_be_qualified",
    "initialize_wrapper_allowed",
    "localize_initialize_field_modifiers",
    "member_jdk_import_allowed",
    "production_java_generation_recipe_policy",
    "region_recovery_shapes",
]
