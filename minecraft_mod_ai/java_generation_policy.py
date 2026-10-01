from __future__ import annotations

"""Compatibility facade over the canonical execution contract.

All hard policy data and helper decisions live in execution_contract_policy. This
module exists only so older imports do not fork the execution contract.
"""

import re

from .execution_contract_policy import (
    DEFAULT_COMPILE_REPAIR_LIMIT,
    DEFAULT_REGION_ATTEMPT_LIMIT,
    JAVA_EXPLICIT_JDK_IMPORT_PATTERN,
    JAVA_FENCE_LANGUAGES,
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
    java_imported_jdk_use_can_be_qualified,
    java_initialize_wrapper_allowed,
    java_localize_initialize_field_modifiers,
    java_member_jdk_import_allowed,
    java_region_recovery_shapes,
)

EXPLICIT_JDK_IMPORT_PATTERN = re.compile(JAVA_EXPLICIT_JDK_IMPORT_PATTERN)
REGION_RECOVERY_SHAPES = JAVA_REGION_RECOVERY_SHAPES


def region_recovery_shapes(region: str) -> tuple[str, ...]:
    return java_region_recovery_shapes(region)


def initialize_wrapper_allowed(
    name: str,
    return_type: str,
    parameters: str,
) -> bool:
    return java_initialize_wrapper_allowed(name, return_type, parameters)


def localize_initialize_field_modifiers(
    modifiers: str,
) -> tuple[str, ...] | None:
    return java_localize_initialize_field_modifiers(modifiers)


def member_jdk_import_allowed(fqcn: str) -> bool:
    return java_member_jdk_import_allowed(fqcn)


def imported_jdk_use_can_be_qualified(role: str) -> bool:
    return java_imported_jdk_use_can_be_qualified(role)


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
