from __future__ import annotations

"""Single production policy for model-authored Java admission and validation.

Parser normalization, atomic validation/retry classification, production generation
budgets, and coder-facing Java rules must read this module instead of maintaining
independent policy literals.
"""

import re
from typing import Any


DEFAULT_REGION_ATTEMPT_LIMIT = 1
MAX_REGION_ATTEMPT_LIMIT = 4
DEFAULT_COMPILE_REPAIR_LIMIT = 0
MAX_COMPILE_REPAIR_LIMIT = 16

PRODUCTION_REGION_ATTEMPT_LIMIT = 3
PRODUCTION_RETRY_STRUCTURAL_REJECTIONS = False
PRODUCTION_CANONICALIZE_LOCAL_FINAL_REBINDINGS = True
PRODUCTION_COMPILE_REPAIR_LIMIT = 0

JAVA_FENCE_LANGUAGES = frozenset({"", "java", "javac"})
EXPLICIT_JDK_IMPORT_PATTERN = re.compile(
    r"import\s+(java\.[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+)\s*;"
)
QUALIFIABLE_JDK_IMPORT_USE_ROLES = frozenset({"type", "static_receiver"})

RECOVERABLE_ATOMIC_ERROR_PREFIXES = (
    "ATOMIC_CONCERN_RESPONSE_INVALID:",
    "ATOMIC_CONCERN_SCOPE_ESCAPE:",
    "ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE:",
    "ATOMIC_CONCERN_SYMBOL_COLLISION:",
    "ATOMIC_CONCERN_OUTPUT_EXHAUSTED:",
    "ATOMIC_CONCERN_SEMANTIC_SHAPE_INVALID:",
    "ATOMIC_CONCERN_PLATFORM_API_FORBIDDEN:",
    "ATOMIC_CONCERN_UNGROUNDED_PLATFORM_API:",
)
TERMINAL_AFTER_NORMALIZATION_PREFIXES = (
    "ATOMIC_CONCERN_SCOPE_ESCAPE:",
    "ATOMIC_CONCERN_SYMBOL_COLLISION:",
    "ATOMIC_CONCERN_PLATFORM_API_FORBIDDEN:",
    "ATOMIC_CONCERN_UNGROUNDED_PLATFORM_API:",
)

COMPILER_FIRST_RULES = (
    "The first answer must compile as Java for the selected host JDK; do not rely on a later repair pass.",
    "Never guess a package or fully-qualified class name. Use only a JDK/external type whose canonical package and API are known from the supplied authority.",
    "For non-java.lang JDK types, prefer canonical fully-qualified names. If an explicit java.* import is emitted, the host will remove it and qualify both type uses and static class receivers before validation.",
    "A field declaration type must be assignment-compatible with its initializer, and every receiver method call must exist on that declared type. Never use Map/List/Object as a lock holder merely because the field also guards cached state.",
    "Every concern-local final field must be definitely assigned before any read. Prefer initialization at the declaration; use a blank final only when the same region performs exactly one unconditional assignment in a static initializer.",
    "Never reassign a final field. If the binding must change, declare a non-final field; if a final field holds a mutable container, mutate the container rather than rebinding the field.",
    "Respect available_sibling_api types, generic arguments, and mutability exactly; final sibling fields are read-only after declaration.",
    "Java generics are invariant. Never narrow Map<K,Object> to Map<K,String>, List<Object> to List<String>, or any sibling generic declaration to a different type argument.",
    "For Map<K,V>.entrySet(), declare the iterator element as Map.Entry<K,V>; entry.getKey() has exact type K and entry.getValue() has exact type V. Preserve those exact types in local variables.",
    "Keep generic types exact. When an API returns Object, never return it directly from a method with a narrower generic/container return type and never use an unchecked cast as a shortcut; narrow the individual Object value with instanceof/pattern matching and provide a type-compatible fallback.",
    "Avoid raw collections and unchecked operations when a parameterized type or runtime type check can express the contract.",
    "For java.util.concurrent locks, Lock and ReentrantLock are in java.util.concurrent.locks, not java.util.concurrent.",
)

JDK_PACKAGE_ANCHORS = (
    ("collections_and_core_util", "java.util"),
    ("concurrency_executors_and_concurrent_collections", "java.util.concurrent"),
    ("locks", "java.util.concurrent.locks"),
    ("lock_interface", "java.util.concurrent.locks.Lock"),
    ("reentrant_lock", "java.util.concurrent.locks.ReentrantLock"),
    ("atomics", "java.util.concurrent.atomic"),
    ("time", "java.time"),
)

PRE_EMIT_COMPILE_CHECKLIST = (
    "Before emitting Java, internally type-check every assignment: declared_type <- expression_type.",
    "For every field or local receiver.method(...), verify the method exists on the receiver's declared type.",
    "For every generic projection, preserve exact invariant type arguments from available_sibling_api.",
    "For every return statement, verify the expression type is assignable to the declared return type.",
    "For every constructor call, verify the canonical JDK/package owner and constructor arguments. If a JDK type is factory-owned, use its public static factory instead of inventing a constructor.",
    "The installed-JDK javap contract is authoritative; never assume that a public JDK class has a public constructor or method.",
    "Only after all checks pass, emit the final Java region with no reasoning prose.",
)


def member_jdk_import_allowed(fqcn: str) -> bool:
    value = str(fqcn or "").strip()
    return bool(value.startswith("java.") and "*" not in value)


def imported_jdk_use_can_be_qualified(role: str) -> bool:
    return str(role or "").strip() in QUALIFIABLE_JDK_IMPORT_USE_ROLES


def atomic_error_recoverable(reason: str) -> bool:
    return str(reason or "").startswith(RECOVERABLE_ATOMIC_ERROR_PREFIXES)


def atomic_error_terminal_after_normalization(reason: str) -> bool:
    return str(reason or "").startswith(TERMINAL_AFTER_NORMALIZATION_PREFIXES)


def production_java_generation_recipe_policy() -> dict[str, Any]:
    return {
        "compiler_first_rules": list(COMPILER_FIRST_RULES),
        "jdk_package_anchors": dict(JDK_PACKAGE_ANCHORS),
        "jdk_runtime_patterns": {
            "exclusive_lock": {
                "declared_type": "java.util.concurrent.locks.Lock",
                "initializer": "new java.util.concurrent.locks.ReentrantLock()",
                "receiver_methods": ["lock", "unlock", "tryLock"],
            },
            "map": {
                "declared_type": "java.util.Map<K,V>",
                "initializer_example": "new java.util.HashMap<>()",
                "receiver_methods": ["get", "put", "remove", "containsKey"],
            },
            "atomic_counter": {
                "declared_type": "java.util.concurrent.atomic.AtomicLong",
                "initializer_example": "new java.util.concurrent.atomic.AtomicLong(0L)",
                "receiver_methods": [
                    "get",
                    "set",
                    "incrementAndGet",
                    "compareAndSet",
                ],
            },
        },
        "pre_emit_compile_checklist": list(PRE_EMIT_COMPILE_CHECKLIST),
    }


__all__ = [
    "DEFAULT_COMPILE_REPAIR_LIMIT",
    "DEFAULT_REGION_ATTEMPT_LIMIT",
    "EXPLICIT_JDK_IMPORT_PATTERN",
    "JAVA_FENCE_LANGUAGES",
    "MAX_COMPILE_REPAIR_LIMIT",
    "MAX_REGION_ATTEMPT_LIMIT",
    "PRODUCTION_CANONICALIZE_LOCAL_FINAL_REBINDINGS",
    "PRODUCTION_COMPILE_REPAIR_LIMIT",
    "PRODUCTION_REGION_ATTEMPT_LIMIT",
    "PRODUCTION_RETRY_STRUCTURAL_REJECTIONS",
    "atomic_error_recoverable",
    "atomic_error_terminal_after_normalization",
    "imported_jdk_use_can_be_qualified",
    "member_jdk_import_allowed",
    "production_java_generation_recipe_policy",
]
