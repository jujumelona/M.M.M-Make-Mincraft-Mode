from __future__ import annotations

"""Shared classification for mutation failures returned after tool execution.

Recoverable mutation-state failures describe a stale or incompatible requested edit
against the current workspace. They require a corrective tool turn, not escalation to
a model-configuration failure. Authority/scope violations are intentionally excluded.
"""

from typing import Any

RECOVERABLE_MUTATION_FAILURE_CODES = frozenset(
    {
        "MUTATION_TARGET_CREATION_CONFLICT",
        "MUTATION_TARGET_MISSING",
        "MUTATION_TARGET_ALREADY_EXISTS",
        "MUTATION_ANCHOR_NOT_FOUND",
        "MUTATION_ANCHOR_AMBIGUOUS",
        "MUTATION_PRECONDITION_FAILED",
    }
)


def normalize_mutation_failure_code(code: Any) -> str:
    return str(code or "").strip().upper()


def is_recoverable_mutation_failure(code: Any) -> bool:
    return normalize_mutation_failure_code(code) in RECOVERABLE_MUTATION_FAILURE_CODES


__all__ = [
    "RECOVERABLE_MUTATION_FAILURE_CODES",
    "is_recoverable_mutation_failure",
    "normalize_mutation_failure_code",
]
