from __future__ import annotations

"""Canonical classification for mutation failures returned after tool execution."""

from collections.abc import Mapping, Sequence
from typing import Any

from .value_shapes import structured_payload

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
_POST_ARGUMENT_SEMANTIC_FAILURE_CODES = frozenset(
    {
        "MUTATION_TARGET_DRIFT",
        "MUTATION_TARGET_UNBOUND",
        "PATH_OUTSIDE_WRITABLE_SET",
        "PHASE_PROTOCOL_VIOLATION",
    }
)
_POST_ARGUMENT_SEMANTIC_FAILURE_PREFIXES = (
    "MUTATION_AUTHORITY_",
    "MUTATION_TARGET_",
    "WRITE_SCOPE_",
)


def normalize_mutation_failure_code(code: Any) -> str:
    return str(code or "").strip().upper()


def is_recoverable_mutation_failure(code: Any) -> bool:
    return normalize_mutation_failure_code(code) in RECOVERABLE_MUTATION_FAILURE_CODES


def is_post_argument_semantic_failure_code(code: Any) -> bool:
    normalized = normalize_mutation_failure_code(code)
    if not normalized or is_recoverable_mutation_failure(normalized):
        return False
    return normalized in _POST_ARGUMENT_SEMANTIC_FAILURE_CODES or normalized.startswith(
        _POST_ARGUMENT_SEMANTIC_FAILURE_PREFIXES
    )


def latest_post_argument_semantic_failure(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, str] | None:
    """Return the latest trailing non-recoverable tool semantic rejection."""

    for message in reversed(messages):
        if not isinstance(message, Mapping):
            break
        if str(message.get("role") or "").strip().casefold() != "tool":
            break
        payload = structured_payload(message.get("content"))
        if not isinstance(payload, Mapping) or payload.get("ok") is not False:
            continue
        code = normalize_mutation_failure_code(payload.get("failure_code"))
        if not is_post_argument_semantic_failure_code(code):
            continue
        reason = str(payload.get("error") or payload.get("reason") or code).strip()
        return code, reason
    return None


__all__ = [
    "RECOVERABLE_MUTATION_FAILURE_CODES",
    "is_post_argument_semantic_failure_code",
    "is_recoverable_mutation_failure",
    "latest_post_argument_semantic_failure",
    "normalize_mutation_failure_code",
]
