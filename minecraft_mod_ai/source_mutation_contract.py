from __future__ import annotations

"""Canonical proof contract for reviewed source-mutation observations.

A successful transport is not a successful source mutation. Critical causal facts,
retry accounting, and final implementation completion all consume this module so a
single host-owned receipt decides whether the workspace was actually changed.
"""

import json
from typing import Any, Mapping, Sequence

SOURCE_MUTATION_NAMES = frozenset(
    {
        "apply_source_patch",
        "apply_source_edit",
        "apply_java_operations",
        "repair_project",
    }
)
STRICT_RECEIPT_MUTATION_NAMES = frozenset(
    {
        "apply_source_patch",
        "apply_source_edit",
    }
)

# Retained for wire compatibility with older tool observations. This marker records
# only that the host runtime executed a mutation primitive; it is not evidence that
# file bytes changed. Strict mutation proof comes from source-patch receipts below.
HOST_MUTATION_PROOF_KEY = "_mmm_source_mutation"

_FAILURE_STATUSES = frozenset(
    {
        "FAIL",
        "FAILED",
        "ERROR",
        "UNAVAILABLE",
        "PARTIAL",
        "BLOCKED",
        "INVALID",
        "REJECTED",
        "CANCELLED",
        "CANCELED",
        "TIMEOUT",
    }
)


def tool_payload(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    content = message.get("content")
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _walk_mappings(value: Any):
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            yield current
            pending.extend(current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current,
            (str, bytes, bytearray),
        ):
            pending.extend(current)


def _receipt_operation_changed(operation: Any) -> bool:
    if not isinstance(operation, Mapping):
        return False
    before = operation.get("before_sha256")
    after = operation.get("after_sha256")
    return before != after


def _has_applied_patch_receipt(payload: Mapping[str, Any]) -> bool:
    """Return true only for a host receipt that proves at least one byte-level diff."""

    for item in _walk_mappings(payload):
        if str(item.get("schema_version", "")) != "mmm/source-patch-receipt-v1":
            continue
        if str(item.get("status", "")).strip().upper() != "APPLIED":
            continue
        operations = item.get("operations")
        if not isinstance(operations, Sequence) or isinstance(
            operations,
            (str, bytes, bytearray),
        ):
            continue
        if any(_receipt_operation_changed(operation) for operation in operations):
            return True
    return False


def _has_semantic_failure(payload: Mapping[str, Any]) -> bool:
    for item in _walk_mappings(payload.get("result", payload)):
        if str(item.get("status", "")).strip().upper() in _FAILURE_STATUSES:
            return True
    return False


def mutation_payload_applied(name: str, payload: Mapping[str, Any]) -> bool:
    """Return true only when one reviewed mutation observation proves a source diff."""

    normalized = str(name).strip()
    if normalized not in SOURCE_MUTATION_NAMES or payload.get("ok") is not True:
        return False
    if _has_applied_patch_receipt(payload):
        return True
    if normalized in STRICT_RECEIPT_MUTATION_NAMES:
        return False
    return not _has_semantic_failure(payload)


def mutation_observation_applied(message: Mapping[str, Any]) -> bool:
    if str(message.get("role", "")).strip().casefold() != "tool":
        return False
    name = str(message.get("name", "")).strip()
    payload = tool_payload(message)
    return payload is not None and mutation_payload_applied(name, payload)


def mutation_history_applied(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return true when retained history contains a receipt-proven source diff."""

    return any(mutation_observation_applied(message) for message in reversed(messages))


__all__ = [
    "HOST_MUTATION_PROOF_KEY",
    "SOURCE_MUTATION_NAMES",
    "STRICT_RECEIPT_MUTATION_NAMES",
    "mutation_history_applied",
    "mutation_observation_applied",
    "mutation_payload_applied",
    "tool_payload",
]
