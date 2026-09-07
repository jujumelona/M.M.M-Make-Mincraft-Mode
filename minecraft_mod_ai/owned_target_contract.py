from __future__ import annotations

"""Single owner for coder target-status semantics.

Target status decides whether an owned path may be created or must already exist. Callers
must not silently reinterpret unknown statuses because doing so can widen write authority.
"""

from typing import Any

CREATABLE_TARGET_STATUSES = frozenset({"host_reserved"})
EXISTING_TARGET_STATUSES = frozenset({"existing", "host_existing", "modify", "reuse"})
WRITABLE_TARGET_STATUSES = CREATABLE_TARGET_STATUSES | EXISTING_TARGET_STATUSES


def normalize_target_status(value: Any) -> str:
    return str(value or "").strip().casefold()


def target_is_creatable(value: Any) -> bool:
    return normalize_target_status(value) in CREATABLE_TARGET_STATUSES


def target_is_existing(value: Any) -> bool:
    return normalize_target_status(value) in EXISTING_TARGET_STATUSES


def target_is_writable(value: Any) -> bool:
    return normalize_target_status(value) in WRITABLE_TARGET_STATUSES


def target_operation(value: Any) -> str:
    status = normalize_target_status(value)
    if status in EXISTING_TARGET_STATUSES:
        return "modify"
    if status in CREATABLE_TARGET_STATUSES:
        return "create_or_modify"
    raise ValueError(f"unsupported owned target status: {status or '<empty>'}")


__all__ = [
    "CREATABLE_TARGET_STATUSES",
    "EXISTING_TARGET_STATUSES",
    "WRITABLE_TARGET_STATUSES",
    "normalize_target_status",
    "target_is_creatable",
    "target_is_existing",
    "target_is_writable",
    "target_operation",
]
