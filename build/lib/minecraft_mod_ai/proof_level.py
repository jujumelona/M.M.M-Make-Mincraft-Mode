from __future__ import annotations

"""Fail-closed proof state machine and lifecycle level definitions.

Every state that represents evidence stronger than discovery is gated by the
minimum receipt fields needed to justify that exact transition.  A non-empty
mapping is never sufficient by itself to manufacture a verified state.
"""

import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

from .reuse_license import is_reusable_source_license


class ProofLevel(str, Enum):
    DISCOVERED = "DISCOVERED"
    LICENSE_VERIFIED = "LICENSE_VERIFIED"
    PINNED = "PINNED"
    CLOSURE_COMPLETE = "CLOSURE_COMPLETE"
    MATERIALIZED = "MATERIALIZED"
    SUBGRAPH_COMPILE_VERIFIED = "SUBGRAPH_COMPILE_VERIFIED"
    PARTIAL_REUSE = "PARTIAL_REUSE"
    COMPILE_VERIFIED = "COMPILE_VERIFIED"
    INTEGRATION_VERIFIED = "INTEGRATION_VERIFIED"
    RUNTIME_BOOT_VERIFIED = "RUNTIME_BOOT_VERIFIED"
    BEHAVIOR_VERIFIED = "BEHAVIOR_VERIFIED"
    HOST_VERIFIED = "HOST_VERIFIED"
    FRESH_REQUIRED = "FRESH_REQUIRED"
    UNVERIFIED = "UNVERIFIED"

    @classmethod
    def from_value(cls, val: Any) -> ProofLevel:
        if isinstance(val, ProofLevel):
            return val
        if isinstance(val, str):
            try:
                return cls(val.strip().upper())
            except ValueError:
                return cls.UNVERIFIED
        return cls.UNVERIFIED

    def is_verified(self) -> bool:
        """Return True only for states backed by executable build/runtime evidence."""
        return self in {
            ProofLevel.COMPILE_VERIFIED,
            ProofLevel.INTEGRATION_VERIFIED,
            ProofLevel.RUNTIME_BOOT_VERIFIED,
            ProofLevel.BEHAVIOR_VERIFIED,
            ProofLevel.HOST_VERIFIED,
        }

    def is_partial(self) -> bool:
        """Return True if isolated subgraphs passed but residual work remains."""
        return self in {
            ProofLevel.PARTIAL_REUSE,
            ProofLevel.SUBGRAPH_COMPILE_VERIFIED,
        }

    def allows_reuse(self) -> bool:
        """Return True only when full or partial executable proof exists."""
        return self.is_verified() or self.is_partial()


_LEGAL_TRANSITIONS: dict[ProofLevel, set[ProofLevel]] = {
    ProofLevel.DISCOVERED: {
        ProofLevel.LICENSE_VERIFIED,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.LICENSE_VERIFIED: {
        ProofLevel.PINNED,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.PINNED: {
        ProofLevel.CLOSURE_COMPLETE,
        ProofLevel.MATERIALIZED,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.CLOSURE_COMPLETE: {
        ProofLevel.MATERIALIZED,
        ProofLevel.SUBGRAPH_COMPILE_VERIFIED,
        ProofLevel.COMPILE_VERIFIED,
        ProofLevel.PARTIAL_REUSE,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.MATERIALIZED: {
        ProofLevel.SUBGRAPH_COMPILE_VERIFIED,
        ProofLevel.COMPILE_VERIFIED,
        ProofLevel.PARTIAL_REUSE,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.SUBGRAPH_COMPILE_VERIFIED: {
        ProofLevel.PARTIAL_REUSE,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.PARTIAL_REUSE: {
        ProofLevel.INTEGRATION_VERIFIED,
        ProofLevel.HOST_VERIFIED,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.COMPILE_VERIFIED: {
        ProofLevel.INTEGRATION_VERIFIED,
        ProofLevel.RUNTIME_BOOT_VERIFIED,
        ProofLevel.BEHAVIOR_VERIFIED,
        ProofLevel.HOST_VERIFIED,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.INTEGRATION_VERIFIED: {
        ProofLevel.RUNTIME_BOOT_VERIFIED,
        ProofLevel.BEHAVIOR_VERIFIED,
        ProofLevel.HOST_VERIFIED,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.RUNTIME_BOOT_VERIFIED: {
        ProofLevel.BEHAVIOR_VERIFIED,
        ProofLevel.HOST_VERIFIED,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.BEHAVIOR_VERIFIED: {
        ProofLevel.HOST_VERIFIED,
        ProofLevel.UNVERIFIED,
        ProofLevel.FRESH_REQUIRED,
    },
    ProofLevel.HOST_VERIFIED: {ProofLevel.UNVERIFIED, ProofLevel.FRESH_REQUIRED},
    ProofLevel.FRESH_REQUIRED: {ProofLevel.UNVERIFIED},
    ProofLevel.UNVERIFIED: {ProofLevel.DISCOVERED, ProofLevel.FRESH_REQUIRED},
}


def _receipt_mapping(receipt: Any) -> Mapping[str, Any] | None:
    return receipt if isinstance(receipt, Mapping) else None


def _nonempty_text(receipt: Mapping[str, Any], key: str) -> bool:
    return bool(str(receipt.get(key) or "").strip())


def _immutable_commit_sha(receipt: Mapping[str, Any], key: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40,64}", str(receipt.get(key) or "").strip()))


def _positive_int(receipt: Mapping[str, Any], key: str) -> bool:
    value = receipt.get(key)
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_receipt(dst: ProofLevel, receipt: Any) -> tuple[bool, str]:
    """Validate the minimum evidence required for one destination state."""

    requirements: dict[ProofLevel, tuple[str, ...]] = {
        ProofLevel.LICENSE_VERIFIED: ("license",),
        ProofLevel.PINNED: ("commit_sha",),
        ProofLevel.CLOSURE_COMPLETE: ("closure_complete",),
        ProofLevel.MATERIALIZED: ("files",),
        ProofLevel.SUBGRAPH_COMPILE_VERIFIED: ("verified_subgraphs",),
        ProofLevel.PARTIAL_REUSE: ("partial",),
        ProofLevel.COMPILE_VERIFIED: ("compile_passed",),
        ProofLevel.INTEGRATION_VERIFIED: ("integration_passed",),
        ProofLevel.RUNTIME_BOOT_VERIFIED: ("runtime_boot_passed",),
        ProofLevel.BEHAVIOR_VERIFIED: (
            "acceptance_passed",
            "count",
            "implementation_bound",
            "exact_results",
            "test_source_hash",
        ),
        ProofLevel.HOST_VERIFIED: ("host_verified",),
    }
    if dst not in requirements:
        return True, "receipt_not_required"

    data = _receipt_mapping(receipt)
    if data is None:
        return False, f"MISSING_RECEIPT: transition to {dst.value} requires an attested proof receipt"

    if dst == ProofLevel.LICENSE_VERIFIED and not is_reusable_source_license(
        data.get("license")
    ):
        return False, "INVALID_RECEIPT: LICENSE_VERIFIED requires a reusable source license"
    if dst == ProofLevel.PINNED and not _immutable_commit_sha(data, "commit_sha"):
        return False, "INVALID_RECEIPT: PINNED requires an immutable 40-64 hex commit_sha"
    if dst == ProofLevel.CLOSURE_COMPLETE and data.get("closure_complete") is not True:
        return False, "INVALID_RECEIPT: CLOSURE_COMPLETE requires closure_complete=true"
    if dst == ProofLevel.MATERIALIZED and not _positive_int(data, "files"):
        return False, "INVALID_RECEIPT: MATERIALIZED requires files>0"
    if dst == ProofLevel.SUBGRAPH_COMPILE_VERIFIED:
        if not _positive_int(data, "verified_subgraphs"):
            return False, "INVALID_RECEIPT: SUBGRAPH_COMPILE_VERIFIED requires verified_subgraphs>0"
        if data.get("authoritative_compile") is not True:
            return False, "INVALID_RECEIPT: SUBGRAPH_COMPILE_VERIFIED requires authoritative compile execution"
    if dst == ProofLevel.PARTIAL_REUSE and data.get("partial") is not True:
        return False, "INVALID_RECEIPT: PARTIAL_REUSE requires partial=true"
    if dst == ProofLevel.COMPILE_VERIFIED:
        if data.get("compile_passed") is not True:
            return False, "INVALID_RECEIPT: COMPILE_VERIFIED requires compile_passed=true"
        if data.get("authoritative_compile") is not True:
            return False, "INVALID_RECEIPT: COMPILE_VERIFIED requires authoritative compile execution"
    if dst == ProofLevel.INTEGRATION_VERIFIED and data.get("integration_passed") is not True:
        return False, "INVALID_RECEIPT: INTEGRATION_VERIFIED requires integration_passed=true"
    if dst == ProofLevel.RUNTIME_BOOT_VERIFIED and data.get("runtime_boot_passed") is not True:
        return False, "INVALID_RECEIPT: RUNTIME_BOOT_VERIFIED requires runtime_boot_passed=true"
    if dst == ProofLevel.BEHAVIOR_VERIFIED:
        if data.get("acceptance_passed") is not True:
            return False, "INVALID_RECEIPT: BEHAVIOR_VERIFIED requires acceptance_passed=true"
        if not _positive_int(data, "count"):
            return False, "INVALID_RECEIPT: BEHAVIOR_VERIFIED requires count>0"
        if data.get("implementation_bound") is not True:
            return False, "INVALID_RECEIPT: BEHAVIOR_VERIFIED requires implementation_bound=true"
        if data.get("exact_results") is not True:
            return False, "INVALID_RECEIPT: BEHAVIOR_VERIFIED requires exact_results=true"
        if not _nonempty_text(data, "test_source_hash"):
            return False, "INVALID_RECEIPT: BEHAVIOR_VERIFIED requires test_source_hash"
    if dst == ProofLevel.HOST_VERIFIED and data.get("host_verified") is not True:
        return False, "INVALID_RECEIPT: HOST_VERIFIED requires host_verified=true"

    return True, "receipt_valid"


def validate_proof_transition(
    from_level: Any,
    to_level: Any,
    *,
    receipt: Any = None,
) -> tuple[bool, str]:
    """Strictly validate a legal transition and the evidence for its destination."""
    src = ProofLevel.from_value(from_level)
    dst = ProofLevel.from_value(to_level)

    if src == dst:
        return True, "identity_transition"

    allowed = _LEGAL_TRANSITIONS.get(src, set())
    if dst not in allowed:
        return False, f"ILLEGAL_TRANSITION: cannot jump from {src.value} to {dst.value}"

    valid_receipt, reason = _validate_receipt(dst, receipt)
    if not valid_receipt:
        return False, reason

    return True, "transition_valid"
