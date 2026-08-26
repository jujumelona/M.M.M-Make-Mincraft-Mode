from __future__ import annotations

"""Fail-Closed Proof State Machine and Lifecycle Level Definitions.

Explicit verifiable lifecycle states:
DISCOVERED -> LICENSE_VERIFIED -> PINNED -> CLOSURE_COMPLETE -> MATERIALIZED ->
SUBGRAPH_COMPILE_VERIFIED -> PARTIAL_REUSE -> COMPILE_VERIFIED -> BEHAVIOR_VERIFIED -> HOST_VERIFIED
"""

from enum import Enum
from typing import Any


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
        """Return True only if the proof level represents an attested build/test proof."""
        return self in {
            ProofLevel.COMPILE_VERIFIED,
            ProofLevel.INTEGRATION_VERIFIED,
            ProofLevel.RUNTIME_BOOT_VERIFIED,
            ProofLevel.BEHAVIOR_VERIFIED,
            ProofLevel.HOST_VERIFIED,
        }

    def is_partial(self) -> bool:
        """Return True if isolated subgraphs passed but residuals are required."""
        return self in {
            ProofLevel.PARTIAL_REUSE,
            ProofLevel.SUBGRAPH_COMPILE_VERIFIED,
        }

    def allows_reuse(self) -> bool:
        """Return True if either full or partial reuse proof is attested."""
        return self.is_verified() or self.is_partial()


_LEGAL_TRANSITIONS: dict[ProofLevel, set[ProofLevel]] = {
    ProofLevel.DISCOVERED: {ProofLevel.LICENSE_VERIFIED, ProofLevel.UNVERIFIED, ProofLevel.FRESH_REQUIRED},
    ProofLevel.LICENSE_VERIFIED: {ProofLevel.PINNED, ProofLevel.UNVERIFIED, ProofLevel.FRESH_REQUIRED},
    ProofLevel.PINNED: {ProofLevel.CLOSURE_COMPLETE, ProofLevel.MATERIALIZED, ProofLevel.UNVERIFIED, ProofLevel.FRESH_REQUIRED},
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
    ProofLevel.SUBGRAPH_COMPILE_VERIFIED: {ProofLevel.PARTIAL_REUSE, ProofLevel.UNVERIFIED, ProofLevel.FRESH_REQUIRED},
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
    ProofLevel.BEHAVIOR_VERIFIED: {ProofLevel.HOST_VERIFIED, ProofLevel.UNVERIFIED, ProofLevel.FRESH_REQUIRED},
    ProofLevel.HOST_VERIFIED: {ProofLevel.UNVERIFIED, ProofLevel.FRESH_REQUIRED},
    ProofLevel.FRESH_REQUIRED: {ProofLevel.UNVERIFIED},
    ProofLevel.UNVERIFIED: {ProofLevel.DISCOVERED, ProofLevel.FRESH_REQUIRED},
}


def validate_proof_transition(
    from_level: Any,
    to_level: Any,
    *,
    receipt: Any = None,
) -> tuple[bool, str]:
    """Strictly validate whether a proof level state transition is legally permissible."""
    src = ProofLevel.from_value(from_level)
    dst = ProofLevel.from_value(to_level)

    if src == dst:
        return True, "identity_transition"

    allowed = _LEGAL_TRANSITIONS.get(src, set())
    if dst not in allowed:
        return False, f"ILLEGAL_TRANSITION: cannot jump from {src.value} to {dst.value}"

    if dst in {ProofLevel.COMPILE_VERIFIED, ProofLevel.BEHAVIOR_VERIFIED, ProofLevel.HOST_VERIFIED, ProofLevel.SUBGRAPH_COMPILE_VERIFIED, ProofLevel.PARTIAL_REUSE}:
        if receipt is None:
            return False, f"MISSING_RECEIPT: transition to {dst.value} requires an attested proof receipt"

    return True, "transition_valid"
