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
