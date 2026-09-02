from __future__ import annotations

"""Request-local host state for grounded pre-design evidence.

The state contains counts only. It never stores source bodies or model-authored facts.
It exists so downstream acceptance guards can reject a blanket "no evidence" statement
when the host has already materialized grounded source pages for the active request.
"""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundedEvidenceState:
    prompt: str = ""
    source_body_count: int = 0
    evidence_card_count: int = 0

    @property
    def available(self) -> bool:
        return self.source_body_count > 0 or self.evidence_card_count > 0


_STATE: ContextVar[GroundedEvidenceState] = ContextVar(
    "mmm_grounded_predesign_evidence_state",
    default=GroundedEvidenceState(),
)


def record_grounded_evidence(
    prompt: str,
    *,
    source_body_count: int,
    evidence_card_count: int,
) -> GroundedEvidenceState:
    """Record request-local host evidence without weakening a prior domain result."""

    normalized_prompt = str(prompt or "")
    current = _STATE.get()
    if current.prompt != normalized_prompt:
        value = GroundedEvidenceState(
            prompt=normalized_prompt,
            source_body_count=max(0, int(source_body_count)),
            evidence_card_count=max(0, int(evidence_card_count)),
        )
    else:
        value = GroundedEvidenceState(
            prompt=normalized_prompt,
            source_body_count=max(current.source_body_count, max(0, int(source_body_count))),
            evidence_card_count=max(current.evidence_card_count, max(0, int(evidence_card_count))),
        )
    _STATE.set(value)
    return value


def current_grounded_evidence() -> GroundedEvidenceState:
    return _STATE.get()


def grounded_evidence_available() -> bool:
    return _STATE.get().available


__all__ = [
    "GroundedEvidenceState",
    "current_grounded_evidence",
    "grounded_evidence_available",
    "record_grounded_evidence",
]
