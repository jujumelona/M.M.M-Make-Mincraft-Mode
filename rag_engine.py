"""Compatibility exports for pinned, authoritative platform evidence.

This module does not pretend to run an unrestricted web retriever. Planning
consumes a code-owned official catalog whose versions and source URLs are
explicit and reviewable.
"""

from minecraft_mod_ai.knowledge import (
    AuthoritativeEvidenceRetriever,
    evidence_for_mvp,
    evidence_snapshot_hash,
    validate_trusted_evidence,
)
from minecraft_mod_ai.spec import EvidenceSource

Evidence = EvidenceSource


def official_evidence_catalog() -> tuple[EvidenceSource, ...]:
    return evidence_for_mvp()


__all__ = [
    "AuthoritativeEvidenceRetriever",
    "Evidence",
    "evidence_snapshot_hash",
    "official_evidence_catalog",
    "validate_trusted_evidence",
]
