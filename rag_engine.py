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


class DeepRAGEngine:
    """9-Tier Authoritative RAG Engine adapter for minecraft_mod_ai."""
    def __init__(self):
        self.retriever = AuthoritativeEvidenceRetriever()

    def execute_6pass_rag(self, prompt: str, target_version: str = "1.20.1"):
        sources = self.retriever.retrieve(prompt, minecraft_version=target_version)
        print(f"[RAG] Retrieved {len(sources)} authoritative evidence sources for version {target_version}")
        return sources


def official_evidence_catalog() -> tuple[EvidenceSource, ...]:
    return evidence_for_mvp()


__all__ = [
    "AuthoritativeEvidenceRetriever",
    "DeepRAGEngine",
    "Evidence",
    "evidence_snapshot_hash",
    "official_evidence_catalog",
    "validate_trusted_evidence",
]
