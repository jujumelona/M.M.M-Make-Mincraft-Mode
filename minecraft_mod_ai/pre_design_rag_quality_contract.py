from __future__ import annotations

"""Compatibility facade for the side-effect-free pre-design RAG quality stack."""

from .pre_design_rag_corrective import (
    _correction_queries,
    _quality_research_document_domain,
    _read_and_verify_document,
)
from .pre_design_rag_fusion import (
    _is_retrieval_query,
    fuse_grounded_domain_evidence,
)
from .pre_design_rag_support import _verify_page_claims

_INSTALLED = False


def install() -> None:
    """Compatibility no-op; canonical research owners call the stack directly."""

    global _INSTALLED
    _INSTALLED = True


__all__ = ["fuse_grounded_domain_evidence", "install"]
