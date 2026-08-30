from __future__ import annotations

"""Compatibility facade for the side-effect-free pre-design RAG quality stack."""

from .pre_design_rag_corrective import (
    _correction_queries as _correction_queries,
)
from .pre_design_rag_corrective import (
    _quality_research_document_domain as _quality_research_document_domain,
)
from .pre_design_rag_corrective import (
    _read_and_verify_document as _read_and_verify_document,
)
from .pre_design_rag_fusion import (
    _is_retrieval_query as _is_retrieval_query,
)
from .pre_design_rag_fusion import (
    fuse_grounded_domain_evidence as fuse_grounded_domain_evidence,
)
from .pre_design_rag_support import _verify_page_claims as _verify_page_claims

_INSTALLED = False


def install() -> None:
    """Compatibility no-op; canonical research owners call the stack directly."""

    global _INSTALLED
    _INSTALLED = True


__all__ = [
    "_correction_queries",
    "_is_retrieval_query",
    "_quality_research_document_domain",
    "_read_and_verify_document",
    "_verify_page_claims",
    "fuse_grounded_domain_evidence",
    "install",
]
