from __future__ import annotations

"""Compatibility policy for dense-reranker fallback RAG receipts.

A retrieval provider may return real hits while dense reranking is unavailable, leaving
coverage/relevance scores at zero. The host must not discard those concrete hits solely
because an optional scorer was unavailable. A receipt is still authoritative about
whether the provider returned anything: fallback acceptance therefore requires both a
positive receipt result_count and non-empty hits.
"""

from functools import wraps
from typing import Any, Mapping, Sequence


def _positive_receipt_with_hits(value: Any) -> bool:
    positive_receipt = False
    nonempty_hits = False

    def visit(item: Any) -> None:
        nonlocal positive_receipt, nonempty_hits
        if isinstance(item, Mapping):
            receipt = item.get("receipt")
            if isinstance(receipt, Mapping):
                try:
                    if int(receipt.get("result_count", 0) or 0) > 0:
                        positive_receipt = True
                except (TypeError, ValueError):
                    pass
            hits = item.get("hits")
            if (
                isinstance(hits, Sequence)
                and not isinstance(hits, (str, bytes, bytearray))
                and bool(hits)
            ):
                nonempty_hits = True
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for child in item:
                visit(child)

    visit(value)
    return positive_receipt and nonempty_hits


def install(model_router_module: Any) -> None:
    original = model_router_module._usable_rag_result
    if bool(getattr(original, "__mmm_reranker_fallback_hits__", False)):
        return

    @wraps(original)
    def usable_rag_result(value: Any) -> bool:
        if original(value):
            return True
        return _positive_receipt_with_hits(value)

    usable_rag_result.__mmm_reranker_fallback_hits__ = True
    model_router_module._usable_rag_result = usable_rag_result


__all__ = ["_positive_receipt_with_hits", "install"]
