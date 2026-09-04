from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Sequence

_GIB = 1024**3
_MAX_RERANK_MICROBATCH = 32


def _retrieval_cache_enabled() -> bool:
    raw = os.environ.get("MMM_CPU_RETRIEVAL_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _retrieval_result_cache_limit() -> int:
    raw = os.environ.get("MMM_CPU_RETRIEVAL_RESULT_CACHE_ENTRIES", "1024").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 1024
    return max(1, min(16384, value))


def _length_prefixed_digest(values: Sequence[str]) -> str:
    """Return a collision-safe digest for an ordered sequence of UTF-8 strings."""

    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _text_digest(text: str) -> str:
    """Fast path for the single-string digest contract used by retrieval caches."""

    encoded = text.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _available_memory_bytes() -> int | None:
    """Return live available system memory without requiring psutil."""

    try:
        import psutil

        return max(0, int(psutil.virtual_memory().available))
    except (ImportError, AttributeError, OSError, ValueError):
        pass

    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return max(0, int(status.ullAvailPhys))
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return max(0, pages * page_size)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _rerank_microbatch_size(document_count: int) -> int:
    """Choose a bounded batch size from an override or live CPU/RAM capacity."""

    if document_count <= 0:
        return 0
    raw_override = os.environ.get("MMM_RERANK_MICROBATCH", "").strip()
    if raw_override:
        try:
            requested = int(raw_override)
        except ValueError as exc:
            raise ValueError(
                "MMM_RERANK_MICROBATCH must be a positive integer."
            ) from exc
        if requested <= 0:
            raise ValueError("MMM_RERANK_MICROBATCH must be a positive integer.")
        return min(document_count, requested, _MAX_RERANK_MICROBATCH)

    cpu_capacity = min(_MAX_RERANK_MICROBATCH, max(1, (os.cpu_count() or 1) * 2))
    available = _available_memory_bytes()
    ram_capacity = (
        _MAX_RERANK_MICROBATCH
        if available is None
        else max(1, min(_MAX_RERANK_MICROBATCH, available // (2 * _GIB)))
    )
    return min(document_count, cpu_capacity, ram_capacity)


def _length_bucketed_batches(
    rendered: Sequence[str],
    batch_size: int,
) -> Iterator[list[tuple[int, str]]]:
    """Yield stable short-to-long batches while retaining each original index."""

    if batch_size <= 0:
        raise ValueError("Reranker microbatch size must be positive.")
    ordered = sorted(enumerate(rendered), key=lambda item: (len(item[1]), item[0]))
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


def install() -> None:
    """Verify that retrieval performance ownership lives in the adapters themselves."""

    from .model_adapters.embedding import EmbeddingAdapter
    from .model_adapters.reranker import RerankerAdapter

    if not getattr(EmbeddingAdapter.embed, "_mmm_cached_embedding_model", False):
        raise RuntimeError("Embedding backend residency contract is not installed natively.")
    if not getattr(RerankerAdapter.score, "_mmm_cached_reranker_model", False):
        raise RuntimeError("Reranker backend residency contract is not installed natively.")
