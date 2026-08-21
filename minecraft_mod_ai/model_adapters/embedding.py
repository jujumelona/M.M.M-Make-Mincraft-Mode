from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import AdapterConfig, ModelBackendError, require_package


@dataclass
class _EmbeddingBackend:
    model: Any
    lock: threading.RLock = field(default_factory=threading.RLock)


_BACKEND_CACHE_LOCK = threading.RLock()
_BACKEND_CACHE: dict[tuple[str, str, str], _EmbeddingBackend] = {}
_VECTOR_CACHE_LOCK = threading.RLock()
_VECTOR_CACHE: OrderedDict[tuple[str, str, str, int, str], tuple[tuple[float, ...], ...]] = OrderedDict()


def _cache_enabled() -> bool:
    raw = os.environ.get("MMM_CPU_RETRIEVAL_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _result_cache_limit() -> int:
    raw = os.environ.get("MMM_CPU_RETRIEVAL_RESULT_CACHE_ENTRIES", "256").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 256
    return max(1, min(4096, value))


def _texts_digest(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in texts:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class EmbeddingAdapter:
    """Sentence embedding adapter with shared CPU residency and exact-result reuse.

    Retrieval adapters are short-lived at the router boundary, so instance-only
    caching still reloads the same model on every RAG batch. The backend cache is
    therefore owned here, next to the model lifecycle it controls. Exact repeated
    batches are memoized as well so adjacent retrieval phases do not recompute the
    same vectors on a CPU-only runtime.
    """

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        self._backend: _EmbeddingBackend | None = None

    def _backend_key(self) -> tuple[str, str, str]:
        return (
            self.config.model_id,
            str(self.config.extra.get("device", "cpu")),
            str(self.config.extra.get("revision", "")).strip(),
        )

    def _vector_cache_key(
        self,
        texts: Sequence[str],
        dimensions: int,
    ) -> tuple[str, str, str, int, str]:
        return (*self._backend_key(), dimensions, _texts_digest(texts))

    def _load_backend(self) -> _EmbeddingBackend:
        require_package("sentence-transformers", minimum="3.0.0")
        from sentence_transformers import SentenceTransformer

        model_id, device, revision = self._backend_key()
        options: dict[str, object] = {
            "device": device,
            "trust_remote_code": False,
        }
        if revision:
            options["revision"] = revision

        started = time.monotonic()
        print(
            "retrieval embedding: model load start",
            f"model={model_id}",
            f"device={device}",
            file=sys.stderr,
            flush=True,
        )
        model = SentenceTransformer(model_id, **options)
        configured_context = max(1, int(self.config.max_context))
        current_context = int(
            getattr(model, "max_seq_length", configured_context) or configured_context
        )
        model.max_seq_length = min(current_context, configured_context)
        backend = _EmbeddingBackend(model)
        print(
            "retrieval embedding: model load done",
            f"elapsed={time.monotonic() - started:.1f}s",
            f"max_seq_length={model.max_seq_length}",
            file=sys.stderr,
            flush=True,
        )
        return backend

    def _ensure_backend(self) -> _EmbeddingBackend:
        if self._backend is not None:
            return self._backend

        key = self._backend_key()
        if not _cache_enabled():
            self._backend = self._load_backend()
            return self._backend

        with _BACKEND_CACHE_LOCK:
            backend = _BACKEND_CACHE.get(key)
            if backend is None:
                backend = self._load_backend()
                _BACKEND_CACHE[key] = backend
            self._backend = backend
            return backend

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [str(text).strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Embedding input must contain non-empty strings.")

        dimensions = int(self.config.extra.get("dimensions", 512))
        cache_key = self._vector_cache_key(cleaned, dimensions)
        if _cache_enabled():
            with _VECTOR_CACHE_LOCK:
                cached = _VECTOR_CACHE.get(cache_key)
                if cached is not None:
                    _VECTOR_CACHE.move_to_end(cache_key)
                    print(
                        "retrieval embedding: encode cache hit",
                        f"batch={len(cleaned)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return [list(row) for row in cached]

        try:
            backend = self._ensure_backend()
            started = time.monotonic()
            print(
                "retrieval embedding: encode start",
                f"batch={len(cleaned)}",
                file=sys.stderr,
                flush=True,
            )
            with backend.lock:
                vectors = backend.model.encode(
                    cleaned,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    truncate_dim=dimensions,
                    show_progress_bar=False,
                )
            print(
                "retrieval embedding: encode done",
                f"batch={len(cleaned)}",
                f"elapsed={time.monotonic() - started:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            result = tuple(tuple(float(value) for value in row) for row in vectors)
            if _cache_enabled():
                with _VECTOR_CACHE_LOCK:
                    _VECTOR_CACHE[cache_key] = result
                    _VECTOR_CACHE.move_to_end(cache_key)
                    while len(_VECTOR_CACHE) > _result_cache_limit():
                        _VECTOR_CACHE.popitem(last=False)
            return [list(row) for row in result]
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc


EmbeddingAdapter.embed._mmm_cached_embedding_model = True  # type: ignore[attr-defined]
