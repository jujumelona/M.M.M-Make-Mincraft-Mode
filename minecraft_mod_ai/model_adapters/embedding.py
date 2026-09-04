from __future__ import annotations

import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..model_runtime_performance import (
    _retrieval_cache_enabled as _cache_enabled,
)
from ..model_runtime_performance import (
    _retrieval_result_cache_limit as _result_cache_limit,
)
from ..model_runtime_performance import (
    _text_digest,
)
from ..retrieval_cpu_budget_contract import require_dense_retrieval_device
from .base import AdapterConfig, ModelBackendError, require_package


@dataclass
class _EmbeddingBackend:
    model: Any
    lock: threading.RLock = field(default_factory=threading.RLock)


_BACKEND_CACHE_LOCK = threading.RLock()
_BACKEND_CACHE: dict[tuple[str, str, str], _EmbeddingBackend] = {}
_VECTOR_CACHE_LOCK = threading.RLock()
_VECTOR_CACHE: OrderedDict[tuple[str, str, str, int, str], tuple[float, ...]] = OrderedDict()


class EmbeddingAdapter:
    """Sentence embedding adapter with shared CPU residency and per-text reuse.

    Retrieval adapters are short-lived at the router boundary, so instance-only
    caching still reloads the same model on every RAG batch. The backend cache is
    therefore owned here, next to the model lifecycle it controls. Vector results
    are memoized per normalized text instead of per batch so overlapping retrieval
    batches never recompute texts that were already embedded on this runtime.
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

    def _vector_cache_keys(
        self,
        texts: Sequence[str],
        dimensions: int,
    ) -> dict[str, tuple[str, str, str, int, str]]:
        """Build each per-text cache key once for the current embed call."""

        model_id, device, revision = self._backend_key()
        prefix = (model_id, device, revision, dimensions)
        return {
            text: (*prefix, _text_digest(text))
            for text in dict.fromkeys(texts)
        }

    def _load_backend(self) -> _EmbeddingBackend:
        model_id, device, revision = self._backend_key()
        require_dense_retrieval_device(
            device,
            role=self.config.role,
            model_id=model_id,
            backend="embedding",
        )
        require_package("sentence-transformers", minimum="3.0.0")
        from sentence_transformers import SentenceTransformer

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
        cache_active = _cache_enabled()
        cache_keys = (
            self._vector_cache_keys(cleaned, dimensions) if cache_active else {}
        )
        vectors_by_text: dict[str, tuple[float, ...]] = {}
        missing: list[str] = []
        seen_missing: set[str] = set()

        if cache_active:
            with _VECTOR_CACHE_LOCK:
                for text in cleaned:
                    cache_key = cache_keys[text]
                    cached = _VECTOR_CACHE.get(cache_key)
                    if cached is not None:
                        _VECTOR_CACHE.move_to_end(cache_key)
                        vectors_by_text[text] = cached
                    elif text not in seen_missing:
                        seen_missing.add(text)
                        missing.append(text)
        else:
            for text in cleaned:
                if text not in seen_missing:
                    seen_missing.add(text)
                    missing.append(text)

        if not missing:
            print(
                "retrieval embedding: encode cache hit",
                f"batch={len(cleaned)}",
                file=sys.stderr,
                flush=True,
            )
            return [list(vectors_by_text[text]) for text in cleaned]

        try:
            backend = self._ensure_backend()
            started = time.monotonic()
            print(
                "retrieval embedding: encode start",
                f"batch={len(cleaned)}",
                f"candidate_missing={len(missing)}",
                file=sys.stderr,
                flush=True,
            )
            with backend.lock:
                if cache_active:
                    with _VECTOR_CACHE_LOCK:
                        still_missing: list[str] = []
                        for text in missing:
                            cache_key = cache_keys[text]
                            cached = _VECTOR_CACHE.get(cache_key)
                            if cached is not None:
                                _VECTOR_CACHE.move_to_end(cache_key)
                                vectors_by_text[text] = cached
                            else:
                                still_missing.append(text)
                    missing = still_missing
                if not missing:
                    return [list(vectors_by_text[text]) for text in cleaned]
                vectors = backend.model.encode(
                    missing,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    truncate_dim=dimensions,
                    show_progress_bar=False,
                )
            computed = tuple(tuple(float(value) for value in row) for row in vectors)
            if len(computed) != len(missing):
                raise RuntimeError(
                    "Embedding backend returned a different number of vectors than inputs."
                )
            print(
                "retrieval embedding: encode done",
                f"batch={len(cleaned)}",
                f"computed={len(missing)}",
                f"elapsed={time.monotonic() - started:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            for text, vector in zip(missing, computed, strict=True):
                vectors_by_text[text] = vector

            if cache_active:
                with _VECTOR_CACHE_LOCK:
                    for text, vector in zip(missing, computed, strict=True):
                        cache_key = cache_keys[text]
                        _VECTOR_CACHE[cache_key] = vector
                        _VECTOR_CACHE.move_to_end(cache_key)
                    cache_limit = _result_cache_limit()
                    while len(_VECTOR_CACHE) > cache_limit:
                        _VECTOR_CACHE.popitem(last=False)
            return [list(vectors_by_text[text]) for text in cleaned]
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc


EmbeddingAdapter.embed._mmm_cached_embedding_model = True  # type: ignore[attr-defined]
