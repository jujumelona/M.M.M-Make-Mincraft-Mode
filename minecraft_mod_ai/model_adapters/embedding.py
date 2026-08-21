from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import AdapterConfig, ModelBackendError, require_package


@dataclass
class _EmbeddingBackend:
    model: Any
    lock: threading.RLock = field(default_factory=threading.RLock)


_BACKEND_CACHE_LOCK = threading.RLock()
_BACKEND_CACHE: dict[tuple[str, str, str], _EmbeddingBackend] = {}


def _cache_enabled() -> bool:
    raw = os.environ.get("MMM_CPU_RETRIEVAL_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class EmbeddingAdapter:
    """Sentence embedding adapter with one shared CPU backend per model profile.

    Retrieval adapters are short-lived at the router boundary, so instance-only
    caching still reloads the same model on every RAG batch. The backend cache is
    therefore owned here, next to the model lifecycle it controls, rather than by a
    bootstrap monkey-patch layer.
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
        current_context = int(getattr(model, "max_seq_length", configured_context) or configured_context)
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

        try:
            backend = self._ensure_backend()
            dimensions = int(self.config.extra.get("dimensions", 512))
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
            return [[float(value) for value in row] for row in vectors]
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc


EmbeddingAdapter.embed._mmm_cached_embedding_model = True  # type: ignore[attr-defined]
