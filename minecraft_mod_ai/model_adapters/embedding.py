from __future__ import annotations

import sys
import threading
import time
from typing import Any, Sequence

from .base import AdapterConfig, ModelBackendError, require_package


class EmbeddingAdapter:
    """Lazy-resident Qwen/compatible sentence embedding adapter.

    The T4 profile intentionally defaults this role to CPU so retrieval does not
    compete with the planner or coder for GPU residency. A router may call this
    adapter once per RAG batch, so the model must remain resident for the adapter's
    lifetime rather than being reconstructed for every batch.
    """

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        require_package("sentence-transformers", minimum="3.0.0")
        from sentence_transformers import SentenceTransformer

        device = str(self.config.extra.get("device", "cpu"))
        load_options: dict[str, object] = {
            "device": device,
            "trust_remote_code": False,
        }
        revision = str(self.config.extra.get("revision", "")).strip()
        if revision:
            load_options["revision"] = revision
        started = time.monotonic()
        print(
            "retrieval embedding: model load start",
            f"model={self.config.model_id}",
            f"device={device}",
            file=sys.stderr,
            flush=True,
        )
        model = SentenceTransformer(self.config.model_id, **load_options)
        self._model = model
        print(
            "retrieval embedding: model load done",
            f"elapsed={time.monotonic() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [str(text).strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Embedding input must contain non-empty strings.")
        try:
            with self._lock:
                model = self._ensure_model()
                dimensions = int(self.config.extra.get("dimensions", 512))
                started = time.monotonic()
                print(
                    "retrieval embedding: encode start",
                    f"batch={len(cleaned)}",
                    file=sys.stderr,
                    flush=True,
                )
                vectors = model.encode(
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
