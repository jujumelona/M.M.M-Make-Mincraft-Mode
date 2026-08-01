from __future__ import annotations

from typing import Sequence

from .base import AdapterConfig, ModelBackendError, require_package


class EmbeddingAdapter:
    """On-demand Qwen/compatible sentence embedding adapter.

    The T4 profile intentionally defaults this role to CPU so retrieval does not
    compete with the planner or coder for GPU residency.
    """

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [str(text).strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Embedding input must contain non-empty strings.")
        try:
            require_package("sentence-transformers", minimum="3.0.0")
            from sentence_transformers import SentenceTransformer

            device = str(self.config.extra.get("device", "cpu"))
            dimensions = int(self.config.extra.get("dimensions", 512))
            load_options: dict[str, object] = {
                "device": device,
                "trust_remote_code": False,
            }
            revision = str(self.config.extra.get("revision", "")).strip()
            if revision:
                load_options["revision"] = revision
            model = SentenceTransformer(self.config.model_id, **load_options)
            vectors = model.encode(
                cleaned,
                normalize_embeddings=True,
                convert_to_numpy=True,
                truncate_dim=dimensions,
                show_progress_bar=False,
            )
            return [[float(value) for value in row] for row in vectors]
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc
