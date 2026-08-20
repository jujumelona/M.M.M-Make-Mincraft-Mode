from __future__ import annotations

"""Keep CPU retrieval models resident for one ModelRouter lifetime.

RAG indexing calls ``router.embed`` once per bounded chunk batch and tool selection
may rerank repeatedly. Constructing a fresh adapter for every call defeats the
adapter-level lazy model cache and repeatedly reloads the same Hugging Face weights.
This late contract preserves the public router API while making adapter ownership
explicit and per-router rather than global.
"""

import threading
from functools import wraps
from typing import Any

_INIT_LOCK = threading.RLock()


def _cache_for(router: Any) -> tuple[threading.RLock, dict[tuple[str, str], Any]]:
    lock = getattr(router, "_mmm_retrieval_adapter_lock", None)
    cache = getattr(router, "_mmm_retrieval_adapters", None)
    if lock is not None and isinstance(cache, dict):
        return lock, cache
    with _INIT_LOCK:
        lock = getattr(router, "_mmm_retrieval_adapter_lock", None)
        cache = getattr(router, "_mmm_retrieval_adapters", None)
        if lock is None:
            lock = threading.RLock()
            setattr(router, "_mmm_retrieval_adapter_lock", lock)
        if not isinstance(cache, dict):
            cache = {}
            setattr(router, "_mmm_retrieval_adapters", cache)
        return lock, cache


def _resident_adapter(router: Any, *, kind: str, role: str, config: Any, factory: Any) -> Any:
    lock, cache = _cache_for(router)
    key = (kind, role)
    with lock:
        adapter = cache.get(key)
        if adapter is None or getattr(adapter, "config", None) != config:
            adapter = factory(config)
            cache[key] = adapter
        return adapter


def install(*, model_router_module: Any) -> None:
    cls = model_router_module.ModelRouter

    current_embed = cls.embed
    if getattr(current_embed, "_mmm_resident_embedding_adapter", False) is not True:
        @wraps(current_embed)
        def embed_resident(self: Any, texts: Any, role: str = "embedding") -> Any:
            config = self.registry.role(self.profile, role)
            if config.adapter != "embedding":
                raise model_router_module.ModelConfigurationError(
                    f"Role {role!r} does not expose an embedding adapter."
                )
            adapter = _resident_adapter(
                self,
                kind="embedding",
                role=role,
                config=config,
                factory=model_router_module.EmbeddingAdapter,
            )
            return adapter.embed(texts)

        embed_resident._mmm_resident_embedding_adapter = True  # type: ignore[attr-defined]
        embed_resident.__wrapped__ = current_embed  # type: ignore[attr-defined]
        cls.embed = embed_resident

    current_rerank = cls.rerank
    if getattr(current_rerank, "_mmm_resident_reranker_adapter", False) is not True:
        @wraps(current_rerank)
        def rerank_resident(
            self: Any,
            query: str,
            documents: Any,
            *,
            role: str = "reranker",
            instruction: str = (
                "Retrieve the Minecraft modding evidence that directly answers the query "
                "for the caller-selected platform target. Do not prefer or infer a different "
                "Minecraft version or mapping namespace."
            ),
        ) -> Any:
            config = self.registry.role(self.profile, role)
            if config.adapter != "reranker":
                raise model_router_module.ModelConfigurationError(
                    f"Role {role!r} does not expose a reranker adapter."
                )
            adapter = _resident_adapter(
                self,
                kind="reranker",
                role=role,
                config=config,
                factory=model_router_module.RerankerAdapter,
            )
            return adapter.score(query, documents, instruction=instruction)

        rerank_resident._mmm_resident_reranker_adapter = True  # type: ignore[attr-defined]
        rerank_resident.__wrapped__ = current_rerank  # type: ignore[attr-defined]
        cls.rerank = rerank_resident


__all__ = ["install"]
