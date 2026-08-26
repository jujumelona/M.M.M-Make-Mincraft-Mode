from __future__ import annotations

"""Keep CPU retrieval models resident and bound expensive reranker work.

RAG indexing calls ``router.embed`` once per bounded chunk batch and retrieval may
rerank repeatedly. Constructing a fresh adapter for every call defeats the adapter-level
lazy model cache and repeatedly reloads the same Hugging Face weights. Broad lexical or
ANN candidate discovery is cheap and may remain large, but a local CPU cross-encoder
must only score the already-ranked shortlist rather than hundreds of repository chunks.
"""

import os
import threading
from collections.abc import Sequence
from functools import wraps
from typing import Any

_INIT_LOCK = threading.RLock()
_DEFAULT_LOCAL_RERANK_DOCUMENTS = 8
_MIN_LOCAL_RERANK_DOCUMENTS = 1
_MAX_LOCAL_RERANK_DOCUMENTS = 64


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
            router._mmm_retrieval_adapter_lock = lock
        if not isinstance(cache, dict):
            cache = {}
            router._mmm_retrieval_adapters = cache
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


def _local_rerank_document_limit() -> int:
    raw = os.environ.get(
        "MMM_LOCAL_RERANK_DOCUMENTS",
        str(_DEFAULT_LOCAL_RERANK_DOCUMENTS),
    ).strip()
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_LOCAL_RERANK_DOCUMENTS
    return max(_MIN_LOCAL_RERANK_DOCUMENTS, min(value, _MAX_LOCAL_RERANK_DOCUMENTS))


def _bounded_rerank_scores(
    adapter: Any,
    query: str,
    documents: Sequence[str],
    *,
    instruction: str,
    local_cpu: bool,
) -> list[float]:
    """Score only the pre-ranked local shortlist while preserving caller cardinality.

    RAG callers present documents in cheap lexical/semantic rank order. When that list
    is larger than the CPU budget, scoring the entire tail adds minutes of latency while
    contributing progressively weaker candidates. The unscored tail receives a strict
    floor below every scored item, so downstream zip/cardinality contracts stay intact
    and no unseen tail item is accidentally promoted by the cross-encoder stage.
    """

    values = tuple(documents)
    if not values:
        return []
    limit = _local_rerank_document_limit() if local_cpu else len(values)
    selected = values[:limit]
    scores = [
        float(score)
        for score in adapter.score(
            query,
            selected,
            instruction=instruction,
        )
    ]
    if len(scores) != len(selected):
        raise ValueError("Reranker returned the wrong score count.")
    if len(selected) == len(values):
        return scores
    floor = (min(scores) - 1.0) if scores else -1.0
    return [*scores, *([floor] * (len(values) - len(selected)))]


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
            return _bounded_rerank_scores(
                adapter,
                query,
                documents,
                instruction=instruction,
                local_cpu=(
                    str(getattr(config, "provider", "")) == "local"
                    and not bool(getattr(config, "exclusive_gpu", False))
                ),
            )

        rerank_resident._mmm_resident_reranker_adapter = True  # type: ignore[attr-defined]
        rerank_resident.__wrapped__ = current_rerank  # type: ignore[attr-defined]
        cls.rerank = rerank_resident


__all__ = ["install"]
