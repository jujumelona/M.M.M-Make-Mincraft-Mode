from __future__ import annotations

"""Training-free retrieval adaptation for low-coverage small-model RAG.

The adapter derives a local centroid from the first-pass hit texts using the
already configured embedding model, blends it with the original query vector,
and retrieves again without changing model weights.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _vector(value: Any) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return []
    return []


def _embedding_rows(router: Any, texts: Sequence[str]) -> list[list[float]]:
    """Embed records in one batch, with compatibility for legacy scalar adapters.

    Production ``ModelRouter.embed`` is a ``Sequence[str] -> list[list[float]]`` API.
    Batching is the authoritative path and prevents a string from being interpreted
    as a sequence of characters. A few lightweight/legacy adapters expose the older
    scalar ``str -> list[float]`` shape; only when the batch contract is unavailable
    do we fall back to one scalar call per record.
    """

    cleaned = [str(text).strip() for text in texts if str(text).strip()]
    if not cleaned:
        return []

    raw: Any = None
    try:
        raw = router.embed(cleaned)
    except Exception:
        raw = None

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        rows = [_vector(row) for row in raw]
        if len(rows) == len(cleaned) and all(rows):
            return rows
        if len(cleaned) == 1:
            scalar = _vector(raw)
            if scalar:
                return [scalar]

    legacy_rows: list[list[float]] = []
    for text in cleaned:
        try:
            row = _vector(router.embed(text))
        except Exception:
            return []
        if not row:
            return []
        legacy_rows.append(row)
    return legacy_rows


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in vector))
    if norm <= 0.0:
        return [float(item) for item in vector]
    return [float(item) / norm for item in vector]


def local_centroid(router: Any, texts: Sequence[str]) -> list[float]:
    candidates = [str(text).strip() for text in texts[:8] if str(text).strip()]
    vectors = [_normalize(row) for row in _embedding_rows(router, candidates)]
    if not vectors:
        return []
    width = min(len(item) for item in vectors)
    centroid = [sum(item[index] for item in vectors) / len(vectors) for index in range(width)]
    return _normalize(centroid)


def adapt_query_vector(
    router: Any,
    query: str,
    hit_texts: Sequence[str],
    *,
    alpha: float = 0.65,
) -> list[float]:
    query_rows = _embedding_rows(router, [query])
    if not query_rows:
        return []
    query_vector = query_rows[0]
    centroid = local_centroid(router, hit_texts)
    if not centroid:
        return []
    width = min(len(query_vector), len(centroid))
    query_norm = _normalize(query_vector[:width])
    centroid = centroid[:width]
    blended = [
        alpha * query_norm[index] + (1.0 - alpha) * centroid[index]
        for index in range(width)
    ]
    return _normalize(blended)


def extract_hit_texts(result: Mapping[str, Any]) -> list[str]:
    values = result.get("results")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        values = result.get("hits")
    texts: list[str] = []
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for item in values[:8]:
            if isinstance(item, Mapping):
                for key in ("text", "content", "chunk", "excerpt"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        texts.append(value[:6000])
                        break
    return texts


__all__ = [
    "_embedding_rows",
    "adapt_query_vector",
    "extract_hit_texts",
    "local_centroid",
]
