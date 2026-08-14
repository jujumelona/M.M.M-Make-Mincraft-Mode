from __future__ import annotations

"""Direct low-coverage q1-vector retrieval over durable project embeddings."""

import heapq
import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SQLITE_HEADER = b"SQLite format 3\x00"


def _is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(_SQLITE_HEADER)) == _SQLITE_HEADER
    except OSError:
        return False


def _vector(raw: Any) -> list[float]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    try:
        values = [float(item) for item in raw]
    except (TypeError, ValueError):
        return []
    return values if values and all(math.isfinite(item) for item in values) else []


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    width = min(len(left), len(right))
    if width <= 0:
        return 0.0
    dot = sum(float(left[index]) * float(right[index]) for index in range(width))
    left_norm = math.sqrt(sum(float(left[index]) ** 2 for index in range(width)))
    right_norm = math.sqrt(sum(float(right[index]) ** 2 for index in range(width)))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _metadata_matches(metadata: Mapping[str, Any], required: Mapping[str, Any] | None) -> bool:
    if not required:
        return True
    for key, expected in required.items():
        actual = metadata.get(key)
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping) or not _metadata_matches(actual, expected):
                return False
        elif actual != expected:
            return False
    return True


def _query_terms(query: str) -> set[str]:
    import re

    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.$:/-]{1,127}|[가-힣]{2,}", query)
        if len(token) > 1
    }


def direct_centroid_vector_search(
    index_path: str | Path,
    *,
    query: str,
    q1_vector: Sequence[float],
    router: Any,
    limit: int = 8,
    required_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Retrieve directly with q1 against stored chunk vectors, then rerank.

    Returns ``None`` when the durable index has no semantic vectors; callers can
    safely fall back to ordinary routed retrieval without pretending adaptation ran.
    """

    target = Path(index_path).expanduser().resolve()
    if not target.is_file() or not _is_sqlite(target) or not q1_vector:
        return None
    connection = sqlite3.connect(str(target))
    connection.row_factory = sqlite3.Row
    try:
        meta_rows = connection.execute("SELECT key, value FROM index_meta").fetchall()
        meta = {str(row[0]): str(row[1]) for row in meta_rows}
        if meta.get("semantic_embeddings") != "1":
            return None
        try:
            index_metadata = json.loads(meta.get("metadata", "{}"))
        except json.JSONDecodeError:
            return None
        if not isinstance(index_metadata, Mapping) or not _metadata_matches(index_metadata, required_metadata):
            return None

        candidate_limit = max(limit * 6, 32)
        heap: list[tuple[float, str, int, dict[str, Any]]] = []
        considered = 0
        for row in connection.execute(
            """
            SELECT chunk_id, source_path, text, start_line, end_line,
                   sha256, embedding
            FROM chunks
            ORDER BY source_path, start_line, chunk_id
            """
        ):
            vector = _vector(row["embedding"])
            if not vector:
                continue
            considered += 1
            score = _cosine(q1_vector, vector)
            if not math.isfinite(score):
                continue
            hit = {
                "chunk_id": str(row["chunk_id"]),
                "source_path": str(row["source_path"]),
                "start_line": int(row["start_line"]),
                "end_line": int(row["end_line"]),
                "sha256": str(row["sha256"]),
                "text": str(row["text"]),
                "metadata": dict(index_metadata),
                "lexical_score": 0.0,
                "semantic_score": round(score, 6),
                "reranker_score": 0.0,
                "relation_score": 0.0,
                "score": round(score, 6),
            }
            key = (score, str(row["chunk_id"]), int(row["start_line"]), hit)
            if len(heap) < candidate_limit:
                heapq.heappush(heap, key)
            elif key[:3] > heap[0][:3]:
                heapq.heapreplace(heap, key)

        candidates = [item[3] for item in sorted(heap, key=lambda item: (-item[0], item[1], item[2]))]
        if not candidates:
            return None
        try:
            reranked = router.rerank(query, [item["text"] for item in candidates])
        except Exception:
            reranked = []
        if len(reranked) == len(candidates):
            for hit, score in zip(candidates, reranked, strict=True):
                numeric = float(score)
                if math.isfinite(numeric):
                    hit["reranker_score"] = round(numeric, 6)
                    hit["score"] = round(float(hit["semantic_score"]) + 2.0 * numeric, 6)
            candidates.sort(key=lambda item: (-float(item["score"]), item["source_path"], item["start_line"]))

        hits = candidates[:limit]
        terms = _query_terms(query)
        covered = {
            term
            for term in terms
            if any(term in str(hit["text"]).casefold() for hit in hits)
        }
        coverage = len(covered) / max(1, len(terms)) if terms else 1.0
        relevance = max((float(hit["score"]) for hit in hits), default=0.0)
        return {
            "schema_version": "mmm/code-rag-result-v1",
            "query": query,
            "hits": hits,
            "receipt": {
                "schema_version": "mmm/rag-search-receipt-v1",
                "query": query,
                "route": "centroid_vector",
                "corrected_query": None,
                "correction_applied": False,
                "lexical_backend": "none",
                "semantic_requested": True,
                "semantic_used": True,
                "rerank_requested": True,
                "rerank_used": bool(reranked and len(reranked) == len(candidates)),
                "candidates_considered": considered,
                "relation_expansions": 0,
                "result_count": len(hits),
                "query_terms": sorted(terms),
                "covered_terms": sorted(covered),
                "missing_terms": sorted(terms - covered),
                "coverage_score": round(coverage, 6),
                "relevance_score": round(relevance, 6),
                "required_metadata": dict(required_metadata or {}),
                "warnings": [] if hits else ["no_relevant_chunks"],
                "adaptation": "q0_topk_centroid_q1_direct_vector",
            },
            "retrieval_mode": "centroid-q1-vector+rerank",
            "centroid_adaptation": True,
            "centroid_vector_direct": True,
        }
    finally:
        connection.close()


__all__ = ["direct_centroid_vector_search"]
