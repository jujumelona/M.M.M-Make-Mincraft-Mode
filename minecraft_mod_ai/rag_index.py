from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .model_router import ModelRouter


_ALLOWED_SUFFIXES = frozenset(
    {
        ".java",
        ".json",
        ".gradle",
        ".kts",
        ".properties",
        ".md",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".mcfunction",
        ".snbt",
    }
)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.$:/-]*|[가-힣]{2,}|\d+(?:\.\d+)+")


@dataclass(frozen=True)
class RAGChunk:
    chunk_id: str
    source_path: str
    text: str
    start_line: int
    end_line: int
    sha256: str
    metadata: dict[str, Any]
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True)
class RAGHit:
    chunk_id: str
    source_path: str
    start_line: int
    end_line: int
    score: float
    lexical_score: float
    semantic_score: float
    reranker_score: float
    text: str
    metadata: dict[str, Any]


class ProjectRAGIndex:
    """Version- and license-aware code/document index.

    Lexical search is always available. Embedding and reranking are explicit
    options so CPU CI never pretends a model was loaded.
    """

    schema_version = "mmm/project-rag-index-v1"

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path).expanduser().resolve()

    def build(
        self,
        roots: Sequence[str | Path],
        *,
        metadata: dict[str, Any],
        router: ModelRouter | None = None,
        semantic: bool = False,
        max_files: int = 5000,
    ) -> dict[str, Any]:
        _validate_metadata(metadata)
        files = list(_iter_files(roots, max_files=max_files))
        chunks: list[RAGChunk] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            for start, end, chunk_text in _chunk_text(path, text):
                digest = hashlib.sha256(
                    (str(path) + "\0" + str(start) + "\0" + chunk_text).encode("utf-8")
                ).hexdigest()
                chunks.append(
                    RAGChunk(
                        chunk_id=f"sha256:{digest}",
                        source_path=str(path),
                        text=chunk_text,
                        start_line=start,
                        end_line=end,
                        sha256=f"sha256:{hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()}",
                        metadata=dict(metadata),
                    )
                )
        if semantic and chunks:
            if router is None:
                raise ValueError("semantic=True requires a ModelRouter.")
            vectors = router.embed([chunk.text for chunk in chunks])
            chunks = [
                RAGChunk(**{**asdict(chunk), "embedding": tuple(vector)})
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "chunks": [
                {**asdict(chunk), "embedding": list(chunk.embedding)}
                for chunk in chunks
            ],
        }
        self.index_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "schema_version": "mmm/rag-build-result-v1",
            "index_path": str(self.index_path),
            "files_indexed": len(files),
            "chunks_indexed": len(chunks),
            "semantic_embeddings": semantic,
            "index_sha256": _sha256(self.index_path),
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        router: ModelRouter | None = None,
        semantic: bool = False,
        rerank: bool = False,
        required_metadata: dict[str, Any] | None = None,
    ) -> list[RAGHit]:
        query = query.strip()
        if not query:
            raise ValueError("RAG query must not be empty.")
        if not 1 <= limit <= 50:
            raise ValueError("RAG limit must be between 1 and 50.")
        chunks = self._load()
        if required_metadata:
            chunks = [
                chunk
                for chunk in chunks
                if all(chunk.metadata.get(key) == value for key, value in required_metadata.items())
            ]
        query_tokens = set(_tokens(query))
        semantic_vector: list[float] | None = None
        if semantic:
            if router is None:
                raise ValueError("semantic=True requires a ModelRouter.")
            semantic_vector = router.embed([query])[0]
        scored: list[tuple[float, float, float, RAGChunk]] = []
        for chunk in chunks:
            chunk_tokens = set(_tokens(chunk.text))
            intersection = len(query_tokens & chunk_tokens)
            exact_bonus = sum(
                2.0 for token in query_tokens if token and token.lower() in chunk.text.lower()
            )
            lexical = (intersection / max(1, len(query_tokens))) + exact_bonus
            semantic_score = (
                _cosine(semantic_vector, chunk.embedding)
                if semantic_vector is not None and chunk.embedding
                else 0.0
            )
            scored.append((lexical + semantic_score, lexical, semantic_score, chunk))
        scored.sort(key=lambda row: (-row[0], row[3].source_path, row[3].start_line))
        candidates = scored[: max(limit * 5, limit)]
        reranker_scores = [0.0] * len(candidates)
        if rerank and candidates:
            if router is None:
                raise ValueError("rerank=True requires a ModelRouter.")
            reranker_scores = router.rerank(
                query,
                [candidate[3].text for candidate in candidates],
            )
        hits: list[RAGHit] = []
        for candidate, reranker_score in zip(candidates, reranker_scores, strict=True):
            _, lexical, semantic_score, chunk = candidate
            final = lexical + semantic_score + (2.0 * reranker_score)
            hits.append(
                RAGHit(
                    chunk_id=chunk.chunk_id,
                    source_path=chunk.source_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=round(final, 6),
                    lexical_score=round(lexical, 6),
                    semantic_score=round(semantic_score, 6),
                    reranker_score=round(reranker_score, 6),
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.source_path, hit.start_line))
        return hits[:limit]

    def _load(self) -> list[RAGChunk]:
        if not self.index_path.is_file():
            raise FileNotFoundError(f"RAG index not found: {self.index_path}")
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != self.schema_version:
            raise ValueError("Unsupported RAG index schema.")
        result: list[RAGChunk] = []
        for item in raw.get("chunks", []):
            result.append(
                RAGChunk(
                    chunk_id=item["chunk_id"],
                    source_path=item["source_path"],
                    text=item["text"],
                    start_line=int(item["start_line"]),
                    end_line=int(item["end_line"]),
                    sha256=item["sha256"],
                    metadata=dict(item["metadata"]),
                    embedding=tuple(float(value) for value in item.get("embedding", [])),
                )
            )
        return result


def _validate_metadata(metadata: dict[str, Any]) -> None:
    required = {
        "minecraft_version",
        "loader",
        "mapping_namespace",
        "java_version",
        "license",
        "source_commit",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"RAG metadata is missing: {sorted(missing)}")
    if metadata["minecraft_version"] != "1.20.1" or metadata["loader"] != "fabric":
        raise ValueError("This index accepts only the pinned Fabric 1.20.1 corpus.")
    if metadata["mapping_namespace"] not in {"yarn", "intermediary", "official"}:
        raise ValueError("Unsupported mapping namespace.")
    if not str(metadata["license"]).strip() or not str(metadata["source_commit"]).strip():
        raise ValueError("RAG source license and commit are required.")


def _iter_files(
    roots: Sequence[str | Path],
    *,
    max_files: int,
) -> Iterable[Path]:
    found = 0
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if found >= max_files:
                raise ValueError(f"RAG file limit exceeded: {max_files}")
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in _ALLOWED_SUFFIXES
                and path.stat().st_size <= 2 * 1024 * 1024
            ):
                found += 1
                yield path


def _chunk_text(path: Path, text: str) -> Iterable[tuple[int, int, str]]:
    lines = text.splitlines()
    if path.suffix.lower() in {".java", ".gradle", ".kts", ".mcfunction", ".snbt"}:
        size, overlap = 160, 24
    else:
        size, overlap = 100, 15
    start = 0
    while start < len(lines):
        end = min(len(lines), start + size)
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            yield start + 1, end, chunk
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
